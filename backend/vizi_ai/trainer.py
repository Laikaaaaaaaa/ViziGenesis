"""
ViziGenesis vizi-o1 — Training Engine
======================================
Handles the complete training lifecycle:

  Phase 1: Per-stock specialist warm-up (optional, **parallel**)
    Train lightweight heads on individual stocks to bootstrap
    good per-stock representations before full-universe training.
    Multiple stocks are processed as a mega-batch on GPU for
    maximum throughput (auto-sized to available VRAM).

  Phase 2: Full-universe multi-modal training
    Stream all stocks through the ViziMarketTransformer with
    multi-task loss. This is the main training phase.

  Phase 3: Meta-model distillation (optional)
    Combine per-stock specialist knowledge into the unified model
    via knowledge distillation on the specialist predictions.

Design rationale
────────────────
•  AMP (automatic mixed precision) halves VRAM usage on RTX 4090,
   enabling larger batch sizes and sequence lengths.

•  Gradient accumulation simulates large effective batches when
   physical batch size is limited by VRAM.

•  Cosine annealing with warm-up is the most reliable scheduler
   for transformer training: the warm-up prevents early divergence,
   and the cosine decay finds flat minima (better generalisation).

•  Multi-task loss weighting uses uncertainty-based automatic
   weighting (Kendall et al. 2018): each task has a learnable
   log-variance σ²; the loss for task i is  L_i / (2σ²_i) + log(σ_i).
   This prevents any single task from dominating gradients.

Upgrades (v2)
──────────────
•  **Training Monitor**: Rich terminal display w/ progress bars, ETA,
   GPU utilization, VRAM usage, throughput (samples/sec, stocks/sec).

•  **H100 Optimizations**: TF32 matmul, fused AdamW, bf16 autodetect,
   CUDA memory-efficient attention when available.

•  **Pre-training Time Estimator**: Profiles forward/backward pass
   on real data to estimate total wall-clock time before committing.

•  **Self-Improvement Loop**: Tracks metric history across runs and
   auto-suggests hyperparameter adjustments for the next run.
"""
from __future__ import annotations

import json, logging, math, os, shutil, time
import importlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from backend.vizi_ai.model import ModelConfig, ViziMarketTransformer
from backend.vizi_ai.data_pipeline import (
    DataConfig, create_dataloader, discover_symbols,
    N_FUNDAMENTAL_FEATURES, FRED_KEYS,
)

logger = logging.getLogger("vizi_ai.trainer")

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


# ═══════════════════════════════════════════════════════════════
#  Training Monitor — rich terminal progress display
# ═══════════════════════════════════════════════════════════════
class TrainingMonitor:
    """
    Real-time training dashboard printed to terminal.

    Shows:
    - Phase / epoch / step progress with bar
    - ETA (estimated time to completion)
    - Loss breakdown (total + per-task)
    - GPU utilization and VRAM usage
    - Throughput (samples/sec)
    - Best validation metrics
    """
    def __init__(self, total_steps: int, total_epochs: int, device: torch.device):
        self.total_steps = max(1, total_steps)
        self.total_epochs = total_epochs
        self.device = device
        self.start_time = time.time()
        self.step_times: deque = deque(maxlen=100)  # recent step durations
        self.last_step_time = time.time()
        self.best_val_loss = float("inf")
        self.best_dir_acc = 0.0
        self.best_sharpe = 0.0
        self._has_pynvml = False
        self._nvml_handle = None
        self._pynvml = None
        try:
            nvml = importlib.import_module("pynvml")
            nvml.nvmlInit()
            self._nvml_handle = nvml.nvmlDeviceGetHandleByIndex(
                device.index if device.index is not None else 0
            )
            self._pynvml = nvml
            self._has_pynvml = True
        except Exception:
            pass

    def _gpu_stats(self) -> Dict[str, str]:
        """Gather GPU utilization & VRAM via pynvml or torch.cuda."""
        stats: Dict[str, str] = {}
        if self.device.type != "cuda":
            return stats

        # VRAM from torch
        alloc_gb = torch.cuda.memory_allocated(self.device) / (1 << 30)
        reserved_gb = torch.cuda.memory_reserved(self.device) / (1 << 30)
        total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1 << 30)
        stats["vram"] = f"{alloc_gb:.1f}/{total_gb:.0f}GB"
        stats["vram_pct"] = f"{alloc_gb / total_gb * 100:.0f}%"

        # GPU utilization from pynvml
        if self._has_pynvml and self._pynvml is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                stats["gpu_util"] = f"{util.gpu}%"
                stats["mem_util"] = f"{util.memory}%"
                temp = self._pynvml.nvmlDeviceGetTemperature(self._nvml_handle, 0)
                stats["temp"] = f"{temp}°C"
                power = self._pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000
                stats["power"] = f"{power:.0f}W"
            except Exception:
                pass
        return stats

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"

    def _progress_bar(self, current: int, total: int, width: int = 30) -> str:
        filled = int(width * current / max(1, total))
        bar = "█" * filled + "░" * (width - filled)
        pct = current / max(1, total) * 100
        return f"[{bar}] {pct:.1f}%"

    def step(self, step: int, epoch: int, loss_info: Dict[str, float],
             samples_in_batch: int = 0):
        """Called after each optimizer step."""
        now = time.time()
        step_dt = now - self.last_step_time
        self.step_times.append(step_dt)
        self.last_step_time = now

        # Only print every 50 steps to avoid spam
        if step % 50 != 0 and step != 1:
            return

        elapsed = now - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - step
        eta = avg_step_time * remaining_steps

        throughput = samples_in_batch / max(step_dt, 1e-6)

        gpu = self._gpu_stats()

        bar = self._progress_bar(step, self.total_steps)
        loss_total = loss_info.get("loss_total", 0)
        loss_dir = loss_info.get("loss_direction", 0)
        loss_ret = loss_info.get("loss_ret_1d", 0)

        parts = [
            f"\r  {bar}",
            f"Step {step}/{self.total_steps}",
            f"Ep {epoch}/{self.total_epochs}",
            f"Loss={loss_total:.4f}",
            f"(dir={loss_dir:.3f} ret1d={loss_ret:.4f})",
            f"LR={loss_info.get('lr', 0):.2e}",
            f"{throughput:.0f} samp/s",
        ]
        if gpu.get("vram"):
            parts.append(f"VRAM={gpu['vram']}")
        if gpu.get("gpu_util"):
            parts.append(f"GPU={gpu['gpu_util']}")
        if gpu.get("temp"):
            parts.append(f"T={gpu['temp']}")

        parts.append(f"ETA={self._format_time(eta)}")
        parts.append(f"Elapsed={self._format_time(elapsed)}")

        line = " | ".join(parts)
        print(line, end="", flush=True)

    def validation(self, step: int, val_loss: float, metrics: Dict[str, float]):
        """Called after each validation."""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        dir_acc = metrics.get("direction_accuracy", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        self.best_dir_acc = max(self.best_dir_acc, dir_acc)
        self.best_sharpe = max(self.best_sharpe, sharpe)

        print()  # newline after progress bar
        logger.info(
            "  ✦ Validation @ step %d: loss=%.4f  dir_acc=%.1f%%  sharpe=%.2f  "
            "regime_acc=%.1f%%  ret1d_IC=%.3f%s",
            step, val_loss,
            dir_acc * 100,
            sharpe,
            metrics.get("regime_accuracy", 0) * 100,
            metrics.get("ret_1d_IC", 0),
            "  ★ BEST" if is_best else "",
        )
        logger.info(
            "    Best so far: loss=%.4f  dir_acc=%.1f%%  sharpe=%.2f",
            self.best_val_loss, self.best_dir_acc * 100, self.best_sharpe,
        )

    def phase_start(self, phase_name: str, n_items: int = 0):
        """Print phase header."""
        print()
        logger.info("═" * 60)
        item_str = f" ({n_items} items)" if n_items else ""
        logger.info("  %s%s", phase_name, item_str)
        logger.info("═" * 60)

    def summary(self, total_steps: int, total_epochs: int):
        """Final training summary."""
        elapsed = time.time() - self.start_time
        print()
        logger.info("═" * 60)
        logger.info("  TRAINING COMPLETE")
        logger.info("═" * 60)
        logger.info("  Total time:    %s", self._format_time(elapsed))
        logger.info("  Total steps:   %d", total_steps)
        logger.info("  Total epochs:  %d", total_epochs)
        logger.info("  Best val_loss: %.4f", self.best_val_loss)
        logger.info("  Best dir_acc:  %.1f%%", self.best_dir_acc * 100)
        logger.info("  Best sharpe:   %.2f", self.best_sharpe)
        avg_step = elapsed / max(1, total_steps)
        logger.info("  Avg step time: %.3fs", avg_step)
        logger.info("═" * 60)


# ═══════════════════════════════════════════════════════════════
#  H100 Optimization — auto-detect and configure
# ═══════════════════════════════════════════════════════════════
def apply_h100_optimizations(device: torch.device) -> Dict[str, bool]:
    """
    Auto-detect H100/A100 and apply optimal CUDA settings.

    Returns dict of which optimizations were applied.
    """
    applied: Dict[str, bool] = {}

    if device.type != "cuda":
        return applied

    gpu_name = torch.cuda.get_device_properties(device).name.lower()
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1 << 30)

    # TF32 for faster matmul on Ampere+ (A100, H100, RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    applied["tf32"] = True

    # cuDNN benchmark for fixed input sizes
    torch.backends.cudnn.benchmark = True
    applied["cudnn_benchmark"] = True

    # bf16 support check (native on H100/A100)
    bf16_ok = torch.cuda.is_bf16_supported()
    applied["bf16_available"] = bf16_ok

    is_h100 = "h100" in gpu_name or "hopper" in gpu_name
    is_a100 = "a100" in gpu_name

    if is_h100 or is_a100:
        applied["high_end_gpu"] = True
        # CUDA memory-efficient attention (Flash Attention)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            applied["sdpa_available"] = True

    logger.info(
        "GPU optimizations: %s (%s, %.0fGB) — %s",
        gpu_name, "H100" if is_h100 else "A100" if is_a100 else "GPU",
        total_gb,
        ", ".join(f"{k}={v}" for k, v in applied.items()),
    )
    return applied


# ═══════════════════════════════════════════════════════════════
#  Pre-training Time Estimator
# ═══════════════════════════════════════════════════════════════
class TimeEstimator:
    """
    Profiles forward/backward pass on real data to estimate total
    training wall-clock time before committing to a full run.
    """

    @staticmethod
    def estimate(
        model: ViziMarketTransformer,
        train_loader,
        train_cfg: "TrainConfig",
        n_profile_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Run a few profiling steps and extrapolate total training time.

        Returns:
            dict with estimated per-step time, total time, and recommendations.
        """
        device = torch.device(train_cfg.device)
        model.to(device)
        model.train()

        loss_fn = UncertaintyMultiTaskLoss(n_tasks=5).to(device)
        optimizer = AdamW(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=train_cfg.lr,
        )
        scaler = _make_grad_scaler(device.type, enabled=train_cfg.use_amp)

        step_times = []
        sample_counts = []

        # Warm-up CUDA, then profile
        batch_iter = iter(train_loader)
        for i in range(n_profile_steps + 2):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break

            # Move to device
            batch_dev: Dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_dev[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dict):
                    batch_dev[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
                else:
                    batch_dev[k] = v

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            with _autocast_ctx(device_type=device.type, enabled=train_cfg.use_amp):
                preds = model(batch_dev)
                task_losses = _compute_losses(preds, batch_dev["targets"])
                total_loss, _ = loss_fn(task_losses)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Skip first 2 iterations (CUDA warm-up)
            if i >= 2:
                step_times.append(dt)
                sample_counts.append(batch["price_seq"].shape[0])

        if not step_times:
            return {"error": "No profiling data — data loader is empty"}

        avg_step_time = sum(step_times) / len(step_times)
        avg_samples = sum(sample_counts) / len(sample_counts)
        samples_per_sec = avg_samples / avg_step_time

        # Estimate total steps per epoch (rough: assume stream yields ~N steps)
        total_time_per_epoch = avg_step_time * (train_cfg.total_steps / max(1, train_cfg.max_epochs))
        total_time = avg_step_time * train_cfg.total_steps

        # VRAM snapshot
        vram_gb = 0.0
        if device.type == "cuda":
            vram_gb = torch.cuda.max_memory_allocated(device) / (1 << 30)
            torch.cuda.reset_peak_memory_stats(device)

        result = {
            "avg_step_time_sec": round(avg_step_time, 4),
            "samples_per_sec": round(samples_per_sec, 1),
            "avg_batch_size": round(avg_samples, 1),
            "peak_vram_gb": round(vram_gb, 2),
            "estimated_total_steps": train_cfg.total_steps,
            "estimated_total_time_sec": round(total_time, 1),
            "estimated_total_time_human": _format_seconds(total_time),
            "estimated_per_epoch_sec": round(total_time_per_epoch, 1),
            "grad_accum_steps": train_cfg.grad_accum_steps,
            "effective_batch_size": train_cfg.effective_batch,
        }

        logger.info("═" * 60)
        logger.info("  PRE-TRAINING TIME ESTIMATE")
        logger.info("═" * 60)
        logger.info("  Avg step time:     %.4fs", avg_step_time)
        logger.info("  Throughput:        %.1f samples/sec", samples_per_sec)
        logger.info("  Peak VRAM:         %.2f GB", vram_gb)
        logger.info("  Estimated total:   %s (%d steps)", result["estimated_total_time_human"], train_cfg.total_steps)
        logger.info("═" * 60)

        return result


def _format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    elif s < 3600:
        return f"{s / 60:.1f}m"
    else:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h}h{m:02d}m"


# ═══════════════════════════════════════════════════════════════
#  Self-Improvement Loop — tracks & recommends across runs
# ═══════════════════════════════════════════════════════════════
class SelfImprover:
    """
    Tracks metric history across training runs and auto-suggests
    hyperparameter adjustments for the next run.

    Saves/loads from ``models/self_improve_history.json``.
    """
    HISTORY_FILE = MODELS_DIR / "self_improve_history.json"

    @classmethod
    def load_history(cls) -> List[Dict]:
        if cls.HISTORY_FILE.exists():
            with open(cls.HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    @classmethod
    def save_run(cls, run_name: str, config: Dict, metrics: Dict):
        """Record a completed run."""
        history = cls.load_history()
        entry = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
        }
        history.append(entry)
        cls.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Self-improvement: saved run '%s' to history (%d total runs)", run_name, len(history))

    @classmethod
    def suggest_next(cls) -> Dict[str, Any]:
        """
        Analyze past runs and suggest hyperparameter changes.

        Heuristics:
        - If direction accuracy is plateauing → increase model capacity
        - If overfitting (train >> val) → increase dropout, reduce LR
        - If underfitting → increase LR, increase epochs
        - If VRAM < 60% on H100 → suggest larger batch size
        """
        history = cls.load_history()
        if len(history) < 2:
            return {"message": "Need at least 2 runs for suggestions"}

        last = history[-1]
        prev = history[-2]
        suggestions: Dict[str, Any] = {}

        last_m = last.get("metrics", {})
        prev_m = prev.get("metrics", {})

        dir_acc = last_m.get("direction_accuracy", 0)
        prev_dir_acc = prev_m.get("direction_accuracy", 0)
        sharpe = last_m.get("sharpe_ratio", 0)
        prev_sharpe = prev_m.get("sharpe_ratio", 0)

        # Plateau detection
        if abs(dir_acc - prev_dir_acc) < 0.005 and dir_acc < 0.6:
            suggestions["increase_capacity"] = (
                "Direction accuracy plateaued at {:.1f}% — consider increasing d_model or n_layers"
                .format(dir_acc * 100)
            )

        # Underfitting
        if dir_acc < 0.52:
            last_cfg = last.get("config", {})
            suggestions["increase_lr"] = (
                "Direction accuracy {:.1f}% is near random — try higher LR or more epochs"
                .format(dir_acc * 100)
            )

        # Sharpe degradation
        if sharpe < prev_sharpe - 0.1:
            suggestions["sharpe_warning"] = (
                "Sharpe dropped from {:.2f} to {:.2f} — possible overfitting, try more regularization"
                .format(prev_sharpe, sharpe)
            )

        # Good progress
        if dir_acc > prev_dir_acc + 0.01 and sharpe > prev_sharpe:
            suggestions["keep_going"] = (
                "Good progress: dir_acc {:.1f}%→{:.1f}%, sharpe {:.2f}→{:.2f} — continue this direction"
                .format(prev_dir_acc * 100, dir_acc * 100, prev_sharpe, sharpe)
            )

        if suggestions:
            logger.info("Self-improvement suggestions:")
            for k, v in suggestions.items():
                logger.info("  • %s: %s", k, v)
        else:
            suggestions["message"] = "No specific suggestions — metrics are within expected range"

        return suggestions


def _autocast_ctx(device_type: str, enabled: bool = True):
    """Compatibility wrapper across PyTorch AMP API versions."""
    if not enabled:
        return nullcontext()

    # Newer API (torch.amp.autocast)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)

    # Older API fallback (torch.cuda.amp.autocast)
    if device_type == "cuda":
        return autocast(enabled=enabled)

    return nullcontext()


def _make_grad_scaler(device_type: str, enabled: bool = True):
    """Compatibility wrapper for GradScaler across PyTorch versions."""
    if not enabled:
        # Keep object creation simple for call sites; disabled scaler is a no-op.
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(device_type, enabled=False)
        return torch.cuda.amp.GradScaler(enabled=False)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        # Preferred API (avoids deprecation warning on newer versions).
        return torch.amp.GradScaler(device_type, enabled=True)

    # Legacy fallback for older PyTorch releases.
    return torch.cuda.amp.GradScaler(enabled=(enabled and device_type == "cuda"))


# ═══════════════════════════════════════════════════════════════
#  Training Configuration
# ═══════════════════════════════════════════════════════════════
class TrainConfig:
    """All training hyperparameters in one place."""
    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # Scheduler
    warmup_steps: int = 500
    total_steps: int = 50_000      # updated dynamically

    # Batch
    batch_size: int = 64           # physical batch per GPU step
    grad_accum_steps: int = 4      # effective batch = 64 × 4 = 256
    effective_batch: int = 256

    # Training
    max_epochs: int = 30
    patience: int = 5              # early stopping patience
    val_every_steps: int = 500     # validate every N training steps

    # Loss weights (initial; auto-balanced via uncertainty)
    loss_w_direction: float = 1.0
    loss_w_ret_1d:    float = 1.0
    loss_w_ret_5d:    float = 0.8
    loss_w_ret_21d:   float = 0.6
    loss_w_regime:    float = 0.5

    # AMP
    use_amp: bool = True

    # Specialist warm-up stability
    specialist_lr_scale: float = 0.1

    # Fused optimizer (faster on H100/A100)
    use_fused_optimizer: bool = False

    # Time estimation before training
    run_time_estimate: bool = False

    # Self-improvement tracking
    self_improve: bool = True

    # Versioning
    run_name: str = "vizi-o1"
    run_tag:  str = ""             # auto-set to timestamp

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    save_dir: Path = MODELS_DIR

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if not self.run_tag:
            self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.effective_batch = self.batch_size * self.grad_accum_steps

        # Auto-detect H100 and bump settings
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name.lower()
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            if total_gb >= 70:  # H100/A100 80GB
                self.use_fused_optimizer = True
                if self.batch_size <= 64:
                    self.batch_size = 128
                    self.effective_batch = self.batch_size * self.grad_accum_steps

    @property
    def run_dir(self) -> Path:
        return self.save_dir / f"{self.run_name}_{self.run_tag}"


# ═══════════════════════════════════════════════════════════════
#  Multi-task loss with uncertainty weighting
# ═══════════════════════════════════════════════════════════════
class UncertaintyMultiTaskLoss(nn.Module):
    """
    Learns per-task uncertainty σ² and balances losses automatically.
    Kendall, Gal & Cipolla (2018): "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics"

    Loss = Σ_i  [ L_i / (2 * exp(log_var_i)) + log_var_i / 2 ]
    """
    def __init__(self, n_tasks: int = 5):
        super().__init__()
        # Initialise log-variances to 0 (σ²=1 initially)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=losses[0].device)
        info = {}
        task_names = ["direction", "ret_1d", "ret_5d", "ret_21d", "regime"]
        # Clamp log_vars to [-4, 4] — prevents precision from exploding
        # (exp(4)≈55× weight) or collapsing (exp(-4)≈0.018× weight),
        # which can cause NaN in early training when loss magnitudes
        # are very different across tasks.
        clamped_lv = self.log_vars.clamp(-4.0, 4.0)
        for i, (loss, name) in enumerate(zip(losses, task_names)):
            # Guard: skip non-finite individual losses
            if not torch.isfinite(loss):
                info[f"loss_{name}"] = 0.0
                info[f"weight_{name}"] = 0.0
                continue
            precision = torch.exp(-clamped_lv[i])
            weighted = precision * loss + clamped_lv[i] * 0.5
            total = total + weighted
            info[f"loss_{name}"] = loss.item()
            info[f"weight_{name}"] = precision.item()
        info["loss_total"] = total.item() if torch.isfinite(total) else 0.0
        return total, info


# ═══════════════════════════════════════════════════════════════
#  Cosine schedule with warm-up
# ═══════════════════════════════════════════════════════════════
def _cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_fn)


# ═══════════════════════════════════════════════════════════════
#  Compute losses for one batch
# ═══════════════════════════════════════════════════════════════
def _compute_losses(preds: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Individual task losses (not weighted yet).

    Includes numerical-stability safeguards:
    - Prediction clamping to prevent extreme logit values.
    - Per-loss NaN replacement with a safe fallback value so a single
      corrupt sample doesn't poison the entire mega-batch.
    """
    losses = []
    _SAFE = torch.tensor(0.0, device=next(iter(preds.values())).device)

    # Direction: binary cross-entropy with logits
    dir_pred = preds["direction"].squeeze(-1)
    dir_target = targets["direction"].squeeze(-1).clamp(0.0, 1.0)
    # Clamp logits to [-20, 20] — prevents exp() overflow inside BCE.
    dir_pred = dir_pred.clamp(-20.0, 20.0)
    l = F.binary_cross_entropy_with_logits(dir_pred, dir_target)
    losses.append(l if torch.isfinite(l) else _SAFE)

    # Return regression: Huber loss (robust to outliers in financial returns)
    for key in ["ret_1d", "ret_5d", "ret_21d"]:
        pred = preds[key].squeeze(-1)
        target = targets[key].squeeze(-1).clamp(-1.0, 1.0)
        l = F.huber_loss(pred, target, delta=0.05)
        losses.append(l if torch.isfinite(l) else _SAFE)

    # Regime: cross-entropy
    regime_pred = preds["regime"]
    regime_target = targets["regime"].squeeze(-1)
    # Clamp logits for cross-entropy stability
    regime_pred = regime_pred.clamp(-20.0, 20.0)
    l = F.cross_entropy(regime_pred, regime_target)
    losses.append(l if torch.isfinite(l) else _SAFE)

    return losses


# ═══════════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════════
def _compute_metrics(all_preds: Dict[str, List], all_targets: Dict[str, List]) -> Dict[str, float]:
    """Compute evaluation metrics from accumulated predictions."""
    if not all_preds["direction"] or not all_targets["direction"]:
        return {
            "direction_accuracy": 0.0,
            "ret_1d_mse": 0.0,
            "ret_1d_mae": 0.0,
            "ret_5d_mse": 0.0,
            "ret_5d_mae": 0.0,
            "ret_21d_mse": 0.0,
            "ret_21d_mae": 0.0,
            "ret_1d_dir_acc": 0.0,
            "ret_5d_dir_acc": 0.0,
            "ret_21d_dir_acc": 0.0,
            "regime_accuracy": 0.0,
            "sharpe_ratio": 0.0,
            "ret_1d_IC": 0.0,
            "ret_5d_IC": 0.0,
        }

    metrics = {}

    # Direction accuracy
    dir_pred = np.concatenate(all_preds["direction"])
    dir_target = np.concatenate(all_targets["direction"])
    dir_binary = (dir_pred > 0.5).astype(float)
    metrics["direction_accuracy"] = float(np.mean(dir_binary == dir_target))

    # Return MSE / MAE
    for key in ["ret_1d", "ret_5d", "ret_21d"]:
        pred = np.concatenate(all_preds[key])
        target = np.concatenate(all_targets[key])
        metrics[f"{key}_mse"] = float(np.mean((pred - target) ** 2))
        metrics[f"{key}_mae"] = float(np.mean(np.abs(pred - target)))

        # Directional accuracy of return predictions
        pred_dir = (pred > 0).astype(float)
        target_dir = (target > 0).astype(float)
        metrics[f"{key}_dir_acc"] = float(np.mean(pred_dir == target_dir))

    # Regime accuracy
    regime_pred = np.concatenate(all_preds["regime"])
    regime_target = np.concatenate(all_targets["regime"])
    metrics["regime_accuracy"] = float(np.mean(regime_pred == regime_target))

    # Simulated Sharpe (using 1d return predictions as signal)
    ret1d_pred = np.concatenate(all_preds["ret_1d"])
    ret1d_actual = np.concatenate(all_targets["ret_1d"])
    # Long when predicted > 0, short when < 0
    signal = np.sign(ret1d_pred)
    strategy_returns = signal * ret1d_actual
    if strategy_returns.std() > 0:
        metrics["sharpe_ratio"] = float(
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        )
    else:
        metrics["sharpe_ratio"] = 0.0

    # Information coefficient (correlation of predicted vs actual returns)
    for key in ["ret_1d", "ret_5d"]:
        pred = np.concatenate(all_preds[key])
        target = np.concatenate(all_targets[key])
        if pred.std() > 0 and target.std() > 0:
            ic = np.corrcoef(pred, target)[0, 1]
            metrics[f"{key}_IC"] = float(ic) if not np.isnan(ic) else 0.0
        else:
            metrics[f"{key}_IC"] = 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════
#  Main Trainer class
# ═══════════════════════════════════════════════════════════════
class ViziTrainer:
    """
    Complete training pipeline for ViziMarketTransformer.

    Usage:
        trainer = ViziTrainer(model_cfg, data_cfg, train_cfg)
        trainer.train()           # Phase 2: full-universe training
        trainer.evaluate("test")  # Generate evaluation report
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        data_cfg: Optional[DataConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
    ):
        self.model_cfg = model_cfg or ModelConfig()
        self.data_cfg = data_cfg or DataConfig()
        self.train_cfg = train_cfg or TrainConfig()

        # Discover symbols
        self.symbols = discover_symbols(self.data_cfg.data_root)
        logger.info("Discovered %d symbols", len(self.symbols))

        # Update model config with actual data dimensions
        self._probe_dimensions()

        # Create model
        self.device = torch.device(self.train_cfg.device)
        self.model = ViziMarketTransformer(self.model_cfg).to(self.device)
        logger.info("\n%s", self.model.summary())

        # Apply H100/GPU optimizations
        self.gpu_opts = apply_h100_optimizations(self.device)

        # Loss
        self.loss_fn = UncertaintyMultiTaskLoss(n_tasks=5).to(self.device)

        # Optimiser (include loss params for uncertainty weighting)
        # Use fused AdamW on H100/A100 for ~15% speedup
        opt_kwargs: Dict[str, Any] = dict(
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
            betas=self.train_cfg.betas,
        )
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        if self.train_cfg.use_fused_optimizer and self.device.type == "cuda":
            try:
                self.optimizer = AdamW(params, fused=True, **opt_kwargs)
            except TypeError:
                logger.info("Fused AdamW is unavailable in this torch build; using standard AdamW.")
                self.optimizer = AdamW(params, **opt_kwargs)
        else:
            self.optimizer = AdamW(params, **opt_kwargs)

        # Scheduler
        self.scheduler = _cosine_warmup_schedule(
            self.optimizer,
            self.train_cfg.warmup_steps,
            self.train_cfg.total_steps,
        )

        # AMP scaler
        self.scaler = _make_grad_scaler(self.device.type, enabled=self.train_cfg.use_amp)

        # Training monitor
        self.monitor = TrainingMonitor(
            self.train_cfg.total_steps,
            self.train_cfg.max_epochs,
            self.device,
        )

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

    def _probe_dimensions(self):
        """Load one sample to determine actual feature dimensions."""
        if not self.symbols:
            return
        from backend.vizi_ai.data_pipeline import StockDataStream
        stream = StockDataStream(self.symbols[0], 0, self.data_cfg, "train", shuffle=False)
        for sample in stream:
            self.model_cfg.n_price_features = sample["price_seq"].shape[-1]
            self.model_cfg.n_macro_features = sample["macro_seq"].shape[-1]
            self.model_cfg.n_market_features = sample["market_seq"].shape[-1]
            self.model_cfg.n_fundamental_features = sample["fundamental"].shape[-1]
            self.model_cfg.n_stocks = max(len(self.symbols), 1)
            logger.info(
                "Probed dims: price=%d, macro=%d, market=%d, fund=%d, stocks=%d",
                self.model_cfg.n_price_features,
                self.model_cfg.n_macro_features,
                self.model_cfg.n_market_features,
                self.model_cfg.n_fundamental_features,
                self.model_cfg.n_stocks,
            )
            break

    # ─────────────────────────────────────────────────────────
    #  Save / Load
    # ─────────────────────────────────────────────────────────
    def _save_checkpoint(self, path: Path, is_best: bool = False):
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_cfg": {k: v for k, v in vars(self.model_cfg).items() if not k.startswith("_")},
        }
        torch.save(state, path / "checkpoint.pt")

        # Save config as readable JSON
        config = {
            "model": {k: v for k, v in vars(self.model_cfg).items() if not k.startswith("_")},
            "train": {k: getattr(self.train_cfg, k) for k in dir(self.train_cfg) if not k.startswith("_") and not callable(getattr(self.train_cfg, k))},
            "data": {k: getattr(self.data_cfg, k) for k in dir(self.data_cfg) if not k.startswith("_") and not callable(getattr(self.data_cfg, k))},
        }
        # Convert non-serializable types
        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=_convert)

        if is_best:
            best_path = path.parent / "best"
            if best_path.exists():
                shutil.rmtree(best_path)
            shutil.copytree(path, best_path)

    def _load_checkpoint(self, path: Path):
        ckpt = torch.load(path / "checkpoint.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.loss_fn.load_state_dict(ckpt["loss_fn"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        logger.info("Loaded checkpoint from %s (step %d)", path, self.global_step)

    # ─────────────────────────────────────────────────────────
    #  Training loop
    # ─────────────────────────────────────────────────────────
    def train(self):
        """
        Main training loop: stream all stocks, multi-task loss, AMP.
        Integrated with training monitor, time estimator, and self-improvement.
        """
        cfg = self.train_cfg
        run_dir = cfg.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training run: %s", run_dir)

        # Create data loaders
        train_loader = create_dataloader(
            self.symbols, self.data_cfg, "train",
            batch_size=cfg.batch_size, shuffle=True,
        )
        val_loader = create_dataloader(
            self.symbols, self.data_cfg, "val",
            batch_size=cfg.batch_size * 2, shuffle=False,
        )

        # Pre-training time estimate (optional)
        if cfg.run_time_estimate:
            self.monitor.phase_start("PRE-TRAINING TIME ESTIMATION")
            est = TimeEstimator.estimate(self.model, train_loader, cfg)
            with open(run_dir / "time_estimate.json", "w") as f:
                json.dump(est, f, indent=2)
            # Re-create train_loader since the iterator was consumed
            train_loader = create_dataloader(
                self.symbols, self.data_cfg, "train",
                batch_size=cfg.batch_size, shuffle=True,
            )

        self.monitor.phase_start("PHASE 2: Full-Universe Multi-Modal Training",
                                 n_items=len(self.symbols))

        self.model.train()
        accum_loss = 0.0
        accum_count = 0
        epoch = 0
        t0 = time.time()
        last_loss_info: Dict[str, float] = {}

        for epoch in range(1, cfg.max_epochs + 1):
            logger.info("── Epoch %d/%d ──", epoch, cfg.max_epochs)
            epoch_loss = 0.0
            epoch_batches = 0

            for batch in train_loader:
                # Move to device
                batch = self._to_device(batch)
                batch = _sanitize_batch(batch)
                batch_size = batch["price_seq"].shape[0]

                with _autocast_ctx(device_type=self.device.type, enabled=cfg.use_amp):
                    preds = self.model(batch)
                    task_losses = _compute_losses(preds, batch["targets"])
                    total_loss, loss_info = self.loss_fn(task_losses)
                    total_loss = total_loss / cfg.grad_accum_steps

                self.scaler.scale(total_loss).backward()

                accum_loss += total_loss.item() * cfg.grad_accum_steps
                accum_count += 1

                if accum_count % cfg.grad_accum_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.global_step += 1
                    epoch_loss += accum_loss / accum_count
                    epoch_batches += 1

                    # Update loss info with LR for monitor
                    loss_info["lr"] = self.optimizer.param_groups[0]["lr"]
                    last_loss_info = loss_info

                    # Training monitor update
                    self.monitor.step(
                        self.global_step, epoch, loss_info,
                        samples_in_batch=batch_size * cfg.grad_accum_steps,
                    )

                    accum_loss = 0.0
                    accum_count = 0

                    # Periodic validation
                    if self.global_step % cfg.val_every_steps == 0:
                        val_loss, val_metrics = self._validate(val_loader)
                        self.val_history.append({
                            "step": self.global_step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            **val_metrics,
                        })

                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1

                        # Save checkpoint
                        ckpt_dir = run_dir / f"step_{self.global_step:06d}"
                        self._save_checkpoint(ckpt_dir, is_best=is_best)

                        # Monitor validation display
                        self.monitor.validation(self.global_step, val_loss, val_metrics)

                        self.model.train()

                        # Early stopping
                        if self.patience_counter >= cfg.patience:
                            logger.info("Early stopping at step %d", self.global_step)
                            break

                # Log training progress
                if self.global_step > 0 and self.global_step % 100 == 0:
                    train_info = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "train_loss": loss_info.get("loss_total", 0),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    train_info.update(loss_info)
                    self.train_history.append(train_info)

            if self.patience_counter >= cfg.patience:
                break

            # Epoch summary
            avg_epoch_loss = epoch_loss / max(1, epoch_batches)
            logger.info("Epoch %d done — avg_loss=%.4f", epoch, avg_epoch_loss)

        # Save final model
        self._save_checkpoint(run_dir / "final")

        # Save training history
        with open(run_dir / "train_history.json", "w") as f:
            json.dump(self.train_history, f, indent=2)
        with open(run_dir / "val_history.json", "w") as f:
            json.dump(self.val_history, f, indent=2)

        # Training monitor summary
        self.monitor.summary(self.global_step, epoch)

        # Generate evaluation report on test set
        report = self.evaluate("test", run_dir)

        # Self-improvement: save this run and get suggestions for next
        if cfg.self_improve and report:
            config_snapshot = {
                "d_model": self.model_cfg.d_model,
                "n_layers": self.model_cfg.n_layers,
                "fusion_layers": self.model_cfg.fusion_layers,
                "use_moe": self.model_cfg.use_moe,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "epochs_run": epoch,
                "total_steps": self.global_step,
            }
            agg_metrics = report.get("aggregate", {})
            SelfImprover.save_run(cfg.run_name, config_snapshot, agg_metrics)
            SelfImprover.suggest_next()

    # ─────────────────────────────────────────────────────────
    #  Validation
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Run validation and return (avg_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: Dict[str, List] = {k: [] for k in ["direction", "ret_1d", "ret_5d", "ret_21d", "regime"]}
        all_targets: Dict[str, List] = {k: [] for k in ["direction", "ret_1d", "ret_5d", "ret_21d", "regime"]}

        for batch in val_loader:
            batch = self._to_device(batch)
            with _autocast_ctx(device_type=self.device.type, enabled=self.train_cfg.use_amp):
                preds = self.model(batch)
                task_losses = _compute_losses(preds, batch["targets"])
                loss, _ = self.loss_fn(task_losses)

            total_loss += loss.item()
            n_batches += 1

            # Accumulate predictions
            all_preds["direction"].append(torch.sigmoid(preds["direction"]).squeeze(-1).cpu().numpy())
            all_targets["direction"].append(batch["targets"]["direction"].squeeze(-1).cpu().numpy())
            for key in ["ret_1d", "ret_5d", "ret_21d"]:
                all_preds[key].append(preds[key].squeeze(-1).cpu().numpy())
                all_targets[key].append(batch["targets"][key].squeeze(-1).cpu().numpy())
            all_preds["regime"].append(preds["regime"].argmax(dim=-1).cpu().numpy())
            all_targets["regime"].append(batch["targets"]["regime"].squeeze(-1).cpu().numpy())

            if n_batches >= 200:  # cap validation to 200 batches
                break

        avg_loss = total_loss / max(1, n_batches)
        metrics = _compute_metrics(all_preds, all_targets) if n_batches > 0 else {}
        return avg_loss, metrics

    # ─────────────────────────────────────────────────────────
    #  Evaluation / Report
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, split: str = "test", save_dir: Optional[Path] = None):
        """
        Full evaluation on a split.  Generates a JSON report with
        per-stock and aggregate metrics.
        """
        self.model.eval()
        loader = create_dataloader(
            self.symbols, self.data_cfg, split,
            batch_size=self.train_cfg.batch_size * 2, shuffle=False,
        )

        all_preds: Dict[str, List] = {k: [] for k in ["direction", "ret_1d", "ret_5d", "ret_21d", "regime"]}
        all_targets: Dict[str, List] = {k: [] for k in ["direction", "ret_1d", "ret_5d", "ret_21d", "regime"]}
        all_stocks: List[int] = []

        for batch in loader:
            batch = self._to_device(batch)
            with _autocast_ctx(device_type=self.device.type, enabled=self.train_cfg.use_amp):
                preds = self.model(batch)

            all_preds["direction"].append(torch.sigmoid(preds["direction"]).squeeze(-1).cpu().numpy())
            all_targets["direction"].append(batch["targets"]["direction"].squeeze(-1).cpu().numpy())
            for key in ["ret_1d", "ret_5d", "ret_21d"]:
                all_preds[key].append(preds[key].squeeze(-1).cpu().numpy())
                all_targets[key].append(batch["targets"][key].squeeze(-1).cpu().numpy())
            all_preds["regime"].append(preds["regime"].argmax(dim=-1).cpu().numpy())
            all_targets["regime"].append(batch["targets"]["regime"].squeeze(-1).cpu().numpy())
            all_stocks.extend(batch["stock_id"].cpu().numpy().tolist())

        if not all_preds["direction"]:
            logger.warning("No samples found for split '%s'. Check data availability (e.g., Git LFS pull).", split)
            report = {
                "split": split,
                "n_samples": 0,
                "n_stocks": 0,
                "aggregate": _compute_metrics(all_preds, all_targets),
                "per_stock": {},
                "timestamp": datetime.now().isoformat(),
            }
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_dir / f"eval_{split}.json", "w") as f:
                    json.dump(report, f, indent=2)
            return report

        # Aggregate metrics
        agg_metrics = _compute_metrics(all_preds, all_targets)

        # Per-stock metrics
        stock_ids = np.array(all_stocks)
        per_stock = {}
        dir_pred = np.concatenate(all_preds["direction"])
        dir_target = np.concatenate(all_targets["direction"])
        ret1d_pred = np.concatenate(all_preds["ret_1d"])
        ret1d_target = np.concatenate(all_targets["ret_1d"])

        for sid in np.unique(stock_ids):
            mask = stock_ids == sid
            if mask.sum() < 10:
                continue
            sym = self.symbols[int(sid)] if sid < len(self.symbols) else f"stock_{sid}"
            dp = dir_pred[mask]
            dt = dir_target[mask]
            rp = ret1d_pred[mask]
            rt = ret1d_target[mask]
            acc = float(np.mean((dp > 0.5) == dt))
            sig = np.sign(rp)
            strat = sig * rt
            sharpe = float(strat.mean() / strat.std() * np.sqrt(252)) if strat.std() > 0 else 0.0
            per_stock[sym] = {
                "n_samples": int(mask.sum()),
                "direction_accuracy": acc,
                "ret_1d_mse": float(np.mean((rp - rt) ** 2)),
                "sharpe_ratio": sharpe,
            }

        report = {
            "split": split,
            "n_samples": len(stock_ids),
            "n_stocks": len(per_stock),
            "aggregate": agg_metrics,
            "per_stock": per_stock,
            "timestamp": datetime.now().isoformat(),
        }

        # Print summary
        logger.info(
            "\n%s Evaluation Report (%s)\n%s\n"
            "  Samples: %d | Stocks: %d\n"
            "  Direction Accuracy: %.1f%%\n"
            "  Ret 1d MAE: %.4f | MSE: %.6f\n"
            "  Ret 1d IC: %.3f\n"
            "  Sharpe Ratio: %.2f\n"
            "  Regime Accuracy: %.1f%%\n%s",
            "═" * 50, split, "═" * 50,
            report["n_samples"], report["n_stocks"],
            agg_metrics.get("direction_accuracy", 0) * 100,
            agg_metrics.get("ret_1d_mae", 0),
            agg_metrics.get("ret_1d_mse", 0),
            agg_metrics.get("ret_1d_IC", 0),
            agg_metrics.get("sharpe_ratio", 0),
            agg_metrics.get("regime_accuracy", 0) * 100,
            "═" * 50,
        )

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"eval_{split}.json", "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Report saved to %s", save_dir / f"eval_{split}.json")

        return report

    # ─────────────────────────────────────────────────────────
    #  Utility
    # ─────────────────────────────────────────────────────────
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively move batch tensors to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                result[k] = self._to_device(v)
            else:
                result[k] = v
        return result


# ═══════════════════════════════════════════════════════════════
#  Phase 1 helpers — parallel specialist warm-up
# ═══════════════════════════════════════════════════════════════

def _estimate_parallel_stocks(
    device: torch.device,
    batch_size: int,
    n_symbols: int,
) -> int:
    """
    Estimate how many stocks can be mega-batched in one forward pass.

    VRAM breakdown per stock (at batch_size=64, d_model=256, AMP FP16):
      • Encoder forward (temporary, no grad stored for frozen params): ~60 MB
      • Fusion activations stored for backward (4 layers, ~251 tokens): ~80 MB
      • Head activations + gradient buffers: ~10 MB
      • Input tensors on device: ~5 MB
      ────────────────────────────────────────
      ≈ 155 MB per stock at batch_size=64

    We reserve 4 GB for the model, optimizer states, CUDA context, and
    a safety margin.  Scales linearly with batch_size.
    """
    if device.type != "cuda":
        return min(4, n_symbols)

    total_gb = torch.cuda.get_device_properties(device).total_memory / (1 << 30)
    reserve_gb = 4.0

    per_stock_mb = 155.0 * (batch_size / 64)
    available_mb = (total_gb - reserve_gb) * 1024

    k = max(1, int(available_mb / per_stock_mb))
    # Cap so that mega-batch doesn't exceed reasonable kernel sizes.
    # Also cap at 8 for training stability — very large mega-batches
    # (50+ stocks) amplify data-quality problems and cause NaN loss.
    k = min(k, n_symbols, 8)
    return k


def _has_non_finite_inputs(batch: Dict[str, Any]) -> bool:
    """Return True if any floating-point tensor in *batch* contains NaN/Inf."""
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all():
            return True
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, torch.Tensor) and vv.is_floating_point() and not torch.isfinite(vv).all():
                    return True
    return False


def _diagnose_non_finite(batch: Dict[str, Any], symbols: List[str]) -> str:
    """Build a human-readable diagnostic string for non-finite inputs.

    Reports which modality tensors contain NaN/Inf and which stock IDs
    are affected so the operator can investigate the underlying CSV.
    """
    parts: List[str] = []
    affected_stocks: set = set()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            bad = ~torch.isfinite(v)
            if bad.any():
                n_bad = int(bad.sum())
                parts.append(f"{k}: {n_bad} non-finite")
                # Identify which batch rows are affected
                if v.dim() >= 1:
                    bad_rows = bad.any(dim=tuple(range(1, v.dim()))).nonzero(as_tuple=True)[0]
                    affected_stocks.update(bad_rows.tolist())
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor) and vv.is_floating_point():
                    bad = ~torch.isfinite(vv)
                    if bad.any():
                        parts.append(f"targets/{kk}: {int(bad.sum())} non-finite")
    # Map row indices to stock symbols
    stock_ids = batch.get("stock_id")
    if stock_ids is not None and affected_stocks:
        sym_names = []
        for idx in sorted(affected_stocks):
            sid = int(stock_ids[idx])
            sym_names.append(symbols[sid] if sid < len(symbols) else f"id:{sid}")
        parts.append(f"stocks: {', '.join(sym_names[:10])}")
    return "; ".join(parts) if parts else "unknown"


def _sanitize_batch(batch: Dict[str, Any], clamp_val: float = 10.0) -> Dict[str, Any]:
    """In-place clamp all floating-point tensors in a batch dict.

    This is the last-resort safety net inside the training loop.
    Even with data-pipeline sanitization, forward-pass activations
    from a prior corrupt batch could leave gradients in a bad state,
    so clamping inputs again right before the forward pass is cheap
    insurance."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            v.clamp_(-clamp_val, clamp_val)
            # Replace NaN with 0 (in-place)
            v[~torch.isfinite(v)] = 0.0
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor) and vv.is_floating_point():
                    vv.clamp_(-1.0, 1.0)  # tighter range for targets
                    vv[~torch.isfinite(vv)] = 0.0
    return batch


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in a batch dict (including nested targets) to *device*."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _cat_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate a list of batch dicts along dim-0 into a single mega-batch."""
    merged: Dict[str, Any] = {}
    for key in batches[0]:
        vals = [b[key] for b in batches]
        if isinstance(vals[0], torch.Tensor):
            merged[key] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], dict):
            merged[key] = {
                k: torch.cat([v[key][k] for v in batches], dim=0)
                for k in vals[0]
            }
        else:
            merged[key] = vals[0]
    return merged


def _prefetch_one(iterator):
    """Pull the next batch from *iterator*; return (batch, True) or (None, False)."""
    try:
        return next(iterator), True
    except StopIteration:
        return None, False


# ═══════════════════════════════════════════════════════════════
#  Per-stock fine-tuning (Phase 1 — parallel specialist warm-up)
# ═══════════════════════════════════════════════════════════════
def train_per_stock_specialists(
    symbols: List[str],
    model: ViziMarketTransformer,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    n_epochs: int = 3,
    num_parallel_stocks: int = 0,
):
    """
    Parallel per-stock fine-tuning pass.

    Freezes the shared encoders and only trains the prediction heads +
    stock embeddings.  This gives each stock a warm-started specialisation
    before the full multi-stock training phase.

    **Parallel mega-batch strategy (new)**
    Stocks are grouped into chunks of ``num_parallel_stocks``.  Within
    each chunk one batch per stock is drawn, concatenated along dim-0
    into a single *mega-batch*, and forwarded through the model in one
    GPU kernel.  This fully saturates tensor-core throughput on large
    GPUs (e.g. H100) instead of leaving SMs idle with small per-stock
    batches.

    Args:
        num_parallel_stocks: How many stocks per mega-batch.
            0 (default) → auto-detect from available VRAM.
    """
    device = torch.device(train_cfg.device)
    model.to(device)

    # Freeze everything except heads + stock embedding
    for name, param in model.named_parameters():
        if "heads" in name or "stock_embed" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=train_cfg.lr * train_cfg.specialist_lr_scale,
        weight_decay=train_cfg.weight_decay,
    )
    # Use fixed task weights during specialist warm-up for numerical stability.
    warmup_weights = [
        train_cfg.loss_w_direction,
        train_cfg.loss_w_ret_1d,
        train_cfg.loss_w_ret_5d,
        train_cfg.loss_w_ret_21d,
        train_cfg.loss_w_regime,
    ]
    scaler = _make_grad_scaler(device.type, enabled=train_cfg.use_amp)

    # ── Determine group size ──
    if num_parallel_stocks <= 0:
        num_parallel_stocks = _estimate_parallel_stocks(
            device, train_cfg.batch_size, len(symbols),
        )
    logger.info(
        "Phase 1 — parallel specialist warm-up: %d stocks in groups of %d "
        "(batch_size=%d → mega-batch up to %d)",
        len(symbols),
        num_parallel_stocks,
        train_cfg.batch_size,
        num_parallel_stocks * train_cfg.batch_size,
    )

    total_sym_done = 0
    phase1_t0 = time.time()

    # ── Process groups ──
    for g_start in range(0, len(symbols), num_parallel_stocks):
        group_syms = symbols[g_start : g_start + num_parallel_stocks]
        K = len(group_syms)

        loaders = [
            create_dataloader(
                [sym], data_cfg, "train",
                batch_size=train_cfg.batch_size,
                shuffle=True,
            )
            for sym in group_syms
        ]

        per_stock_loss = [0.0] * K
        per_stock_batches = [0] * K
        group_t0 = time.time()

        model.train()

        for epoch in range(n_epochs):
            iters = [iter(ld) for ld in loaders]
            exhausted = [False] * K
            round_idx = 0

            while not all(exhausted):
                # ── Prefetch one batch per stock in thread pool ──
                fetched: List[Optional[Dict[str, Any]]] = [None] * K
                with ThreadPoolExecutor(max_workers=min(K, 8)) as pool:
                    futures = {}
                    for i in range(K):
                        if not exhausted[i]:
                            futures[i] = pool.submit(_prefetch_one, iters[i])
                    for i, fut in futures.items():
                        batch, ok = fut.result()
                        if ok:
                            fetched[i] = batch
                        else:
                            exhausted[i] = True

                # ── Move to device & sanitize inputs ──
                ready: List[Tuple[int, Dict[str, Any]]] = []
                for i in range(K):
                    if fetched[i] is None:
                        continue
                    batch_dev = _move_batch_to_device(fetched[i], device)
                    # Sanitize: clamp all float tensors, replace NaN in-place
                    batch_dev = _sanitize_batch(batch_dev)
                    if _has_non_finite_inputs(batch_dev):
                        diag = _diagnose_non_finite(
                            batch_dev, group_syms,
                        )
                        logger.warning(
                            "  Specialist %s: skipping non-finite input "
                            "(epoch %d) — %s",
                            group_syms[i], epoch + 1, diag,
                        )
                        continue
                    ready.append((i, batch_dev))

                if not ready:
                    continue

                # ── Build mega-batch (concatenate along dim-0) ──
                stock_batch_sizes = [b["price_seq"].size(0) for _, b in ready]
                mega_batch = _cat_batches([b for _, b in ready])

                # ── Forward + loss (single GPU kernel on the mega-batch) ──
                optimizer.zero_grad(set_to_none=True)

                try:
                    with _autocast_ctx(device_type=device.type, enabled=train_cfg.use_amp):
                        preds = model(mega_batch)
                        task_losses = _compute_losses(preds, mega_batch["targets"])
                        total_loss = sum(w * l for w, l in zip(warmup_weights, task_losses))
                        # Clamp total loss to prevent initial explosion from
                        # outlier samples.  50 is generous (normal loss ~1-5).
                        total_loss = total_loss.clamp(max=50.0)

                    if not torch.isfinite(total_loss):
                        diag = _diagnose_non_finite(mega_batch, symbols)
                        logger.warning(
                            "  Group %d round %d: non-finite mega-batch loss — "
                            "skipping (%s)",
                            g_start // num_parallel_stocks + 1, round_idx,
                            diag,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        round_idx += 1
                        continue

                    # ── Backward + step ──
                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
                        optimizer.step()

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        # OOM on this mega-batch — log, free memory, skip round
                        logger.warning(
                            "  CUDA OOM in group %d round %d (%d stocks, mega-batch %d) — skipping round",
                            g_start // num_parallel_stocks + 1,
                            round_idx,
                            len(ready),
                            sum(stock_batch_sizes),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        round_idx += 1
                        continue
                    raise

                # ── Per-stock loss tracking (no-grad slicing) ──
                with torch.no_grad():
                    offset = 0
                    for j, (stock_idx, _) in enumerate(ready):
                        bs = stock_batch_sizes[j]
                        sliced_preds = {k: v[offset:offset + bs] for k, v in preds.items()}
                        sliced_tgts = {k: v[offset:offset + bs] for k, v in mega_batch["targets"].items()}
                        tl = _compute_losses(sliced_preds, sliced_tgts)
                        sl = sum(w * l for w, l in zip(warmup_weights, tl))
                        per_stock_loss[stock_idx] += sl.item()
                        per_stock_batches[stock_idx] += 1
                        offset += bs

                round_idx += 1

        # ── Group summary ──
        group_elapsed = time.time() - group_t0
        for i, sym in enumerate(group_syms):
            total_sym_done += 1
            avg_loss = per_stock_loss[i] / max(1, per_stock_batches[i])
            logger.info(
                "  Specialist %-10s (%3d/%d): avg_loss=%.4f  (%d batches)",
                sym, total_sym_done, len(symbols), avg_loss, per_stock_batches[i],
            )
        logger.info(
            "  Group %d/%d done — %d stocks in %.1fs  (%.2f stocks/s)",
            g_start // num_parallel_stocks + 1,
            math.ceil(len(symbols) / num_parallel_stocks),
            K, group_elapsed,
            K / max(0.01, group_elapsed),
        )

    # Unfreeze everything for Phase 2
    for param in model.parameters():
        param.requires_grad = True

    elapsed = time.time() - phase1_t0
    logger.info(
        "Phase 1 specialist warm-up complete — %d stocks, %d groups, %.1fs total (%.2f stocks/s)",
        len(symbols),
        math.ceil(len(symbols) / num_parallel_stocks),
        elapsed,
        len(symbols) / max(0.01, elapsed),
    )
