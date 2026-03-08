"""
ViziGenesis vizi-o1 — Training Engine
======================================
Handles the complete training lifecycle:

  Phase 1: Per-stock specialist warm-up (optional)
    Train lightweight heads on individual stocks to bootstrap
    good per-stock representations before full-universe training.

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
"""
from __future__ import annotations

import json, logging, math, os, shutil, time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
        for i, (loss, name) in enumerate(zip(losses, task_names)):
            precision = torch.exp(-self.log_vars[i])
            weighted = precision * loss + self.log_vars[i] * 0.5
            total = total + weighted
            info[f"loss_{name}"] = loss.item()
            info[f"weight_{name}"] = precision.item()
        info["loss_total"] = total.item()
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
    """Individual task losses (not weighted yet)."""
    losses = []

    # Direction: binary cross-entropy with logits (autocast-safe)
    dir_pred = preds["direction"].squeeze(-1)
    dir_target = targets["direction"].squeeze(-1)
    losses.append(F.binary_cross_entropy_with_logits(dir_pred, dir_target))

    # Return regression: Huber loss (robust to outliers in financial returns)
    for key in ["ret_1d", "ret_5d", "ret_21d"]:
        pred = preds[key].squeeze(-1)
        target = targets[key].squeeze(-1)
        losses.append(F.huber_loss(pred, target, delta=0.05))

    # Regime: cross-entropy
    regime_pred = preds["regime"]
    regime_target = targets["regime"].squeeze(-1)
    losses.append(F.cross_entropy(regime_pred, regime_target))

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

        # Loss
        self.loss_fn = UncertaintyMultiTaskLoss(n_tasks=5).to(self.device)

        # Optimiser (include loss params for uncertainty weighting)
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
            betas=self.train_cfg.betas,
        )

        # Scheduler
        self.scheduler = _cosine_warmup_schedule(
            self.optimizer,
            self.train_cfg.warmup_steps,
            self.train_cfg.total_steps,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.train_cfg.use_amp)

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

        self.model.train()
        accum_loss = 0.0
        accum_count = 0
        epoch = 0
        t0 = time.time()

        for epoch in range(1, cfg.max_epochs + 1):
            logger.info("── Epoch %d/%d ──", epoch, cfg.max_epochs)
            epoch_loss = 0.0
            epoch_batches = 0

            for batch in train_loader:
                # Move to device
                batch = self._to_device(batch)

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

                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - t0
                        logger.info(
                            "Step %d | val_loss=%.4f | dir_acc=%.3f | sharpe=%.2f | "
                            "lr=%.2e | %.0fs%s",
                            self.global_step, val_loss,
                            val_metrics.get("direction_accuracy", 0),
                            val_metrics.get("sharpe_ratio", 0),
                            lr, elapsed,
                            " ★ BEST" if is_best else "",
                        )

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

        elapsed = time.time() - t0
        logger.info("Training complete in %.0fs (%d steps, %d epochs)", elapsed, self.global_step, epoch)

        # Generate evaluation report on test set
        self.evaluate("test", run_dir)

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
#  Per-stock fine-tuning (Phase 1 — optional specialist warm-up)
# ═══════════════════════════════════════════════════════════════
def train_per_stock_specialists(
    symbols: List[str],
    model: ViziMarketTransformer,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    n_epochs: int = 3,
):
    """
    Quick per-stock fine-tuning pass.

    Freezes the shared encoders and only trains the prediction heads +
    stock embeddings.  This gives each stock a warm-started specialisation
    before the full multi-stock training phase.

    Why?
    Different stocks have different volatility profiles, sector dynamics,
    and fundamental characteristics.  A brief per-stock pass helps the
    model's output layers calibrate to individual stock distributions
    before learning cross-stock patterns.
    """
    device = torch.device(train_cfg.device)
    model.to(device)

    # Freeze everything except heads + stock embedding
    for name, param in model.named_parameters():
        if "heads" in name or "stock_embed" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.lr * 0.5,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = UncertaintyMultiTaskLoss(n_tasks=5).to(device)
    scaler = GradScaler(enabled=(train_cfg.use_amp and device.type == "cuda"))

    for sym_idx, sym in enumerate(symbols):
        loader = create_dataloader(
            [sym], data_cfg, "train",
            batch_size=train_cfg.batch_size,
            shuffle=True,
        )

        total_loss = 0.0
        n_batches = 0
        model.train()

        for _epoch in range(n_epochs):
            for batch in loader:
                batch_dev = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_dev[k] = v.to(device, non_blocking=True)
                    elif isinstance(v, dict):
                        batch_dev[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
                    else:
                        batch_dev[k] = v

                with _autocast_ctx(device_type=device.type, enabled=train_cfg.use_amp):
                    preds = model(batch_dev)
                    task_losses = _compute_losses(preds, batch_dev["targets"])
                    loss, _ = loss_fn(task_losses)

                # Skip pathological batches to keep warm-up stable.
                if not torch.isfinite(loss):
                    logger.warning("  Specialist %s: skipping non-finite loss batch", sym)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        train_cfg.grad_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        train_cfg.grad_clip,
                    )
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg = total_loss / max(1, n_batches)
        logger.info("  Specialist %s (%d/%d): avg_loss=%.4f (%d batches)",
                     sym, sym_idx + 1, len(symbols), avg, n_batches)

    # Unfreeze everything for Phase 2
    for param in model.parameters():
        param.requires_grad = True

    logger.info("Per-stock specialist warm-up complete for %d stocks", len(symbols))
