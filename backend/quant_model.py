"""
Quant-mode hybrid deep learning architecture.

Architecture:
  Branch 1 — TFT-inspired: Variable Selection Network → LSTM encoder → Multi-Head Attention
  Branch 2 — Bidirectional LSTM with temporal attention
  Branch 3 — GRU with skip connections

Ensemble — Learned attention-weighted combination of all three branches.

Multi-head output:
  • Direction     : P(UP) via sigmoid  (prob 0–1)
  • Return_1d     : next-day return prediction
  • Return_5d     : 5-day forward return
  • Return_30d    : 30-day forward return
  • Excess_Return : return vs NASDAQ benchmark
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import (
    DEVICE, QUANT_HIDDEN_SIZE, QUANT_NUM_HEADS, QUANT_NUM_LAYERS,
    QUANT_DROPOUT, QUANT_LEARNING_RATE, QUANT_BATCH_SIZE, QUANT_EPOCHS,
    QUANT_EARLY_STOP,
    LOSS_W_DIRECTION, LOSS_W_RET_1D, LOSS_W_RET_5D, LOSS_W_RET_30D, LOSS_W_EXCESS,
    AUGMENT_NOISE_STD, AUGMENT_SCALE_JITTER,
)

logger = logging.getLogger("vizigenesis.quant_model")


# ═══════════════════════════════════════════════════════════════════════
# 1. Building blocks
# ═══════════════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """GRN from Temporal Fusion Transformer — learns non-linear gated transform."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_hidden, d_out)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        proj = self.fc2(h)
        gate = self.sigmoid(self.gate(h))
        return self.layer_norm(self.skip(x) + gate * proj)


class VariableSelectionNetwork(nn.Module):
    """
    VSN from TFT — learns feature importance weights via softmax gating.
    Input: (batch, seq_len, n_features)
    Output: (batch, seq_len, d_model), feature_weights (batch, n_features)
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature embedding
        self.feature_transforms = nn.ModuleList([
            GatedResidualNetwork(1, d_model // 2, d_model, dropout)
            for _ in range(n_features)
        ])

        # Feature-selection weights
        self.weight_grn = GatedResidualNetwork(n_features * d_model, d_model, n_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        transformed = []
        for i in range(F):
            feat = x[:, :, i:i+1]  # (B, T, 1)
            transformed.append(self.feature_transforms[i](feat))  # (B, T, d_model)

        stacked = torch.stack(transformed, dim=2)  # (B, T, F, d_model)

        # Compute attention weights per time step
        flat = stacked.reshape(B, T, F * self.d_model)  # (B, T, F*d_model)
        # Pool across time for global feature importance
        flat_pooled = flat.mean(dim=1)  # (B, F*d_model)
        weights = self.softmax(self.weight_grn(flat_pooled))  # (B, F)

        # Weight and sum features
        weights_expanded = weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        selected = (stacked * weights_expanded).sum(dim=2)  # (B, T, d_model)

        return selected, weights


class TemporalAttention(nn.Module):
    """Multi-head self-attention over temporal dimension."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d)
        attn_out, attn_weights = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out)), attn_weights


# ═══════════════════════════════════════════════════════════════════════
# 2. Three branches
# ═══════════════════════════════════════════════════════════════════════

class TFTBranch(nn.Module):
    """TFT-inspired: VSN → LSTM encoder → Multi-Head Attention → output."""

    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.encoder = nn.LSTM(d_model, d_model, num_layers=n_layers,
                               batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attention = TemporalAttention(d_model, n_heads, dropout)
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

    def forward(self, x):
        selected, feat_weights = self.vsn(x)
        encoded, _ = self.encoder(selected)
        attended, attn_weights = self.attention(encoded)
        out = self.grn(attended[:, -1, :])  # last time step
        return out, feat_weights, attn_weights


class BiLSTMBranch(nn.Module):
    """Bidirectional LSTM with temporal attention pooling."""

    def __init__(self, n_features: int, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.bilstm = nn.LSTM(d_model, d_model // 2, num_layers=n_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.attn_fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        proj = F.gelu(self.input_proj(x))
        out, _ = self.bilstm(proj)  # (B, T, d_model)
        # Temporal attention pooling
        attn_scores = torch.softmax(self.attn_fc(out).squeeze(-1), dim=1)  # (B, T)
        pooled = (out * attn_scores.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return self.dropout(pooled), attn_scores


class GRUBranch(nn.Module):
    """GRU with skip connection and last-hidden output."""

    def __init__(self, n_features: int, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.skip = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        proj = F.gelu(self.input_proj(x))
        out, _ = self.gru(proj)
        last = out[:, -1, :]
        skip = self.skip(x[:, -1, :])  # skip connection from last time step
        return self.dropout(self.norm(last + skip))


# ═══════════════════════════════════════════════════════════════════════
# 3. Full hybrid model with attention ensemble + multi-head output
# ═══════════════════════════════════════════════════════════════════════

class QuantHybridModel(nn.Module):
    """
    Institutional-grade hybrid architecture:
      TFT Branch + BiLSTM Branch + GRU Branch
      → Attention-based ensemble
      → Multi-head output: direction, returns (1d/5d/30d), excess
    """

    def __init__(self, n_features: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Three branches
        self.tft = TFTBranch(n_features, d_model, n_heads, n_layers, dropout)
        self.bilstm = BiLSTMBranch(n_features, d_model, n_layers, dropout)
        self.gru = GRUBranch(n_features, d_model, n_layers, dropout)

        # Attention-based ensemble: learns how to weight branches
        self.ensemble_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1),
        )

        self.ensemble_proj = nn.Linear(d_model, d_model)
        self.ensemble_norm = nn.LayerNorm(d_model)

        # Multi-head output
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.return_1d_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.return_5d_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.return_30d_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.excess_return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_features)

        Returns dict:
            direction   : (batch, 1) — P(up) in [0, 1]
            return_1d   : (batch, 1) — predicted 1d return
            return_5d   : (batch, 1) — predicted 5d return
            return_30d  : (batch, 1) — predicted 30d return
            excess      : (batch, 1) — predicted excess return vs benchmark
            branch_weights : (batch, 3) — ensemble weights
            feat_weights   : (batch, n_features) — feature importance from TFT
        """
        tft_out, feat_weights, _ = self.tft(x)
        bilstm_out, _ = self.bilstm(x)
        gru_out = self.gru(x)

        # Concatenate branch outputs for gating
        concat = torch.cat([tft_out, bilstm_out, gru_out], dim=-1)  # (B, 3*d)
        branch_weights = self.ensemble_gate(concat)  # (B, 3)

        # Weighted combination
        stacked = torch.stack([tft_out, bilstm_out, gru_out], dim=1)  # (B, 3, d)
        weighted = (stacked * branch_weights.unsqueeze(-1)).sum(dim=1)  # (B, d)
        fused = self.ensemble_norm(self.ensemble_proj(weighted))

        return {
            "direction": self.direction_head(fused),
            "return_1d": self.return_1d_head(fused),
            "return_5d": self.return_5d_head(fused),
            "return_30d": self.return_30d_head(fused),
            "excess": self.excess_return_head(fused),
            "branch_weights": branch_weights,
            "feat_weights": feat_weights,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. Multi-task loss function
# ═══════════════════════════════════════════════════════════════════════

class QuantMultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss:
      L = w_dir * BCE(direction) + w_1d * MSE(ret_1d) + w_5d * MSE(ret_5d)
        + w_30d * MSE(ret_30d) + w_ex * MSE(excess)

    Masks NaN targets so they don't contribute to loss.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.mse = nn.SmoothL1Loss(reduction="none", beta=0.5)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}

        # Direction (BCE)
        dir_loss = self.bce(outputs["direction"].squeeze(-1), targets["Direction"])
        dir_mask = masks["Direction"]
        losses["direction"] = (dir_loss * dir_mask).sum() / dir_mask.sum().clamp(min=1)

        # Return heads (MSE)
        for key, w, out_key in [
            ("Return_1d", LOSS_W_RET_1D, "return_1d"),
            ("Return_5d", LOSS_W_RET_5D, "return_5d"),
            ("Return_30d", LOSS_W_RET_30D, "return_30d"),
            ("Excess_Return", LOSS_W_EXCESS, "excess"),
        ]:
            pred = outputs[out_key].squeeze(-1)
            tgt = targets[key]
            mask = masks[key]
            loss = self.mse(pred, tgt)
            losses[key] = (loss * mask).sum() / mask.sum().clamp(min=1)

        total = (
            LOSS_W_DIRECTION * losses["direction"]
            + LOSS_W_RET_1D * losses.get("Return_1d", 0)
            + LOSS_W_RET_5D * losses.get("Return_5d", 0)
            + LOSS_W_RET_30D * losses.get("Return_30d", 0)
            + LOSS_W_EXCESS * losses.get("Excess_Return", 0)
        )

        breakdown = {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
        return total, breakdown


# ═══════════════════════════════════════════════════════════════════════
# 5. Training routine
# ═══════════════════════════════════════════════════════════════════════

def train_quant_model(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    masks_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    masks_val: Dict[str, np.ndarray],
    n_features: int,
    epochs: int = QUANT_EPOCHS,
    lr: float = QUANT_LEARNING_RATE,
    batch_size: int = QUANT_BATCH_SIZE,
    callback=None,
    augment: bool = True,
) -> Tuple["QuantHybridModel", List[dict]]:
    """
    Train the hybrid quant model with multi-task loss.

    Returns (model, history).
    """
    model = QuantHybridModel(
        n_features=n_features,
        d_model=QUANT_HIDDEN_SIZE,
        n_heads=QUANT_NUM_HEADS,
        n_layers=QUANT_NUM_LAYERS,
        dropout=QUANT_DROPOUT,
    ).to(DEVICE)

    criterion = QuantMultiTaskLoss().to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Build tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    target_names = list(y_train.keys())
    y_tr_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in y_train.items()}
    y_val_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in y_val.items()}
    m_tr_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in masks_train.items()}
    m_val_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in masks_val.items()}

    n_train = len(X_train)
    n_val = len(X_val)

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n_train)
        train_losses = []

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]

            xb = X_tr_t[idx].to(DEVICE)

            # Data augmentation
            if augment:
                noise = torch.randn_like(xb) * AUGMENT_NOISE_STD
                xb = xb + noise
                scale = 1.0 + (torch.rand(xb.size(0), 1, 1, device=DEVICE) * 2 - 1) * AUGMENT_SCALE_JITTER
                xb = xb * scale

            yb = {k: y_tr_t[k][idx].to(DEVICE) for k in target_names}
            mb = {k: m_tr_t[k][idx].to(DEVICE) for k in target_names}

            outputs = model(xb)
            loss, _ = criterion(outputs, yb, mb)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                end = min(start + batch_size, n_val)
                xb = X_val_t[start:end].to(DEVICE)
                yb = {k: y_val_t[k][start:end].to(DEVICE) for k in target_names}
                mb = {k: m_val_t[k][start:end].to(DEVICE) for k in target_names}
                outputs = model(xb)
                loss, breakdown = criterion(outputs, yb, mb)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses)) if val_losses else avg_train

        history.append({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
        })

        logger.info(
            "[quant] Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
            epoch, epochs, avg_train, avg_val, optimiser.param_groups[0]["lr"],
        )

        if callback:
            callback(epoch, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= QUANT_EARLY_STOP:
                logger.info("[quant] Early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


# ═══════════════════════════════════════════════════════════════════════
# 6. Inference
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_quant(
    model: QuantHybridModel,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Run inference on the hybrid model.

    Returns dict with numpy arrays:
      direction, return_1d, return_5d, return_30d, excess,
      branch_weights, feat_weights
    """
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # Process in batches to avoid OOM
    batch_size = 256
    results = {k: [] for k in [
        "direction", "return_1d", "return_5d", "return_30d",
        "excess", "branch_weights", "feat_weights",
    ]}

    for start in range(0, len(tensor), batch_size):
        end = min(start + batch_size, len(tensor))
        batch = tensor[start:end]
        out = model(batch)
        for k in results:
            val = out[k].cpu().numpy()
            results[k].append(val)

    return {k: np.concatenate(v, axis=0) for k, v in results.items()}


# ═══════════════════════════════════════════════════════════════════════
# 7. Model persistence
# ═══════════════════════════════════════════════════════════════════════

import os
import json
from datetime import datetime

from backend.config import MODEL_DIR


def _quant_model_dir(symbol: str) -> str:
    safe = symbol.upper().replace("/", "_").replace("\\", "_").strip()
    path = os.path.join(MODEL_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


def save_quant_model(model: QuantHybridModel, symbol: str, meta: dict):
    """Save quant model checkpoint + meta."""
    d = _quant_model_dir(symbol)
    torch.save(model.state_dict(), os.path.join(d, "quant_model.pt"))
    meta["updated"] = datetime.utcnow().isoformat()
    meta["profile"] = "quant"
    with open(os.path.join(d, "quant_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Version snapshot
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    ver_dir = os.path.join(d, "versions_quant", version)
    os.makedirs(ver_dir, exist_ok=True)

    import shutil
    for src_name in ["quant_model.pt", "quant_meta.json", "quant_scaler.pkl",
                     "quant_calibrator.pkl", "quant_metrics.json"]:
        src = os.path.join(d, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ver_dir, src_name))

    latest = {
        "symbol": symbol.upper(),
        "profile": "quant",
        "version": version,
        "updated": meta.get("updated"),
    }
    with open(os.path.join(d, "quant_latest.json"), "w", encoding="utf-8") as f:
        json.dump(latest, f, ensure_ascii=False, indent=2)


def load_quant_model(symbol: str, n_features: int) -> Optional[QuantHybridModel]:
    """Load a previously trained quant model."""
    d = _quant_model_dir(symbol)
    path = os.path.join(d, "quant_model.pt")
    if not os.path.exists(path):
        return None
    model = QuantHybridModel(
        n_features=n_features,
        d_model=QUANT_HIDDEN_SIZE,
        n_heads=QUANT_NUM_HEADS,
        n_layers=QUANT_NUM_LAYERS,
        dropout=QUANT_DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_quant_meta(symbol: str) -> Optional[dict]:
    d = _quant_model_dir(symbol)
    path = os.path.join(d, "quant_meta.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quant_model_exists(symbol: str) -> bool:
    d = _quant_model_dir(symbol)
    return os.path.exists(os.path.join(d, "quant_model.pt"))


def save_quant_scaler(scaler, symbol: str):
    import joblib
    d = _quant_model_dir(symbol)
    joblib.dump(scaler, os.path.join(d, "quant_scaler.pkl"))


def load_quant_scaler(symbol: str):
    import joblib
    d = _quant_model_dir(symbol)
    path = os.path.join(d, "quant_scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def save_quant_metrics(symbol: str, metrics: dict):
    d = _quant_model_dir(symbol)
    with open(os.path.join(d, "quant_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_quant_metrics(symbol: str) -> Optional[dict]:
    d = _quant_model_dir(symbol)
    path = os.path.join(d, "quant_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
