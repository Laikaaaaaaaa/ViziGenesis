"""
ViziGenesis V2 — Hybrid Deep Learning Model (RTX 4090 Optimized)
===================================================================
Production-grade, multi-head architecture with:

  Shared Encoder (384-dim, 12 heads, 6 layers)
  ├── Temporal Fusion Transformer (Variable Selection + LSTM + Cross-Attention)
  ├── Bidirectional LSTM branch with temporal attention pooling
  ├── GRU branch with skip connections
  └── Transformer Encoder branch (pure self-attention)

  Output Heads (6 heads, multi-task learning)
  ├── Direction head   (binary classification — UP/DOWN)
  ├── Return heads     (regression — 1d, 5d, 30d)
  ├── Excess return    (regression — vs benchmark)
  └── Regime head      (3-class classification — bull/bear/sideways)

  Meta-gating: learned attention weights combine 4 branch outputs.
  Positional Encoding: sinusoidal + learned for 120-day sequences.
  Feature Groups: 157 features from 60+ open data sources.

Parameters: ~ 15-25M (RTX 4090 24GB).
"""
import logging, os, json, math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from backend.v2.config import (
    DEVICE, USE_AMP, D_MODEL, N_HEADS, N_LAYERS, DROPOUT,
    N_FEATURES, N_REGIMES, SEQ_LEN, SEED,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS,
    PATIENCE, GRAD_CLIP, GRAD_ACCUM_STEPS,
    LOSS_W_DIRECTION, LOSS_W_RET_1D, LOSS_W_RET_5D,
    LOSS_W_RET_30D, LOSS_W_EXCESS, LOSS_W_REGIME,
    AUGMENT_NOISE_STD, AUGMENT_SCALE_JITTER,
    V2_DIR, MODEL_DIR,
)

logger = logging.getLogger("vizigenesis.v2.model")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════
class GatedResidualNetwork(nn.Module):
    """Gated Residual Network from TFT paper."""
    def __init__(self, d_in: int, d_model: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_out)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        a = self.sigmoid(self.gate(h))
        h = self.fc2(h)
        return self.layer_norm(a * h + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network — learns feature importance via softmax gating."""
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.grn = GatedResidualNetwork(n_features, d_model, n_features, dropout)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model // 4, d_model, dropout)
            for _ in range(n_features)
        ])

    def forward(self, x):
        # x: (B, T, N_feat)
        B, T, N = x.shape

        # Flatten temporal for variable selection weights
        flat = x.reshape(B * T, N)
        weights = torch.softmax(self.grn(flat), dim=-1)  # (B*T, N)

        # Process each feature through its own GRN
        processed = []
        for i in range(N):
            feat_i = x[:, :, i:i+1]  # (B, T, 1)
            feat_i_flat = feat_i.reshape(B * T, 1)
            processed.append(self.feature_grns[i](feat_i_flat))  # (B*T, d_model)

        processed = torch.stack(processed, dim=1)  # (B*T, N, d_model)
        weights = weights.unsqueeze(-1)  # (B*T, N, 1)
        selected = (processed * weights).sum(dim=1)  # (B*T, d_model)
        selected = selected.reshape(B, T, -1)  # (B, T, d_model)

        # Return selected features and weights for interpretability
        feat_weights = weights.reshape(B, T, N).mean(dim=1)  # (B, N)
        return selected, feat_weights


class TemporalAttention(nn.Module):
    """Multi-head self-attention over time dimension."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_out))


# ═══════════════════════════════════════════════════════════════════════
# Branch architectures
# ═══════════════════════════════════════════════════════════════════════
class TFTBranch(nn.Module):
    """TFT-inspired: VSN → LSTM encoder → Multi-Head Attention."""
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm = nn.LSTM(
            d_model, d_model, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True, bidirectional=False,
        )
        self.attention = TemporalAttention(d_model, n_heads, dropout)
        self.grn_out = GatedResidualNetwork(d_model, d_model, d_model, dropout)

    def forward(self, x):
        # x: (B, T, N_feat)
        selected, feat_weights = self.vsn(x)  # (B, T, d_model), (B, N_feat)
        lstm_out, _ = self.lstm(selected)      # (B, T, d_model)
        attn_out = self.attention(lstm_out)     # (B, T, d_model)
        out = self.grn_out(attn_out[:, -1, :]) # (B, d_model)
        return out, feat_weights


class BiLSTMBranch(nn.Module):
    """Bidirectional LSTM with temporal attention pooling."""
    def __init__(self, n_features: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.bilstm = nn.LSTM(
            d_model, d_model // 2, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True, bidirectional=True,
        )
        self.attn_proj = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x)                       # (B, T, d_model)
        lstm_out, _ = self.bilstm(h)           # (B, T, d_model)
        attn_weights = torch.softmax(self.attn_proj(lstm_out), dim=1)  # (B, T, 1)
        pooled = (lstm_out * attn_weights).sum(dim=1)  # (B, d_model)
        return self.dropout(pooled)


class GRUBranch(nn.Module):
    """GRU with skip connections and last-hidden output."""
    def __init__(self, n_features: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.gru = nn.GRU(
            d_model, d_model, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.skip = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x)
        gru_out, _ = self.gru(h)
        last_hidden = gru_out[:, -1, :]
        skip = self.skip(x[:, -1, :])
        return self.dropout(self.norm(last_hidden + skip))


# ═══════════════════════════════════════════════════════════════════════
# NEW: Positional Encoding for Transformer branch
# ═══════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding + learned offset."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.learned_offset = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] + self.learned_offset
        return self.dropout(x)


class TransformerBranch(nn.Module):
    """Pure Transformer encoder branch with sinusoidal positional encoding."""
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=500, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(n_layers // 2, 2))
        self.pool_attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x)                       # (B, T, d_model)
        h = self.pos_enc(h)                    # add positional encoding
        h = self.encoder(h)                    # (B, T, d_model)
        # Attention pooling over time
        attn_w = torch.softmax(self.pool_attn(h), dim=1)  # (B, T, 1)
        pooled = (h * attn_w).sum(dim=1)       # (B, d_model)
        return self.dropout(self.norm(pooled))


# ═══════════════════════════════════════════════════════════════════════
# Main Hybrid Model
# ═══════════════════════════════════════════════════════════════════════
class HybridForecaster(nn.Module):
    """
    Production hybrid model with:
      • 4 branches (TFT, BiLSTM, GRU, Transformer)
      • Learned meta-gating for branch combination
      • 6 output heads: direction, 3× return, excess, regime
      • ~15-25M params optimized for RTX 4090
    """
    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        n_regimes: int = N_REGIMES,
        n_stocks: int = 25,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_stocks = n_stocks
        self.n_branches = 4

        # ── Optional stock embedding ──────────────────────────────────
        self.stock_emb = nn.Embedding(max(n_stocks, 1), d_model // 8)

        # ── Branches (4 parallel pathways) ────────────────────────────
        self.tft = TFTBranch(n_features, d_model, n_heads, n_layers, dropout)
        self.bilstm = BiLSTMBranch(n_features, d_model, n_layers, dropout)
        self.gru = GRUBranch(n_features, d_model, n_layers, dropout)
        self.transformer = TransformerBranch(n_features, d_model, n_heads, n_layers, dropout)

        # ── Meta-gating: learned weights to combine 4 branches ────────
        gate_input_dim = d_model * self.n_branches
        self.meta_gate = nn.Sequential(
            nn.Linear(gate_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_branches),
            nn.Softmax(dim=-1),
        )

        head_input_dim = d_model + d_model // 8

        # ── Output heads (deeper for better representation) ───────────
        # Direction (binary classification)
        self.head_direction = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Return regression heads  (1d, 5d, 30d) — deeper
        def _make_reg_head():
            return nn.Sequential(
                nn.Linear(head_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        self.head_ret_1d = _make_reg_head()
        self.head_ret_5d = _make_reg_head()
        self.head_ret_30d = _make_reg_head()

        # Excess return head
        self.head_excess = _make_reg_head()

        # Regime classification head
        self.head_regime = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_regimes),
        )

    def forward(self, x, stock_ids=None):
        """
        Args:
            x: (B, T, N_feat) — 157-feature sequences
            stock_ids: (B,) optional stock indices for embedding
        Returns: dict of head outputs
        """
        # Run 4 branches in parallel
        tft_out, feat_weights = self.tft(x)     # (B, d_model), (B, N_feat)
        bilstm_out = self.bilstm(x)             # (B, d_model)
        gru_out = self.gru(x)                   # (B, d_model)
        trans_out = self.transformer(x)          # (B, d_model)

        # Meta-gating
        concat = torch.cat([tft_out, bilstm_out, gru_out, trans_out], dim=-1)
        gate_weights = self.meta_gate(concat)    # (B, 4)

        # Weighted combination
        combined = (
            gate_weights[:, 0:1] * tft_out +
            gate_weights[:, 1:2] * bilstm_out +
            gate_weights[:, 2:3] * gru_out +
            gate_weights[:, 3:4] * trans_out
        )  # (B, d_model)

        # Stock embedding
        if stock_ids is not None:
            stock_emb = self.stock_emb(stock_ids)  # (B, d_model//8)
        else:
            stock_emb = torch.zeros(x.size(0), self.d_model // 8, device=x.device)

        h = torch.cat([combined, stock_emb], dim=-1)

        return {
            "direction": self.head_direction(h),         # (B, 1)
            "return_1d": self.head_ret_1d(h),            # (B, 1)
            "return_5d": self.head_ret_5d(h),            # (B, 1)
            "return_30d": self.head_ret_30d(h),          # (B, 1)
            "excess": self.head_excess(h),               # (B, 1)
            "regime": self.head_regime(h),               # (B, n_regimes)
            "branch_weights": gate_weights,              # (B, 4)
            "feat_weights": feat_weights,                # (B, N_feat)
        }


# ═══════════════════════════════════════════════════════════════════════
# Multi-task Loss
# ═══════════════════════════════════════════════════════════════════════
class MultiTaskLoss(nn.Module):
    """
    Combined loss:  regression + α * classification + β * regime

    Loss = w_dir * BCE(direction) +
           w_1d * SmoothL1(ret_1d) + w_5d * SmoothL1(ret_5d) +
           w_30d * SmoothL1(ret_30d) + w_ex * SmoothL1(excess) +
           w_regime * CE(regime)
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.02)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # Direction
        dir_pred = preds["direction"].squeeze(-1)
        dir_true = targets["direction"]
        mask_dir = ~torch.isnan(dir_true)
        if mask_dir.sum() > 0:
            losses["direction"] = self.bce(dir_pred[mask_dir], dir_true[mask_dir])
        else:
            losses["direction"] = torch.tensor(0.0, device=dir_pred.device)

        # Returns
        for key, weight_name in [
            ("return_1d", "ret_1d"), ("return_5d", "ret_5d"),
            ("return_30d", "ret_30d"), ("excess", "excess"),
        ]:
            pred = preds[key].squeeze(-1)
            true = targets[key]
            mask = ~torch.isnan(true)
            if mask.sum() > 0:
                losses[key] = self.smooth_l1(pred[mask], true[mask])
            else:
                losses[key] = torch.tensor(0.0, device=pred.device)

        # Regime
        regime_pred = preds["regime"]
        regime_true = targets["regime"]
        mask_reg = regime_true >= 0
        if mask_reg.sum() > 0:
            losses["regime"] = self.ce(regime_pred[mask_reg], regime_true[mask_reg])
        else:
            losses["regime"] = torch.tensor(0.0, device=regime_pred.device)

        # Weighted total
        total = (
            LOSS_W_DIRECTION * losses["direction"] +
            LOSS_W_RET_1D * losses.get("return_1d", 0) +
            LOSS_W_RET_5D * losses.get("return_5d", 0) +
            LOSS_W_RET_30D * losses.get("return_30d", 0) +
            LOSS_W_EXCESS * losses.get("excess", 0) +
            LOSS_W_REGIME * losses["regime"]
        )

        detail = {k: v.item() for k, v in losses.items()}
        return total, detail


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════
def train_hybrid_model(
    X_train: np.ndarray,
    y_dir_train: np.ndarray,
    y_ret_train: np.ndarray,    # (N, 4) — 1d, 5d, 30d, excess
    y_regime_train: np.ndarray,
    X_val: np.ndarray,
    y_dir_val: np.ndarray,
    y_ret_val: np.ndarray,
    y_regime_val: np.ndarray,
    stock_ids_train: Optional[np.ndarray] = None,
    stock_ids_val: Optional[np.ndarray] = None,
    n_stocks: int = 5,
    n_features: int = N_FEATURES,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    callback=None,
) -> Tuple[HybridForecaster, List[Dict]]:
    """
    Train the hybrid forecaster on panel data.
    Returns: (model, history)
    """
    model = HybridForecaster(
        n_features=n_features,
        n_stocks=n_stocks,
    ).to(DEVICE)

    criterion = MultiTaskLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Setup AMP
    scaler = GradScaler(enabled=USE_AMP)

    # Prepare datasets
    def _make_dataset(X, y_dir, y_ret, y_reg, s_ids):
        tensors = [
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_dir, dtype=torch.float32),
            torch.tensor(y_ret, dtype=torch.float32),
            torch.tensor(y_reg, dtype=torch.long),
        ]
        if s_ids is not None:
            tensors.append(torch.tensor(s_ids, dtype=torch.long))
        else:
            tensors.append(torch.zeros(len(X), dtype=torch.long))
        return TensorDataset(*tensors)

    train_ds = _make_dataset(X_train, y_dir_train, y_ret_train, y_regime_train, stock_ids_train)
    val_ds = _make_dataset(X_val, y_dir_val, y_ret_val, y_regime_val, stock_ids_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          pin_memory=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=0)

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    accum_steps = GRAD_ACCUM_STEPS

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "HybridForecaster: %d parameters (%.1fM), device=%s, AMP=%s, "
        "batch=%d×%d=%d effective, 4 branches, %d features",
        n_params, n_params / 1e6, DEVICE, USE_AMP,
        batch_size, accum_steps, batch_size * accum_steps, n_features,
    )

    for epoch in range(1, epochs + 1):
        # ── Training with gradient accumulation ───────────────────────
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            xb, yb_dir, yb_ret, yb_reg, sb = [t.to(DEVICE) for t in batch]

            # Data augmentation
            noise = torch.randn_like(xb) * AUGMENT_NOISE_STD
            scale = 1.0 + (torch.rand(xb.size(0), 1, 1, device=DEVICE) * 2 - 1) * AUGMENT_SCALE_JITTER
            xb = xb * scale + noise

            with autocast(enabled=USE_AMP):
                preds = model(xb, stock_ids=sb)
                targets = {
                    "direction": yb_dir,
                    "return_1d": yb_ret[:, 0],
                    "return_5d": yb_ret[:, 1],
                    "return_30d": yb_ret[:, 2],
                    "excess": yb_ret[:, 3],
                    "regime": yb_reg,
                }
                loss, _ = criterion(preds, targets)
                loss = loss / accum_steps  # scale for accumulation

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_losses.append(loss.item() * accum_steps)  # unscale for logging

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_details_agg = {}
        with torch.no_grad():
            for batch in val_dl:
                xb, yb_dir, yb_ret, yb_reg, sb = [t.to(DEVICE) for t in batch]
                preds = model(xb, stock_ids=sb)
                targets = {
                    "direction": yb_dir,
                    "return_1d": yb_ret[:, 0],
                    "return_5d": yb_ret[:, 1],
                    "return_30d": yb_ret[:, 2],
                    "excess": yb_ret[:, 3],
                    "regime": yb_reg,
                }
                loss, detail = criterion(preds, targets)
                val_losses.append(loss.item())
                for k, v in detail.items():
                    val_details_agg.setdefault(k, []).append(v)

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        val_detail = {k: float(np.mean(v)) for k, v in val_details_agg.items()}

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train, 6),
            "val_loss": round(avg_val, 6),
            "val_detail": val_detail,
            "lr": optimizer.param_groups[0]["lr"],
        })

        logger.info(
            "Epoch %d/%d  train=%.6f  val=%.6f  dir=%.4f  reg=%.4f  lr=%.2e",
            epoch, epochs, avg_train, avg_val,
            val_detail.get("direction", 0), val_detail.get("regime", 0),
            optimizer.param_groups[0]["lr"],
        )

        if callback:
            callback(epoch, avg_train, avg_val, val_detail)

        # ── Early stopping ────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (best_val=%.6f)", epoch, best_val_loss)
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


# ═══════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_hybrid(
    model: HybridForecaster,
    X: np.ndarray,
    stock_ids: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Run inference on numpy arrays → dict of numpy outputs."""
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    if stock_ids is not None:
        sid = torch.tensor(stock_ids, dtype=torch.long).to(DEVICE)
    else:
        sid = None

    with autocast(enabled=USE_AMP):
        preds = model(tensor, stock_ids=sid)

    result = {}
    for k, v in preds.items():
        result[k] = v.cpu().numpy()
    return result


# ═══════════════════════════════════════════════════════════════════════
# Model persistence
# ═══════════════════════════════════════════════════════════════════════
def _v2_model_dir() -> str:
    os.makedirs(V2_DIR, exist_ok=True)
    return V2_DIR


def save_v2_model(model: HybridForecaster, meta: Dict):
    """Save V2 panel-trained model."""
    d = _v2_model_dir()
    torch.save(model.state_dict(), os.path.join(d, "hybrid_model.pt"))
    meta["updated"] = datetime.utcnow().isoformat()
    meta["n_params"] = sum(p.numel() for p in model.parameters())
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("V2 model saved to %s", d)


def load_v2_model(
    n_features: int = N_FEATURES,
    n_stocks: int = 5,
) -> Optional[HybridForecaster]:
    """Load V2 panel-trained model."""
    path = os.path.join(_v2_model_dir(), "hybrid_model.pt")
    if not os.path.exists(path):
        return None
    model = HybridForecaster(n_features=n_features, n_stocks=n_stocks).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_v2_meta() -> Optional[Dict]:
    path = os.path.join(_v2_model_dir(), "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def v2_model_exists() -> bool:
    return os.path.exists(os.path.join(_v2_model_dir(), "hybrid_model.pt"))


def save_v2_scaler(scaler, label: str = "panel"):
    """Save scaler for V2 pipeline."""
    import joblib
    d = _v2_model_dir()
    joblib.dump(scaler, os.path.join(d, f"scaler_{label}.pkl"))


def load_v2_scaler(label: str = "panel"):
    """Load V2 scaler."""
    import joblib
    path = os.path.join(_v2_model_dir(), f"scaler_{label}.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def save_v2_metrics(metrics: Dict):
    d = _v2_model_dir()
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_v2_metrics() -> Optional[Dict]:
    path = os.path.join(_v2_model_dir(), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
