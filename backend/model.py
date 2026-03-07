"""
LSTM-based stock price prediction model (PyTorch).
Supports GPU training/inference, model save/load, and trend classification.

Storage layout (per symbol):
    models/{SYMBOL}/model.pt
    models/{SYMBOL}/scaler.pkl
    models/{SYMBOL}/meta.json
    models/{SYMBOL}/metrics.json
    models/{SYMBOL}/latest.json
    models/{SYMBOL}/versions/{version}/...
"""
import os, json, logging, shutil
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
import joblib

from backend.config import (
    DEVICE, MODEL_DIR, HIDDEN_SIZE, NUM_LAYERS,
    DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    EARLY_STOP_PATIENCE, SEQUENCE_LENGTH, FEATURE_COLS, TARGET_COL,
    AUGMENT_NOISE_STD, AUGMENT_SCALE_JITTER,
)

logger = logging.getLogger("vizigenesis.model")


def _normalize_profile(profile: str = "simple") -> str:
    raw = (profile or "simple").strip().lower()
    if raw in {"pro", "professional", "advanced"}:
        return "pro"
    return "simple"


def _profile_suffix(profile: str = "simple") -> str:
    p = _normalize_profile(profile)
    return "" if p == "simple" else f"_{p}"


# ═══════════════════════════════════════════════════════════════════════
# 1. LSTM Network
# ═══════════════════════════════════════════════════════════════════════
class StockLSTM(nn.Module):
    """
    Multi-layer LSTM → fully-connected regression head.
    Input shape : (batch, seq_len, num_features)
    Output shape: (batch, 1)  — predicted scaled Close price
    """
    def __init__(self, input_size: int = len(FEATURE_COLS),
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)        # (B, T, H)
        last_hidden = lstm_out[:, -1, :]   # take last time-step
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)    # (B,)


class StockGRU(nn.Module):
    """GRU model used as a second learner for ensemble."""
    def __init__(self, input_size: int = len(FEATURE_COLS),
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


class EnsembleRegressor:
    """Weighted ensemble wrapper for multiple torch sequence models."""
    def __init__(self, members: List[Tuple[str, nn.Module]], weights: Optional[List[float]] = None):
        self.members = members
        if not weights or len(weights) != len(members):
            weights = [1.0 / len(members)] * len(members)
        total = float(sum(weights)) if weights else 1.0
        self.weights = [w / total for w in weights]

    def eval(self):
        for _, model in self.members:
            model.eval()

    @torch.no_grad()
    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        preds = []
        for _, model in self.members:
            preds.append(model(tensor).cpu().numpy())
        stacked = np.stack(preds, axis=0)  # (M, N)
        weighted = np.tensordot(np.array(self.weights), stacked, axes=(0, 0))
        return weighted


# ═══════════════════════════════════════════════════════════════════════
# 2. Training routine
# ═══════════════════════════════════════════════════════════════════════
def _member_model_path(symbol: str, member: str, profile: str = "simple") -> str:
    return os.path.join(symbol_model_dir(symbol), f"model_{member}{_profile_suffix(profile)}.pt")


def _ensemble_path(symbol: str, profile: str = "simple") -> str:
    return os.path.join(symbol_model_dir(symbol), f"ensemble{_profile_suffix(profile)}.json")


def _train_torch_model(
    model_cls,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    symbol: str,
    member_name: str,
    profile: str = "simple",
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    callback=None,
    sample_weights: Optional[np.ndarray] = None,
    augment: bool = False,
    noise_std: float = AUGMENT_NOISE_STD,
    scale_jitter: float = AUGMENT_SCALE_JITTER,
):
    model = model_cls(input_size=int(X_train.shape[2])).to(DEVICE)
    criterion = nn.SmoothL1Loss(beta=0.02)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    # ── Time-weighted sampling (recent data sampled more often) ────────
    if sample_weights is not None and len(sample_weights) == len(X_train):
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(X_train),
            replacement=True,
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_dl = DataLoader(val_ds, batch_size=batch_size)

    history = []
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # ── Data augmentation (training only) ─────────────────────
            if augment:
                # Gaussian noise injection
                noise = torch.randn_like(xb) * noise_std
                xb = xb + noise
                # Random volatility scaling per sample
                scale = 1.0 + (
                    torch.rand(xb.size(0), 1, 1, device=DEVICE) * 2 - 1
                ) * scale_jitter
                xb = xb * scale

            pred = model(xb)
            loss = criterion(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses)) if val_losses else avg_train
        scheduler.step(avg_val)
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        logger.info(f"[{symbol}:{member_name}] Epoch {epoch}/{epochs} train={avg_train:.6f} val={avg_val:.6f}")

        if callback:
            callback(epoch, avg_train, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), _member_model_path(symbol, member_name, profile=profile))
            if member_name == "lstm":
                _save_model(model, symbol, profile=profile)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"[{symbol}:{member_name}] Early stop at epoch {epoch}")
                break

    member_ckpt = _member_model_path(symbol, member_name, profile=profile)
    if os.path.exists(member_ckpt):
        model.load_state_dict(torch.load(member_ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model, history


def _val_rmse(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray) -> float:
    with torch.no_grad():
        pred = predict(model, X_val)
    return float(np.sqrt(np.mean((pred - y_val) ** 2)))


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    symbol: str,
    profile: str = "simple",
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    callback=None,          # optional async callback(epoch, train_loss, val_loss)
    sample_weights: Optional[np.ndarray] = None,
    enable_augmentation: bool = True,
) -> Tuple[StockLSTM, list]:
    """
    Train LSTM+GRU ensemble on prepared sequences. Uses GPU if available.
    Returns (trained_model, loss_history).

    Parameters
    ----------
    sample_weights : per-sample time-based weights (len == len(X_train)).
        Recent data is sampled more often during training.
    enable_augmentation : add noise injection + volatility scaling to prevent overfitting.
    """
    lstm_model, lstm_history = _train_torch_model(
        StockLSTM, X_train, y_train, X_val, y_val,
        symbol=symbol, member_name="lstm", profile=profile, epochs=epochs,
        lr=lr, batch_size=batch_size, callback=callback,
        sample_weights=sample_weights, augment=enable_augmentation,
    )

    gru_model, _ = _train_torch_model(
        StockGRU, X_train, y_train, X_val, y_val,
        symbol=symbol, member_name="gru", profile=profile, epochs=epochs,
        lr=lr, batch_size=batch_size, callback=None,
        sample_weights=sample_weights, augment=enable_augmentation,
    )

    rmse_lstm = max(_val_rmse(lstm_model, X_val, y_val), 1e-8)
    rmse_gru = max(_val_rmse(gru_model, X_val, y_val), 1e-8)

    inv_lstm = 1.0 / rmse_lstm
    inv_gru = 1.0 / rmse_gru
    total = inv_lstm + inv_gru
    weights = [inv_lstm / total, inv_gru / total]

    ensemble = EnsembleRegressor(
        members=[("lstm", lstm_model), ("gru", gru_model)],
        weights=weights,
    )

    val_pred = ensemble.predict_numpy(X_val)
    bias_correction = float(np.mean(y_val - val_pred))

    ensemble_payload = {
        "members": ["lstm", "gru"],
        "profile": _normalize_profile(profile),
        "input_size": int(X_train.shape[2]),
        "weights": weights,
        "val_rmse": {"lstm": rmse_lstm, "gru": rmse_gru},
        "bias_correction_scaled": bias_correction,
        "updated": datetime.utcnow().isoformat(),
    }
    with open(_ensemble_path(symbol, profile=profile), "w", encoding="utf-8") as f:
        json.dump(ensemble_payload, f, ensure_ascii=False, indent=2)
    return ensemble, lstm_history


# ═══════════════════════════════════════════════════════════════════════
# 3. Prediction & trend helpers
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model: StockLSTM, X: np.ndarray) -> np.ndarray:
    """Run inference on numpy array, returns predictions as numpy."""
    if isinstance(model, EnsembleRegressor):
        return model.predict_numpy(X)
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return model(tensor).cpu().numpy()


def classify_trend(current: float, predicted: float, threshold: float = 0.005) -> str:
    """Classify next-day trend with a confidence dead-zone."""
    pct = (predicted - current) / current if current else 0
    if pct > threshold:
        return "UP"
    elif pct < -threshold:
        return "DOWN"
    return "NEUTRAL"


def compute_confidence(current: float, predicted: float) -> float:
    """Simple confidence score: magnitude of predicted % change, 0-100."""
    pct = abs((predicted - current) / current) if current else 0
    return min(round(pct * 1000, 1), 100.0)  # scale for readability


def long_term_trend(predictions: np.ndarray, window: int = 10) -> str:
    """Determine long-term trend from a series of predictions using SMA slope."""
    if len(predictions) < window:
        window = max(2, len(predictions))
    sma = np.convolve(predictions, np.ones(window)/window, mode='valid')
    if len(sma) < 2:
        return "NEUTRAL"
    slope = sma[-1] - sma[0]
    if slope > 0.005:
        return "BULLISH"
    elif slope < -0.005:
        return "BEARISH"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
# 4. Model persistence
# ═══════════════════════════════════════════════════════════════════════
def _safe_symbol(symbol: str) -> str:
    return symbol.upper().replace("/", "_").replace("\\", "_").strip()


def symbol_model_dir(symbol: str) -> str:
    path = os.path.join(MODEL_DIR, _safe_symbol(symbol))
    os.makedirs(path, exist_ok=True)
    return path


def _versions_dir(symbol: str, profile: str = "simple") -> str:
    p = _normalize_profile(profile)
    folder = "versions" if p == "simple" else f"versions_{p}"
    path = os.path.join(symbol_model_dir(symbol), folder)
    os.makedirs(path, exist_ok=True)
    return path


def _latest_pointer_path(symbol: str, profile: str = "simple") -> str:
    suffix = _profile_suffix(profile)
    return os.path.join(symbol_model_dir(symbol), f"latest{suffix}.json")


def _model_path(symbol: str, profile: str = "simple") -> str:
    suffix = _profile_suffix(profile)
    return os.path.join(symbol_model_dir(symbol), f"model{suffix}.pt")


def _scaler_path(symbol: str, profile: str = "simple") -> str:
    suffix = _profile_suffix(profile)
    return os.path.join(symbol_model_dir(symbol), f"scaler{suffix}.pkl")


def _meta_path(symbol: str, profile: str = "simple") -> str:
    suffix = _profile_suffix(profile)
    return os.path.join(symbol_model_dir(symbol), f"meta{suffix}.json")


def _metrics_path(symbol: str, profile: str = "simple") -> str:
    suffix = _profile_suffix(profile)
    return os.path.join(symbol_model_dir(symbol), f"metrics{suffix}.json")


def _legacy_model_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f"{symbol.upper()}_lstm.pt")


def _legacy_scaler_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f"{symbol.upper()}_scaler.pkl")


def _legacy_meta_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f"{symbol.upper()}_meta.json")


def _latest_version_from_pointer(symbol: str, profile: str = "simple") -> Optional[str]:
    pointer = _latest_pointer_path(symbol, profile=profile)
    if not os.path.exists(pointer):
        return None
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("version")
    except Exception:
        return None


def get_latest_version_info(symbol: str, profile: str = "simple") -> Optional[dict]:
    pointer = _latest_pointer_path(symbol, profile=profile)
    if not os.path.exists(pointer):
        return None
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_version_model_path(symbol: str, profile: str = "simple") -> Optional[str]:
    version = _latest_version_from_pointer(symbol, profile=profile)
    if not version:
        return None
    path = os.path.join(_versions_dir(symbol, profile=profile), version, "model.pt")
    return path if os.path.exists(path) else None


def _latest_version_scaler_path(symbol: str, profile: str = "simple") -> Optional[str]:
    version = _latest_version_from_pointer(symbol, profile=profile)
    if not version:
        return None
    path = os.path.join(_versions_dir(symbol, profile=profile), version, "scaler.pkl")
    return path if os.path.exists(path) else None


def _snapshot_version(symbol: str, info: dict, profile: str = "simple"):
    """Create a versioned snapshot after successful training/meta save."""
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    ver_dir = os.path.join(_versions_dir(symbol, profile=profile), version)
    os.makedirs(ver_dir, exist_ok=True)

    files = [
        (_model_path(symbol, profile=profile), "model.pt"),
        (_member_model_path(symbol, "lstm", profile=profile), "model_lstm.pt"),
        (_member_model_path(symbol, "gru", profile=profile), "model_gru.pt"),
        (_ensemble_path(symbol, profile=profile), "ensemble.json"),
        (_scaler_path(symbol, profile=profile), "scaler.pkl"),
        (_meta_path(symbol, profile=profile), "meta.json"),
        (_metrics_path(symbol, profile=profile), "metrics.json"),
    ]
    for src, dst_name in files:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ver_dir, dst_name))

    latest_payload = {
        "symbol": _safe_symbol(symbol),
        "profile": _normalize_profile(profile),
        "version": version,
        "updated": info.get("updated"),
        "path": ver_dir,
    }
    with open(_latest_pointer_path(symbol, profile=profile), "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)


def get_symbol_artifact_paths(symbol: str, profile: str = "simple") -> Dict[str, Optional[str]]:
    """Return active and latest-version artifact paths for packaging/inspection."""
    latest_version = _latest_version_from_pointer(symbol, profile=profile)
    latest_dir = os.path.join(_versions_dir(symbol, profile=profile), latest_version) if latest_version else None
    return {
        "symbol_dir": symbol_model_dir(symbol),
        "profile": _normalize_profile(profile),
        "active_model": _model_path(symbol, profile=profile) if os.path.exists(_model_path(symbol, profile=profile)) else None,
        "active_scaler": _scaler_path(symbol, profile=profile) if os.path.exists(_scaler_path(symbol, profile=profile)) else None,
        "active_meta": _meta_path(symbol, profile=profile) if os.path.exists(_meta_path(symbol, profile=profile)) else None,
        "active_metrics": _metrics_path(symbol, profile=profile) if os.path.exists(_metrics_path(symbol, profile=profile)) else None,
        "latest_version": latest_version,
        "latest_dir": latest_dir,
    }


def migrate_legacy_artifacts() -> Dict[str, object]:
    """
    Move old flat files from models root into models/{SYMBOL}/ structure.
    This prevents new/old symbol artifacts from being mixed at root.
    """
    suffix_map = {
        "_lstm.pt": "model.pt",
        "_scaler.pkl": "scaler.pkl",
        "_meta.json": "meta.json",
        "_metrics.json": "metrics.json",
        "_artifact.zip": None,
    }

    moved = []
    if not os.path.exists(MODEL_DIR):
        return {"moved_count": 0, "moved": moved}

    for name in os.listdir(MODEL_DIR):
        src = os.path.join(MODEL_DIR, name)
        if not os.path.isfile(src):
            continue

        matched_suffix = None
        for suffix in suffix_map:
            if name.endswith(suffix):
                matched_suffix = suffix
                break
        if not matched_suffix:
            continue

        symbol = name[: -len(matched_suffix)].strip()
        if not symbol:
            continue

        dst_dir = symbol_model_dir(symbol)
        if matched_suffix == "_artifact.zip":
            dst_name = f"{_safe_symbol(symbol)}_artifact.zip"
        else:
            dst_name = suffix_map[matched_suffix]

        dst = os.path.join(dst_dir, dst_name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue

        if os.path.exists(dst):
            os.remove(src)
            continue

        shutil.move(src, dst)
        moved.append({"from": src, "to": dst})

    return {"moved_count": len(moved), "moved": moved}


def _save_model(model: StockLSTM, symbol: str, profile: str = "simple"):
    torch.save(model.state_dict(), _model_path(symbol, profile=profile))


def save_scaler(scaler: MinMaxScaler, symbol: str, profile: str = "simple"):
    joblib.dump(scaler, _scaler_path(symbol, profile=profile))


def load_scaler(symbol: str, profile: str = "simple") -> Optional[MinMaxScaler]:
    path = _scaler_path(symbol, profile=profile)
    if os.path.exists(path):
        return joblib.load(path)

    latest_path = _latest_version_scaler_path(symbol, profile=profile)
    if latest_path:
        return joblib.load(latest_path)

    if _normalize_profile(profile) != "simple":
        return None

    legacy = _legacy_scaler_path(symbol)
    return joblib.load(legacy) if os.path.exists(legacy) else None


def save_meta(symbol: str, info: dict, profile: str = "simple"):
    info["updated"] = datetime.utcnow().isoformat()
    info["profile"] = _normalize_profile(profile)
    with open(_meta_path(symbol, profile=profile), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    _snapshot_version(symbol, info, profile=profile)


def load_meta(symbol: str, profile: str = "simple") -> Optional[dict]:
    path = _meta_path(symbol, profile=profile)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    if _normalize_profile(profile) != "simple":
        return None

    legacy = _legacy_meta_path(symbol)
    if os.path.exists(legacy):
        with open(legacy, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_model_weights(model: StockLSTM, symbol: str, profile: str = "simple") -> StockLSTM:
    path = _model_path(symbol, profile=profile)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        return model

    latest_path = _latest_version_model_path(symbol, profile=profile)
    if latest_path:
        model.load_state_dict(torch.load(latest_path, map_location=DEVICE, weights_only=True))
        return model

    if _normalize_profile(profile) != "simple":
        return model

    legacy = _legacy_model_path(symbol)
    if os.path.exists(legacy):
        model.load_state_dict(torch.load(legacy, map_location=DEVICE, weights_only=True))
    return model


def load_trained_model(symbol: str, profile: str = "simple") -> Optional[StockLSTM]:
    """Load a previously trained model from disk."""
    meta = load_meta(symbol, profile=profile) or {}
    input_size = int(meta.get("input_size") or len(meta.get("features") or FEATURE_COLS))

    ens_path = _ensemble_path(symbol, profile=profile)
    lstm_path = _member_model_path(symbol, "lstm", profile=profile)
    gru_path = _member_model_path(symbol, "gru", profile=profile)
    if os.path.exists(ens_path) and os.path.exists(lstm_path) and os.path.exists(gru_path):
        with open(ens_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ens_input_size = int(payload.get("input_size") or input_size)
        lstm = StockLSTM(input_size=ens_input_size).to(DEVICE)
        lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE, weights_only=True))
        lstm.eval()
        gru = StockGRU(input_size=ens_input_size).to(DEVICE)
        gru.load_state_dict(torch.load(gru_path, map_location=DEVICE, weights_only=True))
        gru.eval()
        return EnsembleRegressor(
            members=[("lstm", lstm), ("gru", gru)],
            weights=payload.get("weights", [0.5, 0.5]),
        )

    path = _model_path(symbol, profile=profile)
    if not os.path.exists(path):
        latest_path = _latest_version_model_path(symbol, profile=profile)
        if latest_path:
            path = latest_path
        else:
            if _normalize_profile(profile) != "simple":
                return None
            legacy = _legacy_model_path(symbol)
            if not os.path.exists(legacy):
                return None
            path = legacy
    model = StockLSTM(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def model_exists(symbol: str, profile: str = "simple") -> bool:
    return (
        (os.path.exists(_member_model_path(symbol, "lstm", profile=profile)) and os.path.exists(_member_model_path(symbol, "gru", profile=profile)))
        or os.path.exists(_ensemble_path(symbol, profile=profile))
        or os.path.exists(_model_path(symbol, profile=profile))
        or (_latest_version_model_path(symbol, profile=profile) is not None)
        or (_normalize_profile(profile) == "simple" and os.path.exists(_legacy_model_path(symbol)))
    )


def get_ensemble_info(symbol: str, profile: str = "simple") -> Optional[dict]:
    path = _ensemble_path(symbol, profile=profile)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
