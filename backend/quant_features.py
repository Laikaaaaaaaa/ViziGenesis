"""
Quant-mode feature engineering for institutional-grade predictions.

Adds technical features beyond the PRO set:
  - Stochastic RSI (14-period)
  - Rate of Change (10, 20)
  - Momentum (10, 20)
  - EMA50
  - VWAP (Volume Weighted Average Price)
  - Bollinger Band Width
  - Historical Volatility (20-day, 60-day)
  - Lag features: past returns at lags 1, 2, 3, 5, 10, 20
  - Rolling Z-Score normalization for Close

Also provides target generation for multi-task training:
  - Direction (1 = UP, 0 = DOWN)
  - Forward returns at 1d, 5d, 30d horizons
  - Excess return vs NASDAQ benchmark
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import QUANT_FEATURE_COLS, HORIZONS

logger = logging.getLogger("vizigenesis.quant_features")


# ═══════════════════════════════════════════════════════════════════════
# 1. Extended technical indicators
# ═══════════════════════════════════════════════════════════════════════

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.Series:
    """
    Stochastic RSI = (RSI - RSI_low) / (RSI_high - RSI_low) over `period`.
    Returns the %K line (smoothed).
    """
    rsi = _compute_rsi(close, period)
    rsi_low = rsi.rolling(period, min_periods=period).min()
    rsi_high = rsi.rolling(period, min_periods=period).max()
    stoch_rsi = ((rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)).fillna(0.5)
    # Smooth with SMA
    return stoch_rsi.rolling(smooth_k, min_periods=1).mean()


def add_quant_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full quant feature set from raw OHLCV data.
    Expects columns: Open, High, Low, Close, Volume.
    Returns DataFrame with all technical columns added.
    """
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    vol = out["Volume"].astype(float).replace(0, np.nan).fillna(1)
    returns = close.pct_change().fillna(0)

    # ── Classic indicators (same as PRO) ──────────────────────────────
    out["MA20"] = close.rolling(20, min_periods=20).mean()
    out["MA50"] = close.rolling(50, min_periods=50).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["RSI"] = _compute_rsi(close, period=14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26

    std20 = close.rolling(20, min_periods=20).std()
    bb_upper = out["MA20"] + 2 * std20
    bb_lower = out["MA20"] - 2 * std20
    bb_width_raw = (bb_upper - bb_lower).replace(0, np.nan)
    out["Bollinger_Band"] = ((close - bb_lower) / bb_width_raw).clip(lower=0, upper=1)
    out["OBV"] = (np.sign(close.diff().fillna(0)) * vol).cumsum()

    out["Volume_Change"] = vol.pct_change().fillna(0).clip(-5, 5)
    out["Volatility"] = returns.rolling(20, min_periods=20).std().fillna(0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=14).mean()

    # ── New quant indicators ──────────────────────────────────────────
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["Stochastic_RSI"] = _stochastic_rsi(close, period=14)

    # Rate of Change
    out["ROC_10"] = close.pct_change(periods=10).fillna(0) * 100
    out["ROC_20"] = close.pct_change(periods=20).fillna(0) * 100

    # Momentum (price difference)
    out["Momentum_10"] = (close - close.shift(10)).fillna(0)
    out["Momentum_20"] = (close - close.shift(20)).fillna(0)

    # VWAP (session-level approximation using rolling)
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * vol).rolling(20, min_periods=1).sum()
    cum_vol = vol.rolling(20, min_periods=1).sum()
    out["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Bollinger Band Width (normalised by middle band)
    out["BB_Width"] = (bb_width_raw / out["MA20"].replace(0, np.nan)).fillna(0)

    # Historical Volatility (annualised)
    out["Hist_Vol_20"] = returns.rolling(20, min_periods=20).std().fillna(0) * np.sqrt(252)
    out["Hist_Vol_60"] = returns.rolling(60, min_periods=60).std().fillna(0) * np.sqrt(252)

    # ── Lag features: past returns ────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"Return_Lag_{lag}"] = returns.shift(lag).fillna(0) * 100  # in %

    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. Rolling Z-Score normalisation (applied per-feature)
# ═══════════════════════════════════════════════════════════════════════

def rolling_zscore(df: pd.DataFrame, window: int = 60, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply rolling z-score normalisation to selected columns.
    z = (x - rolling_mean) / rolling_std

    This makes each feature locally stationary — important for time-series models.
    """
    out = df.copy()
    target_cols = cols or [c for c in out.columns if c not in ("Open", "High", "Low", "Close", "Volume")]

    for col in target_cols:
        if col not in out.columns:
            continue
        series = out[col].astype(float)
        roll_mean = series.rolling(window, min_periods=max(window // 2, 1)).mean()
        roll_std = series.rolling(window, min_periods=max(window // 2, 1)).std().replace(0, np.nan)
        out[f"{col}"] = ((series - roll_mean) / roll_std).fillna(0).clip(-5, 5)

    return out


# ═══════════════════════════════════════════════════════════════════════
# 3. Target variable generation for multi-task learning
# ═══════════════════════════════════════════════════════════════════════

def generate_quant_targets(
    df: pd.DataFrame,
    nasdaq_close: Optional[pd.Series] = None,
    horizons: List[int] = None,
) -> pd.DataFrame:
    """
    Generate target variables for quant multi-task training.

    Returns DataFrame with columns:
      - Direction      : 1 if next-day return > 0, else 0  (binary)
      - Return_1d      : 1-day forward return (%)
      - Return_5d      : 5-day forward return (%)
      - Return_30d     : 30-day forward return (%)
      - Excess_Return  : 1-day return minus NASDAQ 1-day return (%)

    NaN values indicate unavailable future data (end of dataset).
    """
    horizons = horizons or HORIZONS
    close = df["Close"].astype(float)
    targets = pd.DataFrame(index=df.index)

    # Direction: next-day
    next_ret = close.pct_change(periods=1).shift(-1)
    targets["Direction"] = (next_ret > 0).astype(float)

    # Multi-horizon forward returns
    for h in horizons:
        fwd_ret = (close.shift(-h) / close - 1) * 100  # in %
        targets[f"Return_{h}d"] = fwd_ret

    # Excess return vs NASDAQ (1-day)
    if nasdaq_close is not None and len(nasdaq_close) > 0:
        # Align NASDAQ to stock dates
        nasdaq_aligned = nasdaq_close.reindex(df.index, method="ffill")
        nasdaq_ret = nasdaq_aligned.pct_change(periods=1).shift(-1) * 100
        stock_ret = close.pct_change(periods=1).shift(-1) * 100
        targets["Excess_Return"] = (stock_ret - nasdaq_ret).fillna(0)
    else:
        targets["Excess_Return"] = 0.0

    return targets


def prepare_quant_sequences(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
    train_ratio: float = 0.8,
) -> Dict:
    """
    Create sliding-window sequences for multi-task quant training.

    Returns dict with:
      X_train, X_val: (N, seq_len, n_features)
      y_train, y_val: dict of {target_name: np.ndarray}
      scaler: fitted MinMaxScaler
      masks_train, masks_val: dict of {target_name: bool_mask} for NaN targets
    """
    from sklearn.preprocessing import MinMaxScaler

    data = features_df[feature_cols].values.astype(np.float32)
    n = len(data)

    split_idx = int(n * train_ratio)
    split_idx = max(split_idx, seq_len + 5)
    split_idx = min(split_idx, n - 1)

    scaler = MinMaxScaler()
    scaler.fit(data[:split_idx])
    scaled = scaler.transform(data)

    target_names = [c for c in targets_df.columns]
    target_data = {}
    for col in target_names:
        target_data[col] = targets_df[col].values.astype(np.float32)

    X, Y, masks = [], {t: [] for t in target_names}, {t: [] for t in target_names}
    for i in range(seq_len, n):
        X.append(scaled[i - seq_len: i])
        for t in target_names:
            val = target_data[t][i]
            Y[t].append(val if np.isfinite(val) else 0.0)
            masks[t].append(np.isfinite(target_data[t][i]))

    X = np.array(X, dtype=np.float32)
    for t in target_names:
        Y[t] = np.array(Y[t], dtype=np.float32)
        masks[t] = np.array(masks[t], dtype=bool)

    split = int(len(X) * train_ratio)
    split = max(split, 1)
    split = min(split, len(X) - 1)

    return {
        "X_train": X[:split],
        "X_val": X[split:],
        "y_train": {t: Y[t][:split] for t in target_names},
        "y_val": {t: Y[t][split:] for t in target_names},
        "masks_train": {t: masks[t][:split] for t in target_names},
        "masks_val": {t: masks[t][split:] for t in target_names},
        "scaler": scaler,
        "target_names": target_names,
    }
