"""
ViziGenesis V2 — Unified Feature Engineering
===============================================
Produces 59 features from raw OHLCV + macro + sentiment + regime proxies.

Feature groups:
  OHLCV (5) + Classic TA (8) + Engineered (3) + Advanced TA (9) +
  Lag Returns (6) + FRED Macro (9) + FOMC Policy (3) +
  Market Context (5) + Sector/Commodity (4) + Sentiment (4) +
  Regime Proxies (3) = 59 total

All features are NaN-safe, forward-filled, and optionally z-score normalised.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.v2.config import (
    V2_FEATURE_COLS, FEAT_FRED, FEAT_FOMC, FEAT_MARKET,
    FEAT_SECTOR, FEAT_SENTIMENT, FEAT_REGIME_PROXY,
    HORIZONS, SEQ_LEN, N_REGIMES,
)

logger = logging.getLogger("vizigenesis.v2.features")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Technical indicators from OHLCV
# ═══════════════════════════════════════════════════════════════════════
def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _stochastic_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3) -> pd.Series:
    rsi = _compute_rsi(close, period)
    min_rsi = rsi.rolling(period, min_periods=1).min()
    max_rsi = rsi.rolling(period, min_periods=1).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)
    return stoch.rolling(smooth_k, min_periods=1).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to OHLCV dataframe."""
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # Classic technicals
    df["MA20"] = close.rolling(20, min_periods=1).mean()
    df["MA50"] = close.rolling(50, min_periods=1).mean()
    df["EMA20"] = close.ewm(span=20, min_periods=1).mean()
    df["EMA50"] = close.ewm(span=50, min_periods=1).mean()
    df["RSI"] = _compute_rsi(close)

    ema12 = close.ewm(span=12, min_periods=1).mean()
    ema26 = close.ewm(span=26, min_periods=1).mean()
    df["MACD"] = ema12 - ema26

    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.001)
    df["Bollinger_Band"] = (close - sma20) / std20.clip(lower=0.001)

    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    df["OBV"] = obv

    # Engineered
    df["Volume_Change"] = volume.pct_change().fillna(0).clip(-5, 5)
    df["Volatility"] = close.pct_change().rolling(20, min_periods=2).std().fillna(0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=1).mean()

    # Advanced technicals
    df["Stochastic_RSI"] = _stochastic_rsi(close)
    df["ROC_10"] = close.pct_change(10).fillna(0) * 100
    df["ROC_20"] = close.pct_change(20).fillna(0) * 100
    df["Momentum_10"] = close.diff(10).fillna(0)
    df["Momentum_20"] = close.diff(20).fillna(0)

    cum_vol = volume.cumsum()
    cum_vp = (close * volume).cumsum()
    df["VWAP"] = cum_vp / cum_vol.clip(lower=1)

    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_Width"] = (upper - lower) / sma20.clip(lower=0.001)

    df["Hist_Vol_20"] = close.pct_change().rolling(20, min_periods=2).std() * np.sqrt(252) * 100
    df["Hist_Vol_60"] = close.pct_change().rolling(60, min_periods=5).std() * np.sqrt(252) * 100

    # Lag returns
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"Return_Lag_{lag}"] = close.pct_change(lag).fillna(0) * 100

    # Regime proxies from price data
    df["Realised_Vol_20"] = df["Hist_Vol_20"].copy()

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.  Merge external data sources
# ═══════════════════════════════════════════════════════════════════════
def merge_fred_features(stock_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join FRED macro features onto stock DataFrame."""
    if fred_df.empty:
        for col in FEAT_FRED:
            if col not in stock_df.columns:
                stock_df[col] = 0.0
        return stock_df

    for col in fred_df.columns:
        if col in FEAT_FRED:
            aligned = fred_df[col].reindex(stock_df.index, method="ffill")
            stock_df[col] = aligned.bfill().fillna(0)
    # Fill any missing FRED columns
    for col in FEAT_FRED:
        if col not in stock_df.columns:
            stock_df[col] = 0.0
    return stock_df


def merge_fomc_features(stock_df: pd.DataFrame, fomc_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join FOMC policy features."""
    if fomc_df.empty:
        for col in FEAT_FOMC:
            if col not in stock_df.columns:
                stock_df[col] = 0.0
        return stock_df

    for col in FEAT_FOMC:
        if col in fomc_df.columns:
            stock_df[col] = fomc_df[col].reindex(stock_df.index).ffill().fillna(0)
        else:
            stock_df[col] = 0.0
    return stock_df


def merge_market_context(stock_df: pd.DataFrame, mkt_df: pd.DataFrame) -> pd.DataFrame:
    """Merge market context features (SP500, NASDAQ, VIX, etc.)."""
    if mkt_df.empty:
        for col in FEAT_MARKET:
            if col not in stock_df.columns:
                stock_df[col] = 0.0
        return stock_df

    for col in FEAT_MARKET:
        if col in mkt_df.columns:
            stock_df[col] = mkt_df[col].reindex(stock_df.index).ffill().bfill().fillna(0)
        elif col not in stock_df.columns:
            stock_df[col] = 0.0
    return stock_df


def merge_sector_features(stock_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sector/commodity return features."""
    if sector_df.empty:
        for col in FEAT_SECTOR:
            if col not in stock_df.columns:
                stock_df[col] = 0.0
        return stock_df

    for col in FEAT_SECTOR:
        if col in sector_df.columns:
            stock_df[col] = sector_df[col].reindex(stock_df.index).ffill().fillna(0)
        elif col not in stock_df.columns:
            stock_df[col] = 0.0
    return stock_df


def merge_sentiment_features(stock_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment features."""
    if sent_df.empty:
        for col in FEAT_SENTIMENT:
            if col not in stock_df.columns:
                stock_df[col] = 0.0
        return stock_df

    for col in FEAT_SENTIMENT:
        if col in sent_df.columns:
            stock_df[col] = sent_df[col].reindex(stock_df.index).ffill().fillna(0)
        elif col not in stock_df.columns:
            stock_df[col] = 0.0
    return stock_df


def add_regime_proxy_features(stock_df: pd.DataFrame, mkt_df: pd.DataFrame) -> pd.DataFrame:
    """Add regime proxy features from market data."""
    idx = stock_df.index

    # Realised_Vol_20 already computed in add_technical_indicators
    if "Realised_Vol_20" not in stock_df.columns:
        stock_df["Realised_Vol_20"] = (
            stock_df["Close"].pct_change().rolling(20, min_periods=2).std() * np.sqrt(252) * 100
        )

    # VIX regime z-score
    if "VIX" in stock_df.columns:
        vix = stock_df["VIX"]
    elif "VIX" in mkt_df.columns:
        vix = mkt_df["VIX"].reindex(idx).ffill().bfill()
    else:
        vix = pd.Series(20.0, index=idx)

    vix_mean = vix.rolling(60, min_periods=10).mean()
    vix_std = vix.rolling(60, min_periods=10).std().clip(lower=1)
    stock_df["VIX_Regime"] = ((vix - vix_mean) / vix_std).clip(-3, 3).fillna(0)

    # Market breadth proxy: SP500 rolling momentum
    if "SP500_Ret" in stock_df.columns:
        stock_df["Market_Breadth_Proxy"] = (
            stock_df["SP500_Ret"].rolling(20, min_periods=5).sum().fillna(0)
        )
    else:
        stock_df["Market_Breadth_Proxy"] = 0.0

    return stock_df


# ═══════════════════════════════════════════════════════════════════════
# 3.  Rolling z-score normalization
# ═══════════════════════════════════════════════════════════════════════
def rolling_zscore(df: pd.DataFrame, window: int = 60, clip: float = 5.0) -> pd.DataFrame:
    """Apply rolling z-score normalization to all numeric columns."""
    result = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        roll_mean = df[col].rolling(window, min_periods=10).mean()
        roll_std = df[col].rolling(window, min_periods=10).std().clip(lower=1e-8)
        result[col] = ((df[col] - roll_mean) / roll_std).clip(-clip, clip)
    return result.fillna(0)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Target generation
# ═══════════════════════════════════════════════════════════════════════
def generate_targets(
    stock_df: pd.DataFrame,
    nasdaq_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Generate multi-task targets:
      Direction   — 1 if next day return > 0, else 0
      Return_1d   — next-day return (%)
      Return_5d   — 5-day return (%)
      Return_30d  — 30-day return (%)
      Excess_Return — stock return - NASDAQ return (1d, %)
    """
    close = stock_df["Close"].astype(float)

    targets = pd.DataFrame(index=stock_df.index)
    targets["Return_1d"] = close.pct_change(-1).shift(-1) * -100  # forward return
    # Corrected: forward returns
    targets["Return_1d"] = (close.shift(-1) / close - 1) * 100
    targets["Return_5d"] = (close.shift(-5) / close - 1) * 100
    targets["Return_30d"] = (close.shift(-30) / close - 1) * 100
    targets["Direction"] = (targets["Return_1d"] > 0).astype(float)

    if nasdaq_close is not None and len(nasdaq_close) > 0:
        nq = nasdaq_close.reindex(stock_df.index).ffill()
        nq_ret = (nq.shift(-1) / nq - 1) * 100
        targets["Excess_Return"] = targets["Return_1d"] - nq_ret
    else:
        targets["Excess_Return"] = targets["Return_1d"]

    return targets


def generate_regime_labels(
    stock_df: pd.DataFrame,
    window: int = 60,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
) -> pd.Series:
    """
    Label market regime: 0=bull, 1=bear, 2=sideways.
    Based on rolling return + volatility.
    """
    close = stock_df["Close"].astype(float)
    rolling_ret = close.pct_change(window).fillna(0)
    rolling_vol = close.pct_change().rolling(window, min_periods=10).std().fillna(0)

    regime = pd.Series(2, index=stock_df.index, dtype=int)  # default sideways

    # Bull: positive returns, moderate volatility
    regime[rolling_ret > bull_threshold] = 0

    # Bear: negative returns
    regime[rolling_ret < bear_threshold] = 1

    # Override: very high volatility → sideways (choppy)
    vol_threshold = rolling_vol.quantile(0.85)
    high_vol = rolling_vol > vol_threshold
    sideways_cond = high_vol & (rolling_ret.abs() < bull_threshold)
    regime[sideways_cond] = 2

    regime.name = "Regime"
    return regime


# ═══════════════════════════════════════════════════════════════════════
# 5.  Sequence preparation for panel training
# ═══════════════════════════════════════════════════════════════════════
def prepare_panel_sequences(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    regime_labels: pd.Series,
    feature_cols: List[str],
    seq_len: int = SEQ_LEN,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
) -> Dict:
    """
    Create sequences for panel training from a single stock's data.
    Returns dict with X, y_direction, y_returns, y_regime, dates, scaler.
    """
    # Align all
    common_idx = features_df.index.intersection(targets_df.index).intersection(regime_labels.index)
    features_df = features_df.loc[common_idx]
    targets_df = targets_df.loc[common_idx]
    regime_labels = regime_labels.loc[common_idx]

    # Extract feature matrix
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    data = features_df[feature_cols].values.astype(np.float32)

    # Replace inf/nan
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    if scaler is None:
        if fit_scaler:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            scaled = scaler.transform(data)
        else:
            # Caller (panel builder) will apply global scaling later.
            scaled = data
    else:
        if fit_scaler:
            scaler.fit(data)
        scaled = scaler.transform(data)

    # Build sequences
    X_list, dir_list, ret_list, regime_list, date_list = [], [], [], [], []
    target_cols = ["Direction", "Return_1d", "Return_5d", "Return_30d", "Excess_Return"]

    for i in range(seq_len, len(scaled)):
        # Check if target is valid (not NaN for direction)
        if np.isnan(targets_df["Direction"].iloc[i]):
            continue

        X_list.append(scaled[i - seq_len: i])
        dir_list.append(targets_df["Direction"].iloc[i])

        ret_row = []
        for col in ["Return_1d", "Return_5d", "Return_30d", "Excess_Return"]:
            v = targets_df[col].iloc[i]
            ret_row.append(v if np.isfinite(v) else 0.0)
        ret_list.append(ret_row)

        regime_list.append(int(regime_labels.iloc[i]))
        date_list.append(features_df.index[i])

    if not X_list:
        return {"X": np.array([]), "n_samples": 0, "scaler": scaler}

    return {
        "X": np.array(X_list, dtype=np.float32),
        "y_direction": np.array(dir_list, dtype=np.float32),
        "y_returns": np.array(ret_list, dtype=np.float32),
        "y_regime": np.array(regime_list, dtype=np.int64),
        "dates": date_list,
        "scaler": scaler,
        "n_samples": len(X_list),
    }


# ═══════════════════════════════════════════════════════════════════════
# 6.  Full feature pipeline (orchestrator)
# ═══════════════════════════════════════════════════════════════════════
def build_full_features(
    stock_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    fomc_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Full feature pipeline: add TA, merge all external sources, ensure all
    V2_FEATURE_COLS are present and NaN-free.
    """
    if feature_cols is None:
        feature_cols = V2_FEATURE_COLS

    # Technical indicators
    df = add_technical_indicators(stock_df)

    # Merge external
    df = merge_fred_features(df, fred_df)
    df = merge_fomc_features(df, fomc_df)
    df = merge_market_context(df, market_df)
    df = merge_sector_features(df, sector_df)
    df = merge_sentiment_features(df, sentiment_df)
    df = add_regime_proxy_features(df, market_df)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Final NaN handling
    df = df.ffill().bfill()
    df[feature_cols] = df[feature_cols].fillna(0)

    # Replace inf
    df = df.replace([np.inf, -np.inf], 0)

    return df
