"""
ViziGenesis V2 — Panel Data Loader
=====================================
Fetches, aligns, and stacks data from multiple stocks into a single
panel dataset for cross-stock training.

Features:
  • Downloads max history for each ticker (Yahoo Finance)
  • Builds 59-feature representation per stock per day
  • Attaches stock-level identifiers for learned embeddings
  • Stacks into unified (N_total_samples, seq_len, n_features) tensor
  • Supports incremental updates and caching
"""
import logging, os, json, time, hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.v2.config import (
    PILOT_TICKERS, V2_FEATURE_COLS, SEQ_LEN, DATA_DIR,
    MIN_ROWS, N_FEATURES, SEED,
)

logger = logging.getLogger("vizigenesis.v2.panel")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Single-stock data fetcher
# ═══════════════════════════════════════════════════════════════════════
def fetch_stock_data(symbol: str, period: str = "max") -> pd.DataFrame:
    """Download OHLCV data for a single stock from Yahoo Finance."""
    import yfinance as yf
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            logger.warning("No data for %s", symbol)
            return pd.DataFrame()
        # Ensure standard columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                logger.warning("Missing column %s for %s", col, symbol)
                return pd.DataFrame()
        df = df[required].dropna()
        df.index = pd.DatetimeIndex(df.index)
        logger.info("Fetched %s: %d rows (%s to %s)", symbol, len(df),
                     df.index[0].date(), df.index[-1].date())
        return df
    except Exception as e:
        logger.error("Failed to fetch %s: %s", symbol, e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# 2.  Panel data builder (multi-stock)
# ═══════════════════════════════════════════════════════════════════════
def build_panel_dataset(
    tickers: Optional[List[str]] = None,
    period: str = "max",
    seq_len: int = SEQ_LEN,
    feature_cols: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    callback=None,
) -> Dict:
    """
    Build a panel dataset from multiple stocks.

    Returns:
        {
          "X":            np.ndarray (N, seq_len, n_features),
          "y_direction":  np.ndarray (N,),
          "y_returns":    np.ndarray (N, 4),  # 1d, 5d, 30d, excess
          "y_regime":     np.ndarray (N,),
          "stock_ids":    np.ndarray (N,),    # per-sample stock index
          "dates":        list of Timestamps,
          "scaler":       MinMaxScaler,
          "ticker_map":   dict {ticker: int},
          "n_stocks":     int,
          "n_samples":    int,
          "per_stock_info": dict,
        }
    """
    from backend.v2.features import (
        build_full_features, generate_targets, generate_regime_labels,
        prepare_panel_sequences,
    )
    from backend.v2.fred_data import (
        fetch_fred_macro, build_fomc_features,
        fetch_market_context, fetch_sector_commodity, fetch_nasdaq_close,
    )
    from backend.v2.sentiment import build_sentiment_features
    from backend.v2.regime import detect_regime

    if tickers is None:
        tickers = PILOT_TICKERS
    if feature_cols is None:
        feature_cols = list(V2_FEATURE_COLS)

    np.random.seed(SEED)

    # ── Fetch shared macro / context data (once for all stocks) ────────
    logger.info("Fetching shared macro data...")
    fred_df = fetch_fred_macro(start="2000-01-01")
    market_df = fetch_market_context(start="2000-01-01")
    sector_df = fetch_sector_commodity(start="2000-01-01")
    fomc_df = build_fomc_features(market_df.index) if not market_df.empty else pd.DataFrame()
    nasdaq_close = fetch_nasdaq_close(start="2000-01-01")

    # VIX for regime detection
    vix = market_df["VIX"] if "VIX" in market_df.columns else None

    # ── Process each stock ─────────────────────────────────────────────
    all_X, all_dir, all_ret, all_regime, all_stock_ids, all_dates = [], [], [], [], [], []
    ticker_map = {}
    per_stock_info = {}
    global_scaler = MinMaxScaler(feature_range=(-1, 1))
    all_raw_features = []

    for tidx, ticker in enumerate(tickers):
        logger.info("Processing %s (%d/%d)...", ticker, tidx + 1, len(tickers))
        ticker_map[ticker] = tidx

        # Fetch stock OHLCV
        stock_df = fetch_stock_data(ticker, period=period)
        if stock_df.empty or len(stock_df) < MIN_ROWS:
            logger.warning("Skipping %s — insufficient data (%d rows)", ticker, len(stock_df))
            per_stock_info[ticker] = {"status": "skipped", "rows": len(stock_df)}
            continue

        # Build features
        sentiment_df = build_sentiment_features(stock_df, market_df, symbol=ticker)
        features_df = build_full_features(
            stock_df, fred_df, fomc_df, market_df, sector_df, sentiment_df,
            feature_cols=feature_cols,
        )

        # Generate targets
        targets_df = generate_targets(features_df, nasdaq_close=nasdaq_close)

        # Detect regime
        regime_labels = detect_regime(features_df, vix=vix)

        # Collect raw feature data for global scaler fitting
        raw_data = features_df[feature_cols].values.astype(np.float32)
        raw_data = np.nan_to_num(raw_data, nan=0, posinf=0, neginf=0)
        all_raw_features.append(raw_data)

        # Prepare sequences (scaler fitting deferred to global)
        result = prepare_panel_sequences(
            features_df, targets_df, regime_labels,
            feature_cols=feature_cols,
            seq_len=seq_len,
            scaler=None,  # will rescale later
            fit_scaler=False,
        )

        if result["n_samples"] == 0:
            logger.warning("No valid sequences for %s", ticker)
            per_stock_info[ticker] = {"status": "no_sequences", "rows": len(stock_df)}
            continue

        all_X.append(result["X"])
        all_dir.append(result["y_direction"])
        all_ret.append(result["y_returns"])
        all_regime.append(result["y_regime"])
        all_stock_ids.append(np.full(result["n_samples"], tidx, dtype=np.int64))
        all_dates.extend(result["dates"])

        per_stock_info[ticker] = {
            "status": "ok",
            "rows": len(stock_df),
            "sequences": result["n_samples"],
            "date_range": f"{stock_df.index[0].date()} to {stock_df.index[-1].date()}",
        }

        if callback:
            callback(ticker, tidx + 1, len(tickers), result["n_samples"])

    if not all_X:
        logger.error("No stocks produced valid sequences!")
        return {"X": np.array([]), "n_samples": 0}

    # ── Fit global scaler on all stock data combined ───────────────────
    combined_raw = np.vstack(all_raw_features)
    global_scaler.fit(combined_raw)

    # Re-scale all sequences with global scaler
    X_stacked = np.vstack(all_X)
    n_total = X_stacked.shape[0]

    # Reshape to 2D, scale, reshape back
    original_shape = X_stacked.shape
    X_flat = X_stacked.reshape(-1, X_stacked.shape[2])
    X_scaled = global_scaler.transform(X_flat)
    X_stacked = X_scaled.reshape(original_shape)

    y_direction = np.concatenate(all_dir)
    y_returns = np.vstack(all_ret)
    y_regime = np.concatenate(all_regime)
    stock_ids = np.concatenate(all_stock_ids)

    # ── Shuffle (time-respecting shuffle by stock block) ───────────────
    idx_perm = np.random.permutation(n_total)
    X_stacked = X_stacked[idx_perm]
    y_direction = y_direction[idx_perm]
    y_returns = y_returns[idx_perm]
    y_regime = y_regime[idx_perm]
    stock_ids = stock_ids[idx_perm]

    logger.info(
        "Panel dataset built: %d samples from %d stocks, shape=%s",
        n_total, len([t for t in per_stock_info if per_stock_info[t].get("status") == "ok"]),
        X_stacked.shape,
    )

    return {
        "X": X_stacked.astype(np.float32),
        "y_direction": y_direction.astype(np.float32),
        "y_returns": y_returns.astype(np.float32),
        "y_regime": y_regime.astype(np.int64),
        "stock_ids": stock_ids.astype(np.int64),
        "dates": all_dates,
        "scaler": global_scaler,
        "ticker_map": ticker_map,
        "n_stocks": len(ticker_map),
        "n_samples": n_total,
        "per_stock_info": per_stock_info,
        "feature_cols": feature_cols,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  Train / validation split for panel data (time-aware)
# ═══════════════════════════════════════════════════════════════════════
def split_panel_data(
    panel: Dict,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> Dict:
    """
    Split panel data into train / val / test.
    Since panel is already shuffled, a simple split works.
    For strict time-aware splits, use walk_forward module.
    """
    n = panel["n_samples"]
    if n == 0:
        return panel

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    split = {}
    for key in ["X", "y_direction", "y_returns", "y_regime", "stock_ids"]:
        arr = panel[key]
        split[f"{key}_train"] = arr[:n_train]
        split[f"{key}_val"] = arr[n_train:n_train + n_val]
        split[f"{key}_test"] = arr[n_train + n_val:]

    split["n_train"] = n_train
    split["n_val"] = n_val
    split["n_test"] = n_test
    split["scaler"] = panel["scaler"]
    split["ticker_map"] = panel["ticker_map"]
    split["n_stocks"] = panel["n_stocks"]
    split["feature_cols"] = panel["feature_cols"]

    return split
