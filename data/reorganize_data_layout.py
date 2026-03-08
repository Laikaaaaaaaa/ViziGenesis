"""
Reorganize dataset layout:
- data/market_data -> macro-only datasets
- data/stocks/<SYMBOL>/ -> ohlcv + technical indicators

This script migrates existing files without re-downloading.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MARKET_DIR = DATA_DIR / "market_data"
STOCKS_DIR = DATA_DIR / "stocks"

OLD_YF_STOCKS = MARKET_DIR / "yfinance" / "stocks"
OLD_YF_MARKETS = MARKET_DIR / "yfinance" / "markets"
NEW_MACRO_YF = MARKET_DIR / "macro_yahoo"


def _ensure_dirs() -> None:
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    (MARKET_DIR / "fred").mkdir(parents=True, exist_ok=True)
    (MARKET_DIR / "data360").mkdir(parents=True, exist_ok=True)
    NEW_MACRO_YF.mkdir(parents=True, exist_ok=True)


def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    req = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in req):
        return pd.DataFrame(index=df.index)

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    out = pd.DataFrame(index=df.index)

    out["Return_1d"] = close.pct_change() * 100
    out["Return_5d"] = close.pct_change(5) * 100
    out["Return_20d"] = close.pct_change(20) * 100

    out["SMA_20"] = close.rolling(20, min_periods=5).mean()
    out["SMA_50"] = close.rolling(50, min_periods=10).mean()
    out["SMA_200"] = close.rolling(200, min_periods=50).mean()
    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    out["RSI_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    bb_mid = close.rolling(20, min_periods=5).mean()
    bb_std = close.rolling(20, min_periods=5).std(ddof=0)
    out["BB_Middle"] = bb_mid
    out["BB_Upper"] = bb_mid + 2 * bb_std
    out["BB_Lower"] = bb_mid - 2 * bb_std
    out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / bb_mid.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR_14"] = tr.rolling(14, min_periods=5).mean()

    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()

    out["ROC_10"] = close.pct_change(10) * 100
    out["ROC_20"] = close.pct_change(20) * 100
    out["Momentum_10"] = close - close.shift(10)
    out["Momentum_20"] = close - close.shift(20)

    r = close.pct_change()
    out["HistVol_20"] = r.rolling(20, min_periods=5).std(ddof=0) * np.sqrt(252) * 100
    out["HistVol_60"] = r.rolling(60, min_periods=10).std(ddof=0) * np.sqrt(252) * 100

    typical = (high + low + close) / 3
    cum_vol = volume.cumsum().replace(0, np.nan)
    out["VWAP"] = (typical * volume).cumsum() / cum_vol

    ll14 = low.rolling(14, min_periods=5).min()
    hh14 = high.rolling(14, min_periods=5).max()
    out["StochK_14"] = (close - ll14) / (hh14 - ll14).replace(0, np.nan) * 100
    out["StochD_3"] = out["StochK_14"].rolling(3, min_periods=2).mean()

    return out.replace([np.inf, -np.inf], np.nan)


def _load_price_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "Datetime" in df.columns:
        df = df.set_index("Datetime")
    else:
        df = df.set_index(df.columns[0])
    df = _to_dt_index(df)
    return df


def migrate_stock_files() -> Dict[str, int]:
    rows_by_symbol: Dict[str, int] = {}
    if not OLD_YF_STOCKS.exists():
        return rows_by_symbol

    for csv_path in sorted(OLD_YF_STOCKS.glob("*.csv")):
        symbol = csv_path.stem
        stock_dir = STOCKS_DIR / symbol
        stock_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw = _load_price_csv(csv_path)
        except Exception:
            rows_by_symbol[symbol] = 0
            continue

        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        ohlcv = raw.copy()
        for col in req_cols:
            if col not in ohlcv.columns:
                ohlcv[col] = np.nan
        ohlcv = ohlcv[req_cols].dropna(how="all")

        indicators = _compute_indicators(ohlcv)
        features = pd.concat([ohlcv, indicators], axis=1)

        ohlcv.to_csv(stock_dir / "ohlcv.csv")
        indicators.to_csv(stock_dir / "technical_indicators.csv")
        features.to_csv(stock_dir / "features.csv")

        rows_by_symbol[symbol] = int(len(ohlcv))

    return rows_by_symbol


def migrate_macro_yahoo_files() -> Dict[str, int]:
    rows_by_name: Dict[str, int] = {}
    if not OLD_YF_MARKETS.exists():
        return rows_by_name

    for csv_path in sorted(OLD_YF_MARKETS.glob("*.csv")):
        name = csv_path.stem
        dst = NEW_MACRO_YF / f"{name}.csv"
        shutil.copy2(csv_path, dst)
        try:
            df = _load_price_csv(dst)
            rows_by_name[name] = int(len(df))
        except Exception:
            rows_by_name[name] = 0

    return rows_by_name


def count_folder_rows(folder: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not folder.exists():
        return out
    for p in sorted(folder.glob("*.csv")):
        try:
            out[p.stem] = int(len(pd.read_csv(p)))
        except Exception:
            out[p.stem] = 0
    return out


def main() -> None:
    _ensure_dirs()

    stock_rows = migrate_stock_files()
    macro_yahoo_rows = migrate_macro_yahoo_files()
    fred_rows = count_folder_rows(MARKET_DIR / "fred")
    wb_rows = count_folder_rows(MARKET_DIR / "data360")

    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "layout": {
            "market_data": "macro-only (fred, data360, macro_yahoo)",
            "stocks": "data/stocks/<SYMBOL>/{ohlcv.csv, technical_indicators.csv, features.csv}",
        },
        "counts": {
            "stocks_total": len(stock_rows),
            "stocks_ok": sum(1 for v in stock_rows.values() if v > 0),
            "macro_yahoo_total": len(macro_yahoo_rows),
            "macro_yahoo_ok": sum(1 for v in macro_yahoo_rows.values() if v > 0),
            "fred_total": len(fred_rows),
            "fred_ok": sum(1 for v in fred_rows.values() if v > 0),
            "data360_total": len(wb_rows),
            "data360_ok": sum(1 for v in wb_rows.values() if v > 0),
        },
        "rows": {
            "stocks": stock_rows,
            "macro_yahoo": macro_yahoo_rows,
            "fred": fred_rows,
            "data360": wb_rows,
        },
    }

    with open(DATA_DIR / "manifest_reorganized.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["counts"], indent=2))
    print("Saved reorganized manifest:", DATA_DIR / "manifest_reorganized.json")


if __name__ == "__main__":
    main()
