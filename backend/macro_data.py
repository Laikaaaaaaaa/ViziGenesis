"""
Macro-economic data fetching for enhanced AI training.

Primary source for core macro features is World Bank Data360 API.
If Data360 is unavailable for a series, this module falls back to Yahoo
Finance tickers for that specific feature.
"""
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger("vizigenesis.macro")

# ── Core macro series (Data360 first, Yahoo fallback) ────────────────
DATA360_API_ROOT = "https://data360api.worldbank.org"

MACRO_WB_SERIES: Dict[str, Dict[str, str]] = {
    # Data360 currently has limited direct US market index levels.
    # We use the closest available indicators as primary source.
    "SP500": {
        "DATABASE_ID": "WB_FSI",
        "INDICATOR": "WB_FSI_CM_MKT_INDX_ZG",
        "REF_AREA": "USA",
        "FREQ": "A",
        "VALUE_TYPE": "pct",
    },
    "NASDAQ": {
        "DATABASE_ID": "IMF_IFS",
        "INDICATOR": "IMF_IFS_FPEPNAS",
        "REF_AREA": "USA",
        "VALUE_TYPE": "level",
    },
    "VIX": {
        "DATABASE_ID": "WB_WDI",
        "INDICATOR": "WB_WDI_FR_INR_RISK",
        "REF_AREA": "USA",
        "FREQ": "A",
        "VALUE_TYPE": "pct",
    },
    "BOND_10Y": {
        "DATABASE_ID": "IMF_IFS",
        "INDICATOR": "IMF_IFS_FIGBY_MT",
        "REF_AREA": "USA",
        "VALUE_TYPE": "level",
    },
}

MACRO_YAHOO_FALLBACK: Dict[str, str] = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "BOND_10Y": "^TNX",
}
_DXY_TICKER = "DX-Y.NYB"      # US Dollar Index — used for inflation proxy

# Quant-mode additional tickers
QUANT_EXTRA_TICKERS: Dict[str, str] = {
    "SOXX":  "SOXX",           # iShares Semiconductor ETF
    "SMH":   "SMH",            # VanEck Semiconductor ETF
    "GOLD":  "GC=F",           # Gold Futures
    "OIL":   "CL=F",           # WTI Crude Oil Futures
}

MACRO_FEATURE_NAMES: List[str] = [
    "SP500", "NASDAQ", "VIX", "BOND_10Y", "INFLATION_PROXY",
]

QUANT_EXTRA_FEATURE_NAMES: List[str] = [
    "SOXX_Return", "SMH_Return", "Gold_Return", "Oil_Return", "DXY_Level",
]

# ── In-memory cache ──────────────────────────────────────────────────
_macro_cache: Dict[str, pd.DataFrame] = {}
_macro_cache_ts: Dict[str, float] = {}
_MACRO_CACHE_TTL = 7200  # 2 hours


# ── Helpers ───────────────────────────────────────────────────────────
def _fetch_single_close(ticker: str, period: str = "10y") -> Optional[pd.Series]:
    """Fetch daily Close prices for one ticker from yfinance."""
    import yfinance as yf
    for attempt in range(2):
        try:
            df = yf.download(
                ticker, period=period, interval="1d",
                progress=False, auto_adjust=False, actions=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if "Close" in df.columns:
                    series = df["Close"].dropna()
                    if len(series) > 0:
                        return series
        except Exception as e:
            logger.debug("Macro fetch attempt %d for %s: %s", attempt + 1, ticker, e)
            time.sleep(0.5)
    return None


def _period_to_start(period: str) -> str:
    """Convert Yahoo-like period string to an ISO start date."""
    raw = (period or "10y").strip().lower()
    mapping_days = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
    }
    if raw == "max":
        return "1900-01-01"
    days = mapping_days.get(raw, 3650)
    start = datetime.utcnow().date() - timedelta(days=days)
    return start.isoformat()


def _parse_time_period(raw: str) -> Optional[pd.Timestamp]:
    """Parse Data360 TIME_PERIOD values like YYYY, YYYY-MM, YYYY-Qn."""
    if not raw:
        return None
    text = str(raw).strip()
    try:
        if len(text) == 4 and text.isdigit():
            return pd.Timestamp(f"{text}-01-01")
        if "-Q" in text:
            year, quarter = text.split("-Q", 1)
            q = int(quarter)
            month = min(max((q - 1) * 3 + 1, 1), 12)
            return pd.Timestamp(f"{year}-{month:02d}-01")
        return pd.to_datetime(text, errors="coerce")
    except Exception:
        return None


def _fetch_worldbank_series(series_cfg: Dict[str, str], period: str = "10y") -> Optional[pd.Series]:
    """Fetch a single Data360 indicator series as date-indexed float values."""
    start = _period_to_start(period)
    params = {
        "DATABASE_ID": series_cfg.get("DATABASE_ID", ""),
        "INDICATOR": series_cfg.get("INDICATOR", ""),
        "REF_AREA": series_cfg.get("REF_AREA", ""),
        "timePeriodFrom": start,
        "format": "json",
        "top": 1000,
        "skip": 0,
    }
    if series_cfg.get("FREQ"):
        params["FREQ"] = series_cfg["FREQ"]

    obs_dates: List[pd.Timestamp] = []
    obs_values: List[float] = []
    expected_count: Optional[int] = None

    for attempt in range(2):
        try:
            while True:
                qs = urlencode(params)
                req = Request(f"{DATA360_API_ROOT}/data360/data?{qs}", headers={"User-Agent": "ViziGenesis/2.0"})
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                if not isinstance(data, dict):
                    break

                if expected_count is None:
                    try:
                        expected_count = int(data.get("count", 0))
                    except Exception:
                        expected_count = 0

                values = data.get("value", [])
                if not isinstance(values, list) or not values:
                    break

                for row in values:
                    if not isinstance(row, dict):
                        continue
                    tp = _parse_time_period(str(row.get("TIME_PERIOD", "")))
                    try:
                        ov = float(row.get("OBS_VALUE"))
                    except Exception:
                        continue
                    if tp is None:
                        continue
                    obs_dates.append(tp)
                    obs_values.append(ov)

                params["skip"] = int(params.get("skip", 0)) + int(params.get("top", 1000))
                if len(values) < int(params.get("top", 1000)):
                    break
                if expected_count is not None and int(params["skip"]) >= expected_count:
                    break

            if obs_dates and obs_values:
                s = pd.Series(obs_values, index=pd.DatetimeIndex(obs_dates), name=series_cfg.get("INDICATOR", ""))
                s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
                if len(s) > 0:
                    return s
            return None
        except Exception as e:
            logger.debug(
                "Data360 fetch attempt %d for %s/%s failed: %s",
                attempt + 1,
                series_cfg.get("DATABASE_ID", ""),
                series_cfg.get("INDICATOR", ""),
                e,
            )
            time.sleep(0.5)
    return None


# ── Public API ────────────────────────────────────────────────────────
def fetch_macro_data(period: str = "10y") -> pd.DataFrame:
    """
    Fetch all macro indicators and return a DataFrame aligned by date.

    Columns returned: SP500, NASDAQ, VIX, BOND_10Y, INFLATION_PROXY

    - SP500 and NASDAQ → 1-day percentage returns (%) — scale-invariant.
    - VIX → raw level (0–100).
    - BOND_10Y → 10-Year yield (%).
    - INFLATION_PROXY → inverted 20-day DXY momentum (weak $ → +inflation).

    All values are forward-filled / backward-filled so that every trading day
    has a value, even across different holiday calendars.
    """
    cache_key = f"macro_{period}"
    if cache_key in _macro_cache:
        if time.time() - _macro_cache_ts.get(cache_key, 0) < _MACRO_CACHE_TTL:
            return _macro_cache[cache_key].copy()

    # Fetch each macro series from Data360 first, then fallback to Yahoo.
    frames: Dict[str, pd.Series] = {}
    wb_value_type: Dict[str, str] = {}
    for name, series_cfg in MACRO_WB_SERIES.items():
        series = _fetch_worldbank_series(series_cfg, period)
        if series is not None and len(series) > 0:
            frames[name] = series
            wb_value_type[name] = series_cfg.get("VALUE_TYPE", "level")
            logger.info(
                "Macro %s (Data360:%s/%s): %d rows",
                name,
                series_cfg.get("DATABASE_ID", ""),
                series_cfg.get("INDICATOR", ""),
                len(series),
            )
            continue

        yahoo_ticker = MACRO_YAHOO_FALLBACK.get(name, "")
        if yahoo_ticker:
            series = _fetch_single_close(yahoo_ticker, period)
            if series is not None and len(series) > 0:
                frames[name] = series
                logger.warning(
                    "Macro %s Data360 unavailable (%s/%s) — fallback Yahoo (%s): %d rows",
                    name,
                    series_cfg.get("DATABASE_ID", ""),
                    series_cfg.get("INDICATOR", ""),
                    yahoo_ticker,
                    len(series),
                )
                continue

        logger.warning(
            "Macro %s (Data360:%s/%s): unavailable",
            name,
            series_cfg.get("DATABASE_ID", ""),
            series_cfg.get("INDICATOR", ""),
        )

    # DXY (Dollar Index) for inflation proxy
    dxy_series = _fetch_single_close(_DXY_TICKER, period)
    if dxy_series is not None and len(dxy_series) > 0:
        frames["DXY"] = dxy_series
        logger.info("Macro DXY (%s): %d rows", _DXY_TICKER, len(dxy_series))

    if not frames:
        logger.error("No macro data fetched — all API attempts failed")
        return pd.DataFrame(columns=MACRO_FEATURE_NAMES)

    # Combine into a single DataFrame
    raw_df = pd.DataFrame(frames)
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()

    # ── Transform to training-friendly features ──────────────────────
    macro = pd.DataFrame(index=raw_df.index)

    # SP500 & NASDAQ → daily % return (bounded to +-15 %).
    # If Data360 already returns percent values, use them directly.
    for col in ("SP500", "NASDAQ"):
        if col in raw_df.columns:
            if wb_value_type.get(col) == "pct":
                ret = pd.to_numeric(raw_df[col], errors="coerce").ffill().fillna(0)
            else:
                ret = raw_df[col].pct_change(fill_method=None).fillna(0) * 100
            macro[col] = ret.clip(-15, 15)

    # VIX proxy from Data360 (or Yahoo fallback).
    if "VIX" in raw_df.columns:
        macro["VIX"] = raw_df["VIX"].ffill()

    # BOND_10Y → yield in %
    if "BOND_10Y" in raw_df.columns:
        macro["BOND_10Y"] = raw_df["BOND_10Y"].ffill()

    # INFLATION_PROXY from DXY momentum (inverted 20-day return)
    if "DXY" in raw_df.columns and raw_df["DXY"].notna().sum() > 20:
        dxy_ret = raw_df["DXY"].pct_change(periods=20, fill_method=None).fillna(0) * 100
        macro["INFLATION_PROXY"] = (-dxy_ret).clip(-10, 10)
    elif "BOND_10Y" in raw_df.columns:
        # Fallback: use yield momentum as rough inflation signal
        macro["INFLATION_PROXY"] = raw_df["BOND_10Y"].diff(periods=20).fillna(0).clip(-5, 5)
    else:
        macro["INFLATION_PROXY"] = 0.0

    # Ensure all expected columns exist
    for col in MACRO_FEATURE_NAMES:
        if col not in macro.columns:
            macro[col] = 0.0

    macro = macro[MACRO_FEATURE_NAMES].copy()
    macro = macro.replace([np.inf, -np.inf], np.nan)
    macro = macro.ffill().bfill().fillna(0)
    macro = macro.sort_index()

    logger.info(
        "Macro data ready: %d rows, range=%s → %s",
        len(macro),
        macro.index.min().strftime("%Y-%m-%d") if len(macro) else "N/A",
        macro.index.max().strftime("%Y-%m-%d") if len(macro) else "N/A",
    )

    _macro_cache[cache_key] = macro
    _macro_cache_ts[cache_key] = time.time()
    return macro.copy()


def merge_macro_with_stock(
    stock_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join macro data onto a stock DataFrame by date.
    Forward-fills gaps for non-overlapping trading days (e.g. VN holidays vs US holidays).
    If macro_df is empty or None, returns stock_df with zero-filled macro columns.
    """
    if macro_df is None or macro_df.empty:
        out = stock_df.copy()
        for col in MACRO_FEATURE_NAMES:
            if col not in out.columns:
                out[col] = 0.0
        return out

    stock = stock_df.copy()
    macro = macro_df[MACRO_FEATURE_NAMES].copy()

    # Normalize to date-only for clean merge
    stock.index = pd.to_datetime(stock.index).normalize()
    macro.index = pd.to_datetime(macro.index).normalize()
    macro = macro[~macro.index.duplicated(keep="last")]

    merged = stock.join(macro, how="left")

    for col in MACRO_FEATURE_NAMES:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill().fillna(0)
        else:
            merged[col] = 0.0

    return merged


# ═══════════════════════════════════════════════════════════════════════
# Quant-mode: extended macro / sector / commodity data
# ═══════════════════════════════════════════════════════════════════════
def fetch_quant_extra_data(period: str = "10y") -> pd.DataFrame:
    """
    Fetch SOXX, SMH, Gold, Oil, and DXY for the quant feature set.

    Returns DataFrame with columns:
        SOXX_Return, SMH_Return, Gold_Return, Oil_Return, DXY_Level
    """
    cache_key = f"quant_extra_{period}"
    if cache_key in _macro_cache:
        if time.time() - _macro_cache_ts.get(cache_key, 0) < _MACRO_CACHE_TTL:
            return _macro_cache[cache_key].copy()

    frames: Dict[str, pd.Series] = {}
    for name, ticker in QUANT_EXTRA_TICKERS.items():
        series = _fetch_single_close(ticker, period)
        if series is not None and len(series) > 0:
            frames[name] = series
            logger.info("Quant extra %s (%s): %d rows", name, ticker, len(series))
        else:
            logger.warning("Quant extra %s (%s): unavailable", name, ticker)

    # Also fetch DXY raw level for quant mode
    dxy_series = _fetch_single_close(_DXY_TICKER, period)
    if dxy_series is not None and len(dxy_series) > 0:
        frames["DXY_RAW"] = dxy_series

    if not frames:
        logger.error("No quant extra data fetched")
        return pd.DataFrame(columns=QUANT_EXTRA_FEATURE_NAMES)

    raw_df = pd.DataFrame(frames)
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()

    extra = pd.DataFrame(index=raw_df.index)

    # SOXX & SMH → daily % return (bounded ±15 %)
    for name, col_out in [("SOXX", "SOXX_Return"), ("SMH", "SMH_Return")]:
        if name in raw_df.columns:
            ret = raw_df[name].pct_change(fill_method=None).fillna(0) * 100
            extra[col_out] = ret.clip(-15, 15)
        else:
            extra[col_out] = 0.0

    # Gold & Oil → daily % return (bounded ±15 %)
    for name, col_out in [("GOLD", "Gold_Return"), ("OIL", "Oil_Return")]:
        if name in raw_df.columns:
            ret = raw_df[name].pct_change(fill_method=None).fillna(0) * 100
            extra[col_out] = ret.clip(-15, 15)
        else:
            extra[col_out] = 0.0

    # DXY raw level
    if "DXY_RAW" in raw_df.columns:
        extra["DXY_Level"] = raw_df["DXY_RAW"].ffill()
    else:
        extra["DXY_Level"] = 0.0

    for col in QUANT_EXTRA_FEATURE_NAMES:
        if col not in extra.columns:
            extra[col] = 0.0

    extra = extra[QUANT_EXTRA_FEATURE_NAMES].copy()
    extra = extra.replace([np.inf, -np.inf], np.nan)
    extra = extra.ffill().bfill().fillna(0)

    logger.info("Quant extra data ready: %d rows", len(extra))

    _macro_cache[cache_key] = extra
    _macro_cache_ts[cache_key] = time.time()
    return extra.copy()


def merge_quant_extra_with_stock(
    stock_df: pd.DataFrame,
    quant_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join quant extra features onto a stock DataFrame by date."""
    if quant_df is None or quant_df.empty:
        out = stock_df.copy()
        for col in QUANT_EXTRA_FEATURE_NAMES:
            if col not in out.columns:
                out[col] = 0.0
        return out

    stock = stock_df.copy()
    extra = quant_df[QUANT_EXTRA_FEATURE_NAMES].copy()

    stock.index = pd.to_datetime(stock.index).normalize()
    extra.index = pd.to_datetime(extra.index).normalize()
    extra = extra[~extra.index.duplicated(keep="last")]

    merged = stock.join(extra, how="left")
    for col in QUANT_EXTRA_FEATURE_NAMES:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill().fillna(0)
        else:
            merged[col] = 0.0
    return merged


def fetch_nasdaq_close(period: str = "10y") -> Optional[pd.Series]:
    """Fetch raw NASDAQ close prices for computing excess returns."""
    return _fetch_single_close("^IXIC", period)
