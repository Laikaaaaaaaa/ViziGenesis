"""
ViziGenesis V2 — Comprehensive Market Data Engine
====================================================
Multi-source data ingestion from 60+ free, open market instruments.
Covers macro-economics, market structure, cross-asset flows, yield curves,
currencies, commodities, crypto risk appetite, sector rotation, volatility
term structure, and international markets.

Data Sources (all free, no API key required):
  • Yahoo Finance — 60+ tickers across all asset classes
  • World Bank Data360 — macro-economic fundamentals (GDP, CPI, unemployment)
  • FOMC calendar — Federal Reserve policy decisions

Public API (backward-compatible):
  fetch_macro_data(start)     → core macro DataFrame (9 cols, legacy)
  fetch_extended_macro(start) → extended 40+ column macro DataFrame
  build_fomc_features(idx)    → FOMC policy event flags
  fetch_market_context(start) → market indices & benchmarks
  fetch_sector_commodity(start) → sector & commodity returns
  fetch_cross_asset_data(start) → bonds, currencies, crypto, international
  fetch_nasdaq_close(start)   → NASDAQ close for excess return
  fetch_all_market_data(start)→ unified master DataFrame (all sources)
"""
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

logger = logging.getLogger("vizigenesis.v2.market_data")

# ───────────────────────────────────────────────────────────────────────
# In-memory cache
# ───────────────────────────────────────────────────────────────────────
_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
_CACHE_TTL = 4 * 3600  # 4 hours


def _get_cached(key: str) -> Optional[pd.DataFrame]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0].copy()
    return None


def _set_cache(key: str, df: pd.DataFrame):
    _cache[key] = (df.copy(), time.time())


# ═══════════════════════════════════════════════════════════════════════
# 1. Yahoo Finance Bulk Fetcher
# ═══════════════════════════════════════════════════════════════════════

# ── Master Ticker Registry ────────────────────────────────────────────
# Organized by category for maximum market coverage

YAHOO_TICKERS = {
    # ── Market Indices ────────────────────────────────────────────────
    "SP500":          "^GSPC",
    "NASDAQ":         "^IXIC",
    "DOW":            "^DJI",
    "RUSSELL2000":    "^RUT",       # small-cap proxy
    "SP400_MID":      "^MID",       # mid-cap
    "NASDAQ100":      "^NDX",
    "NYSE_COMP":      "^NYA",       # NYSE composite

    # ── Volatility ────────────────────────────────────────────────────
    "VIX":            "^VIX",
    "VXN":            "^VXN",       # NASDAQ volatility

    # ── Treasury Yields ───────────────────────────────────────────────
    "YIELD_13W":      "^IRX",       # 13-week T-bill
    "YIELD_5Y":       "^FVX",       # 5-year treasury
    "YIELD_10Y":      "^TNX",       # 10-year treasury
    "YIELD_30Y":      "^TYX",       # 30-year treasury

    # ── Bond ETFs ─────────────────────────────────────────────────────
    "TLT":            "TLT",        # 20+ year treasury
    "IEF":            "IEF",        # 7-10 year treasury
    "SHY":            "SHY",        # 1-3 year treasury
    "HYG":            "HYG",        # high yield corporate
    "LQD":            "LQD",        # investment grade corporate
    "TIP":            "TIP",        # TIPS (inflation-protected)
    "AGG":            "AGG",        # aggregate bond

    # ── Sector ETFs (all 11 S&P sectors) ──────────────────────────────
    "XLK":            "XLK",        # Technology
    "XLF":            "XLF",        # Financials
    "XLE":            "XLE",        # Energy
    "XLV":            "XLV",        # Healthcare
    "XLI":            "XLI",        # Industrials
    "XLP":            "XLP",        # Consumer Staples
    "XLY":            "XLY",        # Consumer Discretionary
    "XLB":            "XLB",        # Materials
    "XLU":            "XLU",        # Utilities
    "XLRE":           "XLRE",       # Real Estate
    "XLC":            "XLC",        # Communication Services

    # ── Semiconductor ─────────────────────────────────────────────────
    "SOXX":           "SOXX",
    "SMH":            "SMH",

    # ── Commodities ───────────────────────────────────────────────────
    "GOLD":           "GC=F",
    "SILVER":         "SI=F",
    "OIL_WTI":        "CL=F",
    "OIL_BRENT":      "BZ=F",
    "NATGAS":         "NG=F",
    "COPPER":         "HG=F",       # economic bellwether

    # ── Currencies (vs USD) ───────────────────────────────────────────
    "DXY":            "DX-Y.NYB",   # US Dollar Index
    "EURUSD":         "EURUSD=X",
    "GBPUSD":         "GBPUSD=X",
    "USDJPY":         "USDJPY=X",
    "USDCNY":         "USDCNY=X",

    # ── Crypto (risk appetite proxy) ──────────────────────────────────
    "BTC":            "BTC-USD",
    "ETH":            "ETH-USD",

    # ── International Markets ─────────────────────────────────────────
    "EEM":            "EEM",        # Emerging Markets ETF
    "EFA":            "EFA",        # EAFE (developed ex-US)
    "FXI":            "FXI",        # China large-cap
    "EWJ":            "EWJ",        # Japan
    "EWZ":            "EWZ",        # Brazil
    "INDA":           "INDA",       # India
    "EWG":            "EWG",        # Germany/Europe
}


def _yf_download_safe(ticker: str, start: str = "2000-01-01",
                      period: Optional[str] = None, retries: int = 3) -> pd.DataFrame:
    """Download Yahoo Finance data with retries.

    Uses ``yf.Ticker().history()`` which is more reliable on cloud
    servers (RunPod, Colab, etc.) than ``yf.download()``.
    """
    import yfinance as yf
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            if period:
                df = t.history(period=period, auto_adjust=True)
            else:
                df = t.history(start=start, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                # Normalize tz-aware index to tz-naive (yfinance ≥1.0)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception as e:
            if attempt < retries:
                time.sleep(1.0 + attempt * 0.5)
            else:
                logger.debug("Yahoo download %s failed: %s", ticker, e)
    return pd.DataFrame()


def fetch_yahoo_bulk(
    tickers: Optional[Dict[str, str]] = None,
    start: str = "2000-01-01",
    column: str = "Close",
) -> pd.DataFrame:
    """
    Fetch a single column (default: Close) for many Yahoo tickers.
    Uses sequential ``Ticker.history()`` calls which are more reliable
    on cloud servers than multi-ticker ``yf.download()``.
    """
    if tickers is None:
        tickers = YAHOO_TICKERS

    cached = _get_cached(f"yahoo_bulk_{column}_{start}")
    if cached is not None:
        return cached

    frames = {}
    total = len(tickers)
    for idx, (name, tkr) in enumerate(tickers.items()):
        df = _yf_download_safe(tkr, start=start)
        if not df.empty and column in df.columns:
            frames[name] = df[column]
        # Brief pause every 8 tickers to avoid throttling
        if (idx + 1) % 8 == 0 and idx + 1 < total:
            time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.DatetimeIndex(result.index)
    result = result.sort_index()
    _set_cache(f"yahoo_bulk_{column}_{start}", result)
    logger.info("Yahoo bulk fetched: %d/%d tickers OK", len(frames), total)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. World Bank Data360 API (macro fundamentals)
# ═══════════════════════════════════════════════════════════════════════

DATA360_API_BASE = "https://data360api.worldbank.org/data360/data"

DATA360_INDICATORS = {
    "GDP_GROWTH":     {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "USA", "FREQ": "A"},
    "GDP_PER_CAPITA": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_PCAP_CD", "REF_AREA": "USA", "FREQ": "A"},
    "CPI":            {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL", "REF_AREA": "USA", "FREQ": "A"},
    "INFLATION":      {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL_ZG", "REF_AREA": "USA", "FREQ": "A"},
    "UNEMPLOYMENT":   {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_SL_UEM_TOTL_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "LENDING_RATE":   {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FR_INR_LEND", "REF_AREA": "USA", "FREQ": "A"},
    "M2_GDP":         {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FM_LBL_BMNY_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "TRADE_PCT_GDP":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NE_TRD_GNFS_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "FDI_PCT_GDP":    {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_BX_KLT_DINV_WD_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "GOV_DEBT_GDP":   {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_GC_DOD_TOTL_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "CURRENT_ACCT":   {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_BN_CAB_XOKA_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "STOCKS_TRADED":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_CM_MKT_TRAD_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "MARKET_CAP_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_CM_MKT_LCAP_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
}


def _parse_time_period(raw: str) -> Optional[pd.Timestamp]:
    """Parse Data360 TIME_PERIOD values like YYYY, YYYY-MM, YYYY-Qn."""
    if not raw:
        return None
    text = str(raw).strip()
    try:
        if len(text) == 4 and text.isdigit():
            return pd.Timestamp(f"{text}-07-01")  # mid-year for annual
        if "-Q" in text:
            year, quarter = text.split("-Q", 1)
            month = min(max((int(quarter) - 1) * 3 + 2, 1), 12)
            return pd.Timestamp(f"{year}-{month:02d}-15")
        return pd.to_datetime(text, errors="coerce")
    except Exception:
        return None


def _fetch_data360(indicator_name: str, start: str = "2000-01-01") -> pd.Series:
    """Fetch a single Data360 indicator with pagination."""
    cached = _get_cached(f"d360_{indicator_name}")
    if cached is not None and not cached.empty:
        s = cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
        return s

    cfg = DATA360_INDICATORS.get(indicator_name)
    if not cfg:
        return pd.Series(dtype=float, name=indicator_name)

    params = {
        "DATABASE_ID": cfg["DATABASE_ID"],
        "INDICATOR": cfg["INDICATOR"],
        "REF_AREA": cfg.get("REF_AREA", "USA"),
        "FREQ": cfg.get("FREQ", "A"),
        "timePeriodFrom": start[:4],
        "format": "json",
        "top": 1000,
        "skip": 0,
    }

    obs_dates, obs_values = [], []
    try:
        while True:
            qs = urlencode(params)
            req = Request(f"{DATA360_API_BASE}?{qs}", headers={"User-Agent": "ViziGenesis/2.0"})
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            rows = data.get("value", []) if isinstance(data, dict) else []
            if not rows:
                break

            for row in rows:
                if not isinstance(row, dict):
                    continue
                ts = _parse_time_period(str(row.get("TIME_PERIOD", "")))
                try:
                    val = float(row.get("OBS_VALUE"))
                except Exception:
                    continue
                if ts is not None:
                    obs_dates.append(ts)
                    obs_values.append(val)

            if len(rows) < params["top"]:
                break
            params["skip"] += params["top"]
    except Exception as e:
        logger.warning("Data360 fetch %s failed: %s", indicator_name, e)

    if not obs_dates:
        return pd.Series(dtype=float, name=indicator_name)

    s = pd.Series(obs_values, index=pd.DatetimeIndex(obs_dates), name=indicator_name)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    _set_cache(f"d360_{indicator_name}", s.to_frame())
    return s


def fetch_worldbank_macro(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch all World Bank macro indicators, resampled to daily."""
    frames = {}
    for name in DATA360_INDICATORS:
        s = _fetch_data360(name, start)
        if len(s) > 0:
            frames[name] = s

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames)
    # Resample annual → daily with forward-fill
    daily_idx = pd.bdate_range(start=combined.index.min(), end=pd.Timestamp.today())
    combined = combined.reindex(daily_idx).ffill().bfill()
    return combined


# ═══════════════════════════════════════════════════════════════════════
# 3. Derived Features from Raw Market Data
# ═══════════════════════════════════════════════════════════════════════

def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0) * 100


def compute_yield_curve_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute yield curve slope, curvature, and level features."""
    result = pd.DataFrame(index=bulk.index)

    y13w = bulk.get("YIELD_13W", pd.Series(dtype=float))
    y5y = bulk.get("YIELD_5Y", pd.Series(dtype=float))
    y10y = bulk.get("YIELD_10Y", pd.Series(dtype=float))
    y30y = bulk.get("YIELD_30Y", pd.Series(dtype=float))

    # Spreads
    if not y10y.empty and not y13w.empty:
        result["Yield_10Y_3M_Spread"] = (y10y - y13w).fillna(0)
    else:
        result["Yield_10Y_3M_Spread"] = 0.0

    if not y10y.empty and not y5y.empty:
        result["Yield_10Y_5Y_Spread"] = (y10y - y5y).fillna(0)
    else:
        result["Yield_10Y_5Y_Spread"] = 0.0

    if not y30y.empty and not y10y.empty:
        result["Yield_30Y_10Y_Spread"] = (y30y - y10y).fillna(0)
    else:
        result["Yield_30Y_10Y_Spread"] = 0.0

    # Yield curve curvature: 2*Y5 - Y13W - Y10Y
    if not y5y.empty and not y13w.empty and not y10y.empty:
        result["Yield_Curve_Curvature"] = (2 * y5y - y13w - y10y).fillna(0)
    else:
        result["Yield_Curve_Curvature"] = 0.0

    # Level (average yield)
    yields = [s for s in [y13w, y5y, y10y, y30y] if not s.empty]
    if yields:
        result["Yield_Level"] = pd.concat(yields, axis=1).mean(axis=1).fillna(0)
    else:
        result["Yield_Level"] = 0.0

    # Changes
    if not y10y.empty:
        result["Yield_10Y_Change_5d"] = y10y.diff(5).fillna(0)
        result["Yield_10Y_Change_20d"] = y10y.diff(20).fillna(0)
    else:
        result["Yield_10Y_Change_5d"] = 0.0
        result["Yield_10Y_Change_20d"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_credit_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute credit spread and risk features from bond ETFs."""
    result = pd.DataFrame(index=bulk.index)

    hyg = bulk.get("HYG", pd.Series(dtype=float))
    lqd = bulk.get("LQD", pd.Series(dtype=float))
    tlt = bulk.get("TLT", pd.Series(dtype=float))
    agg_bond = bulk.get("AGG", pd.Series(dtype=float))

    # HY vs IG spread proxy (relative performance)
    if not hyg.empty and not lqd.empty:
        result["HY_IG_Spread"] = _safe_pct_change(hyg, 5) - _safe_pct_change(lqd, 5)
    else:
        result["HY_IG_Spread"] = 0.0

    # Credit risk appetite (HYG/TLT ratio)
    if not hyg.empty and not tlt.empty:
        ratio = hyg / tlt.clip(lower=1)
        result["Credit_Risk_Appetite"] = _safe_pct_change(ratio, 20)
    else:
        result["Credit_Risk_Appetite"] = 0.0

    # Bond momentum
    if not tlt.empty:
        result["TLT_Ret_5d"] = _safe_pct_change(tlt, 5)
        result["TLT_Ret_20d"] = _safe_pct_change(tlt, 20)
    else:
        result["TLT_Ret_5d"] = 0.0
        result["TLT_Ret_20d"] = 0.0

    # TIPS performance (inflation expectation proxy)
    tip = bulk.get("TIP", pd.Series(dtype=float))
    if not tip.empty and not tlt.empty:
        result["TIPS_vs_Treasury"] = _safe_pct_change(tip, 20) - _safe_pct_change(tlt, 20)
    else:
        result["TIPS_vs_Treasury"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_sector_rotation_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute sector rotation and relative strength features."""
    result = pd.DataFrame(index=bulk.index)
    sector_names = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
    semi_names = ["SOXX", "SMH"]

    # Individual sector returns (5d and 20d)
    for name in sector_names:
        s = bulk.get(name, pd.Series(dtype=float))
        if not s.empty:
            result[f"{name}_Ret_5d"] = _safe_pct_change(s, 5)
            result[f"{name}_Ret_20d"] = _safe_pct_change(s, 20)
        else:
            result[f"{name}_Ret_5d"] = 0.0
            result[f"{name}_Ret_20d"] = 0.0

    # Semiconductor returns
    for name in semi_names:
        s = bulk.get(name, pd.Series(dtype=float))
        if not s.empty:
            result[f"{name}_Ret"] = _safe_pct_change(s, 1)
        else:
            result[f"{name}_Ret"] = 0.0

    # Risk-on vs Risk-off rotation
    xlk = bulk.get("XLK", pd.Series(dtype=float))
    xly = bulk.get("XLY", pd.Series(dtype=float))
    xlp = bulk.get("XLP", pd.Series(dtype=float))
    xlu = bulk.get("XLU", pd.Series(dtype=float))

    if not xlk.empty and not xlu.empty:
        result["Risk_On_Off"] = _safe_pct_change(xlk, 20) - _safe_pct_change(xlu, 20)
    else:
        result["Risk_On_Off"] = 0.0

    if not xly.empty and not xlp.empty:
        result["Consumer_Cyclical_vs_Defensive"] = _safe_pct_change(xly, 20) - _safe_pct_change(xlp, 20)
    else:
        result["Consumer_Cyclical_vs_Defensive"] = 0.0

    # Sector dispersion (std of sector returns → high = stock-picking market)
    returns_20d = []
    for name in sector_names:
        s = bulk.get(name, pd.Series(dtype=float))
        if not s.empty:
            returns_20d.append(_safe_pct_change(s, 20))
    if len(returns_20d) >= 5:
        sector_ret_df = pd.concat(returns_20d, axis=1)
        result["Sector_Dispersion"] = sector_ret_df.std(axis=1).fillna(0)
    else:
        result["Sector_Dispersion"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_commodity_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute commodity-derived economic signals."""
    result = pd.DataFrame(index=bulk.index)

    for name, label in [("GOLD", "Gold"), ("SILVER", "Silver"), ("OIL_WTI", "Oil"),
                         ("OIL_BRENT", "Brent"), ("NATGAS", "NatGas"), ("COPPER", "Copper")]:
        s = bulk.get(name, pd.Series(dtype=float))
        if not s.empty:
            result[f"{label}_Ret"] = _safe_pct_change(s, 1)
            result[f"{label}_Ret_20d"] = _safe_pct_change(s, 20)
        else:
            result[f"{label}_Ret"] = 0.0
            result[f"{label}_Ret_20d"] = 0.0

    # Gold/Oil ratio (inflation/growth proxy)
    gold = bulk.get("GOLD", pd.Series(dtype=float))
    oil = bulk.get("OIL_WTI", pd.Series(dtype=float))
    if not gold.empty and not oil.empty:
        ratio = gold / oil.clip(lower=1)
        result["Gold_Oil_Ratio"] = _safe_pct_change(ratio, 20)
    else:
        result["Gold_Oil_Ratio"] = 0.0

    # Copper/Gold ratio (economic growth expectation)
    copper = bulk.get("COPPER", pd.Series(dtype=float))
    if not copper.empty and not gold.empty:
        ratio = copper / gold.clip(lower=1)
        result["Copper_Gold_Ratio"] = _safe_pct_change(ratio, 20)
    else:
        result["Copper_Gold_Ratio"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_currency_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute currency-derived risk and flow signals."""
    result = pd.DataFrame(index=bulk.index)

    for name, label in [("DXY", "DXY"), ("EURUSD", "EURUSD"), ("GBPUSD", "GBPUSD"),
                         ("USDJPY", "USDJPY"), ("USDCNY", "USDCNY")]:
        s = bulk.get(name, pd.Series(dtype=float))
        if not s.empty:
            result[f"{label}_Change"] = _safe_pct_change(s, 1)
            result[f"{label}_Change_20d"] = _safe_pct_change(s, 20)
        else:
            result[f"{label}_Change"] = 0.0
            result[f"{label}_Change_20d"] = 0.0

    # Dollar strength momentum
    dxy = bulk.get("DXY", pd.Series(dtype=float))
    if not dxy.empty:
        result["DXY_Momentum"] = dxy.rolling(20, min_periods=5).mean() - dxy.rolling(60, min_periods=10).mean()
        result["DXY_Momentum"] = result["DXY_Momentum"].fillna(0)
    else:
        result["DXY_Momentum"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_crypto_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute crypto-derived risk appetite signals."""
    result = pd.DataFrame(index=bulk.index)

    btc = bulk.get("BTC", pd.Series(dtype=float))
    eth = bulk.get("ETH", pd.Series(dtype=float))

    if not btc.empty:
        result["BTC_Ret"] = _safe_pct_change(btc, 1)
        result["BTC_Ret_20d"] = _safe_pct_change(btc, 20)
        result["BTC_Vol_20d"] = btc.pct_change().rolling(20, min_periods=5).std().fillna(0) * np.sqrt(252) * 100
    else:
        result["BTC_Ret"] = 0.0
        result["BTC_Ret_20d"] = 0.0
        result["BTC_Vol_20d"] = 0.0

    if not eth.empty:
        result["ETH_Ret"] = _safe_pct_change(eth, 1)
    else:
        result["ETH_Ret"] = 0.0

    # Crypto risk appetite (BTC momentum as proxy)
    if not btc.empty:
        btc_ma20 = btc.rolling(20, min_periods=5).mean()
        btc_ma60 = btc.rolling(60, min_periods=10).mean()
        result["Crypto_Risk_Appetite"] = ((btc_ma20 / btc_ma60.clip(lower=1)) - 1).fillna(0) * 100
    else:
        result["Crypto_Risk_Appetite"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_international_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute international market signals."""
    result = pd.DataFrame(index=bulk.index)

    for name, label in [("EEM", "EM"), ("EFA", "EAFE"), ("FXI", "China"),
                         ("EWJ", "Japan"), ("EWZ", "Brazil"), ("INDA", "India"), ("EWG", "Europe")]:
        s = bulk.get(name, pd.Series(dtype=float))
        if not s.empty:
            result[f"{label}_Ret"] = _safe_pct_change(s, 1)
            result[f"{label}_Ret_20d"] = _safe_pct_change(s, 20)
        else:
            result[f"{label}_Ret"] = 0.0
            result[f"{label}_Ret_20d"] = 0.0

    # US vs International relative strength
    sp500 = bulk.get("SP500", pd.Series(dtype=float))
    efa = bulk.get("EFA", pd.Series(dtype=float))
    if not sp500.empty and not efa.empty:
        result["US_vs_Intl"] = _safe_pct_change(sp500, 20) - _safe_pct_change(efa, 20)
    else:
        result["US_vs_Intl"] = 0.0

    # Emerging vs Developed
    eem = bulk.get("EEM", pd.Series(dtype=float))
    if not eem.empty and not efa.empty:
        result["EM_vs_DM"] = _safe_pct_change(eem, 20) - _safe_pct_change(efa, 20)
    else:
        result["EM_vs_DM"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_volatility_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility term structure and regime features."""
    result = pd.DataFrame(index=bulk.index)

    vix = bulk.get("VIX", pd.Series(dtype=float))
    vxn = bulk.get("VXN", pd.Series(dtype=float))
    sp500 = bulk.get("SP500", pd.Series(dtype=float))

    if not vix.empty:
        result["VIX_Level"] = vix.fillna(20)
        result["VIX_Change_5d"] = vix.diff(5).fillna(0)
        # VIX z-score
        vix_mean = vix.rolling(60, min_periods=10).mean()
        vix_std = vix.rolling(60, min_periods=10).std().clip(lower=1)
        result["VIX_ZScore"] = ((vix - vix_mean) / vix_std).clip(-3, 3).fillna(0)
        # VIX term structure (VIX vs realized vol divergence)
        if not sp500.empty:
            rv20 = sp500.pct_change().rolling(20, min_periods=5).std() * np.sqrt(252) * 100
            result["VIX_RealizedVol_Gap"] = (vix - rv20).fillna(0)
        else:
            result["VIX_RealizedVol_Gap"] = 0.0
    else:
        result["VIX_Level"] = 20.0
        result["VIX_Change_5d"] = 0.0
        result["VIX_ZScore"] = 0.0
        result["VIX_RealizedVol_Gap"] = 0.0

    # NASDAQ volatility
    if not vxn.empty:
        result["VXN_Level"] = vxn.fillna(20)
    else:
        result["VXN_Level"] = 20.0

    # VIX/VXN spread
    if not vix.empty and not vxn.empty:
        result["VIX_VXN_Spread"] = (vix - vxn).fillna(0)
    else:
        result["VIX_VXN_Spread"] = 0.0

    return result.ffill().bfill().fillna(0)


def compute_market_breadth_features(bulk: pd.DataFrame) -> pd.DataFrame:
    """Compute market breadth and structure signals."""
    result = pd.DataFrame(index=bulk.index)

    sp500 = bulk.get("SP500", pd.Series(dtype=float))
    russell = bulk.get("RUSSELL2000", pd.Series(dtype=float))
    ndx = bulk.get("NASDAQ100", pd.Series(dtype=float))
    dow = bulk.get("DOW", pd.Series(dtype=float))
    nyse = bulk.get("NYSE_COMP", pd.Series(dtype=float))

    # Index returns
    for name, s in [("SP500", sp500), ("NASDAQ", bulk.get("NASDAQ", pd.Series(dtype=float))),
                     ("DOW", dow), ("RUSSELL2000", russell)]:
        if not s.empty:
            result[f"{name}_Ret"] = _safe_pct_change(s, 1)
            result[f"{name}_Ret_20d"] = _safe_pct_change(s, 20)
        else:
            result[f"{name}_Ret"] = 0.0
            result[f"{name}_Ret_20d"] = 0.0

    # Large-cap vs Small-cap (breadth proxy)
    if not sp500.empty and not russell.empty:
        result["LargeCap_vs_SmallCap"] = _safe_pct_change(sp500, 20) - _safe_pct_change(russell, 20)
    else:
        result["LargeCap_vs_SmallCap"] = 0.0

    # Narrow vs Broad (concentration risk)
    if not ndx.empty and not nyse.empty:
        result["Narrow_vs_Broad"] = _safe_pct_change(ndx, 20) - _safe_pct_change(nyse, 20)
    else:
        result["Narrow_vs_Broad"] = 0.0

    # Market advance/decline proxy (SP500 momentum dispersion)
    if not sp500.empty:
        ret_5d = _safe_pct_change(sp500, 5)
        ret_20d = _safe_pct_change(sp500, 20)
        result["Market_Momentum_5d"] = ret_5d
        result["Market_Momentum_20d"] = ret_20d

        # Breadth: rolling % of positive return days
        sp_ret = sp500.pct_change()
        result["Breadth_Positive_Pct"] = sp_ret.rolling(20, min_periods=5).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        ).fillna(0.5) * 100
    else:
        result["Market_Momentum_5d"] = 0.0
        result["Market_Momentum_20d"] = 0.0
        result["Breadth_Positive_Pct"] = 50.0

    return result.ffill().bfill().fillna(0)


# ═══════════════════════════════════════════════════════════════════════
# 4. FOMC Calendar & Rate Decision Flags
# ═══════════════════════════════════════════════════════════════════════

FOMC_MEETING_DATES = [
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17",
    "2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15",
    "2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14",
    "2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
    "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
    "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
    "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-10-29","2025-12-17",
    # 2026
    "2026-01-28","2026-03-18","2026-05-06","2026-06-17",
    "2026-07-29","2026-09-16","2026-10-28","2026-12-16",
]

FOMC_RATE_DECISIONS_BPS: Dict[str, int] = {
    "2015-12-16": 25,
    "2016-12-14": 25,
    "2017-03-15": 25, "2017-06-14": 25, "2017-12-13": 25,
    "2018-03-21": 25, "2018-06-13": 25, "2018-09-26": 25, "2018-12-19": 25,
    "2019-07-31": -25, "2019-09-18": -25, "2019-10-30": -25,
    "2020-03-03": -50, "2020-03-15": -100,
    "2022-03-16": 25, "2022-05-04": 50, "2022-06-15": 75,
    "2022-07-27": 75, "2022-09-21": 75, "2022-11-02": 75, "2022-12-14": 50,
    "2023-02-01": 25, "2023-03-22": 25, "2023-05-03": 25, "2023-07-26": 25,
    "2024-09-18": -50, "2024-11-07": -25, "2024-12-18": -25,
}


def build_fomc_features(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate FOMC policy features aligned to a date index:
      FOMC_Decision_Flag, FOMC_Rate_Surprise, Policy_Stance
    """
    fomc_set = set(pd.Timestamp(d) for d in FOMC_MEETING_DATES)

    flag = np.zeros(len(date_index), dtype=np.float32)
    surprise = np.zeros(len(date_index), dtype=np.float32)
    stance = np.zeros(len(date_index), dtype=np.float32)

    rate_map: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): bps / 100.0
        for d, bps in FOMC_RATE_DECISIONS_BPS.items()
    }

    cum_stance = 0.0
    last_surprise = 0.0
    surprise_countdown = 0

    for i, dt in enumerate(date_index):
        dt_day = pd.Timestamp(dt.date())
        if dt_day in fomc_set:
            flag[i] = 1.0
            decision = rate_map.get(dt_day, 0.0)
            last_surprise = decision
            surprise_countdown = 5
            if decision > 0:
                cum_stance = min(cum_stance + 0.25, 1.0)
            elif decision < 0:
                cum_stance = max(cum_stance - 0.25, -1.0)

        if surprise_countdown > 0:
            surprise[i] = last_surprise
            surprise_countdown -= 1

        stance[i] = cum_stance
        cum_stance *= 0.995

    return pd.DataFrame({
        "FOMC_Decision_Flag": flag,
        "FOMC_Rate_Surprise": surprise,
        "Policy_Stance": stance,
    }, index=date_index)


# ═══════════════════════════════════════════════════════════════════════
# 5. Legacy-Compatible Public API
# ═══════════════════════════════════════════════════════════════════════

def fetch_macro_data(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Legacy-compatible macro DataFrame (9 columns).
    Maps to Yahoo Finance for daily-frequency data.
    """
    cached = _get_cached(f"macro_compat_{start}")
    if cached is not None:
        return cached

    tickers_needed = {
        "YIELD_13W": "^IRX",
        "YIELD_10Y": "^TNX",
        "SP500": "^GSPC",
        "VIX": "^VIX",
    }

    bulk = fetch_yahoo_bulk(tickers_needed, start=start)
    wb = fetch_worldbank_macro(start=start)

    df = pd.DataFrame(index=bulk.index if not bulk.empty else pd.DatetimeIndex([]))

    # Fed Funds Rate (use 13-week T-bill as proxy)
    irx = bulk.get("YIELD_13W", pd.Series(dtype=float))
    if not irx.empty:
        df["Fed_Funds_Rate"] = irx / 100
        df["Delta_Fed_Funds"] = df["Fed_Funds_Rate"].diff(20)
    else:
        df["Fed_Funds_Rate"] = np.nan
        df["Delta_Fed_Funds"] = np.nan

    # Fed Balance Change (proxy from WB macro if available)
    if "M2_GDP" in wb.columns:
        m2 = wb["M2_GDP"].reindex(df.index).ffill()
        df["Fed_Balance_Change"] = m2.pct_change(20).fillna(0) * 100
    else:
        df["Fed_Balance_Change"] = 0.0

    # CPI YoY
    if "INFLATION" in wb.columns:
        df["CPI_YoY"] = wb["INFLATION"].reindex(df.index).ffill().fillna(0)
    elif "CPI" in wb.columns:
        cpi = wb["CPI"].reindex(df.index).ffill()
        df["CPI_YoY"] = cpi.pct_change(252).fillna(0) * 100
    else:
        df["CPI_YoY"] = 0.0

    # PCE YoY (use CPI as proxy)
    df["PCE_YoY"] = df["CPI_YoY"] * 0.95  # PCE typically slightly below CPI

    # Unemployment
    if "UNEMPLOYMENT" in wb.columns:
        df["Unemployment"] = wb["UNEMPLOYMENT"].reindex(df.index).ffill().fillna(4.0)
    else:
        df["Unemployment"] = 4.0

    # Term Spread (10Y - 13W)
    y10 = bulk.get("YIELD_10Y", pd.Series(dtype=float))
    if not irx.empty and not y10.empty:
        df["Term_Spread"] = (y10 - irx / 100 * 10).fillna(0)
    else:
        df["Term_Spread"] = 0.0

    # BAA Spread (VIX as credit risk proxy)
    vix = bulk.get("VIX", pd.Series(dtype=float))
    if not vix.empty:
        df["BAA_Spread"] = (vix / 10).fillna(2.0)  # rough proxy
    else:
        df["BAA_Spread"] = 2.0

    # M2 Growth
    if "M2_GDP" in wb.columns:
        m2 = wb["M2_GDP"].reindex(df.index).ffill()
        df["M2_Growth"] = m2.pct_change(252).fillna(0) * 100
    else:
        df["M2_Growth"] = 0.0

    df = df.ffill().bfill().fillna(0)
    _set_cache(f"macro_compat_{start}", df)
    return df


# Backward-compatible alias
fetch_fred_macro = fetch_macro_data


def fetch_market_context(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch market context: SP500_Ret, NASDAQ_Ret, VIX, BOND_10Y, DXY."""
    cached = _get_cached(f"market_ctx_{start}")
    if cached is not None:
        return cached

    tickers = {
        "SP500": "^GSPC", "NASDAQ": "^IXIC", "VIX": "^VIX",
        "BOND_10Y": "^TNX", "DXY": "DX-Y.NYB",
    }
    bulk = fetch_yahoo_bulk(tickers, start=start)

    result = pd.DataFrame(index=bulk.index if not bulk.empty else pd.DatetimeIndex([]))

    if "SP500" in bulk.columns:
        result["SP500_Ret"] = _safe_pct_change(bulk["SP500"])
    else:
        result["SP500_Ret"] = 0.0

    if "NASDAQ" in bulk.columns:
        result["NASDAQ_Ret"] = _safe_pct_change(bulk["NASDAQ"])
    else:
        result["NASDAQ_Ret"] = 0.0

    result["VIX"] = bulk["VIX"].fillna(20) if "VIX" in bulk.columns else 20.0
    result["BOND_10Y"] = bulk["BOND_10Y"].fillna(3) if "BOND_10Y" in bulk.columns else 3.0
    result["DXY"] = bulk["DXY"].fillna(100) if "DXY" in bulk.columns else 100.0

    result = result.ffill().bfill()
    _set_cache(f"market_ctx_{start}", result)
    return result


def fetch_sector_commodity(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch sector & commodity returns: SOXX_Ret, SMH_Ret, Gold_Ret, Oil_Ret."""
    cached = _get_cached(f"sector_comm_{start}")
    if cached is not None:
        return cached

    tickers = {"SOXX": "SOXX", "SMH": "SMH", "GOLD": "GC=F", "OIL": "CL=F"}
    bulk = fetch_yahoo_bulk(tickers, start=start)

    result = pd.DataFrame(index=bulk.index if not bulk.empty else pd.DatetimeIndex([]))
    for name, col_name in [("SOXX", "SOXX_Ret"), ("SMH", "SMH_Ret"),
                            ("GOLD", "Gold_Ret"), ("OIL", "Oil_Ret")]:
        if name in bulk.columns:
            result[col_name] = _safe_pct_change(bulk[name])
        else:
            result[col_name] = 0.0

    result = result.ffill().bfill()
    _set_cache(f"sector_comm_{start}", result)
    return result


def fetch_nasdaq_close(start: str = "2000-01-01") -> pd.Series:
    """Return raw NASDAQ close for excess-return computation."""
    df = _yf_download_safe("^IXIC", start=start)
    if not df.empty and "Close" in df.columns:
        return df["Close"].dropna()
    return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# 6. Master Data Fetcher (All Sources Combined)
# ═══════════════════════════════════════════════════════════════════════

def fetch_all_market_data(start: str = "2000-01-01") -> Dict[str, pd.DataFrame]:
    """
    Fetch ALL market data from all sources. Returns dict of DataFrames:
      - "bulk_close":    raw close prices for all 60+ Yahoo tickers
      - "worldbank":     World Bank macro indicators (annual → daily)
      - "yield_curve":   yield curve features
      - "credit":        credit/bond features
      - "sectors":       sector rotation features
      - "commodities":   commodity features
      - "currencies":    currency features
      - "crypto":        crypto risk appetite features
      - "international": international market features
      - "volatility":    volatility features
      - "breadth":       market breadth features
      - "fomc":          FOMC policy features
      - "macro_compat":  legacy 9-column macro DataFrame
    """
    logger.info("Fetching comprehensive market data from all sources...")

    # 1) Bulk Yahoo download (all 60+ tickers at once)
    bulk = fetch_yahoo_bulk(YAHOO_TICKERS, start=start)
    logger.info("Yahoo bulk: %d tickers, %d rows", len(bulk.columns), len(bulk))

    # 2) World Bank macro
    wb = fetch_worldbank_macro(start=start)
    logger.info("World Bank: %d indicators", len(wb.columns) if not wb.empty else 0)

    # 3) Derived features
    yield_curve = compute_yield_curve_features(bulk)
    credit = compute_credit_features(bulk)
    sectors = compute_sector_rotation_features(bulk)
    commodities = compute_commodity_features(bulk)
    currencies = compute_currency_features(bulk)
    crypto = compute_crypto_features(bulk)
    intl = compute_international_features(bulk)
    volatility = compute_volatility_features(bulk)
    breadth = compute_market_breadth_features(bulk)

    # 4) FOMC
    idx = bulk.index if not bulk.empty else pd.DatetimeIndex([])
    fomc = build_fomc_features(idx) if len(idx) > 0 else pd.DataFrame()

    # 5) Legacy macro compat
    macro_compat = fetch_macro_data(start=start)

    logger.info(
        "Total derived features: yield=%d, credit=%d, sectors=%d, commodities=%d, "
        "currencies=%d, crypto=%d, intl=%d, vol=%d, breadth=%d",
        len(yield_curve.columns), len(credit.columns), len(sectors.columns),
        len(commodities.columns), len(currencies.columns), len(crypto.columns),
        len(intl.columns), len(volatility.columns), len(breadth.columns),
    )

    return {
        "bulk_close": bulk,
        "worldbank": wb,
        "yield_curve": yield_curve,
        "credit": credit,
        "sectors": sectors,
        "commodities": commodities,
        "currencies": currencies,
        "crypto": crypto,
        "international": intl,
        "volatility": volatility,
        "breadth": breadth,
        "fomc": fomc,
        "macro_compat": macro_compat,
    }
