"""
ViziGenesis V2 — FRED Macro-Economic Data Ingestion
=====================================================
Fetches macro, Fed policy, and financial conditions data from:
  • FRED (Federal Reserve Economic Data) — free API
  • Yahoo Finance (for daily market proxies)

Series ingested:
  Fed Funds Rate, Fed Balance Sheet, M2 Supply,
  CPI, PCE (core), Unemployment, Nonfarm Payrolls,
  2Y/10Y Treasury yields, BAA spread, Term spread.

FOMC meeting dates and rate-decision flags are generated from
a hardcoded calendar (updated periodically) + web scrape fallback.
"""
import logging, os, json, time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

logger = logging.getLogger("vizigenesis.v2.fred")

# ───────────────────────────────────────────────────────────────────────
# In-memory cache
# ───────────────────────────────────────────────────────────────────────
_fred_cache: Dict[str, Tuple[pd.Series, float]] = {}
_CACHE_TTL = 6 * 3600  # 6 hours


# ═══════════════════════════════════════════════════════════════════════
# 1.  FRED data fetcher  (JSON API — no API key needed for most series)
# ═══════════════════════════════════════════════════════════════════════
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_series(
    series_id: str,
    start: str = "2000-01-01",
    api_key: Optional[str] = None,
    retries: int = 2,
) -> pd.Series:
    """Fetch a single FRED series as a Pandas Series (date-indexed)."""
    cached = _fred_cache.get(series_id)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0].copy()

    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        # Fallback: try yfinance macro proxy or return empty
        logger.warning("No FRED_API_KEY — using Yahoo Finance proxy for %s", series_id)
        return _yahoo_proxy_for_fred(series_id, start)

    url = (
        f"{FRED_API_BASE}?series_id={series_id}"
        f"&observation_start={start}&api_key={key}&file_type=json"
    )

    timeout_retries = max(0, int(os.getenv("FRED_TIMEOUT_RETRIES", "0")))
    request_timeout = float(os.getenv("FRED_HTTP_TIMEOUT", "8"))
    effective_retries = min(retries, timeout_retries)

    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "ViziGenesis/2.0"})
            with urlopen(req, timeout=request_timeout) as resp:
                data = json.loads(resp.read().decode())
            obs = data.get("observations", [])
            dates, values = [], []
            for o in obs:
                try:
                    v = float(o["value"])
                    dates.append(pd.Timestamp(o["date"]))
                    values.append(v)
                except (ValueError, KeyError):
                    continue
            s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
            s = s[~s.index.duplicated(keep="last")]
            _fred_cache[series_id] = (s, time.time())
            return s.copy()
        except Exception as e:
            err = str(e).lower()
            is_timeout = "timed out" in err or "timeout" in err
            logger.warning("FRED fetch %s attempt %d failed: %s", series_id, attempt, e)
            if is_timeout:
                if attempt < effective_retries:
                    time.sleep(1)
                    continue
                logger.warning("FRED timeout for %s — fallback to Yahoo proxy", series_id)
                return _yahoo_proxy_for_fred(series_id, start)

            if attempt < retries:
                time.sleep(1)

    logger.warning("FRED unavailable for %s — fallback to Yahoo proxy", series_id)
    return _yahoo_proxy_for_fred(series_id, start)


def _yahoo_proxy_for_fred(series_id: str, start: str) -> pd.Series:
    """Approximate FRED series using Yahoo Finance tickers when no API key."""
    import yfinance as yf

    proxy_map = {
        "DFF":      ("^IRX", lambda df: df["Close"] / 100),          # 3-mo T-bill ≈ fed funds
        "DGS2":     (None, None),                                     # no good Yahoo proxy
        "DGS10":    ("^TNX", lambda df: df["Close"]),                 # 10Y yield
        "WALCL":    (None, None),
        "M2SL":     (None, None),
        "CPIAUCSL": (None, None),
        "PCEPI":    (None, None),
        "PCEPILFE": (None, None),
        "UNRATE":   (None, None),
        "PAYEMS":   (None, None),
        "BAAFFM":   (None, None),
        "TEDRATE":  (None, None),
    }
    entry = proxy_map.get(series_id, (None, None))
    ticker, transform = entry
    if ticker is None:
        return pd.Series(dtype=float, name=series_id)
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
        s = transform(df)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = series_id
        return s.dropna()
    except Exception as e:
        logger.warning("Yahoo proxy for %s failed: %s", series_id, e)
        return pd.Series(dtype=float, name=series_id)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Build macro DataFrame  (all FRED features aligned to business days)
# ═══════════════════════════════════════════════════════════════════════
def fetch_fred_macro(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Return a date-indexed DataFrame with all FRED-derived features:
        Fed_Funds_Rate, Delta_Fed_Funds, Fed_Balance_Change,
        CPI_YoY, PCE_YoY, Unemployment, Term_Spread,
        BAA_Spread, M2_Growth
    All columns are forward-filled to daily frequency.
    """
    from backend.v2.config import FRED_SERIES

    raw: Dict[str, pd.Series] = {}
    for name, sid in FRED_SERIES.items():
        raw[name] = _fetch_fred_series(sid, start=start)

    # Build date index
    all_dates = set()
    for s in raw.values():
        if len(s) > 0:
            all_dates.update(s.index.tolist())
    if not all_dates:
        logger.warning("No FRED data available — returning empty macro frame")
        return pd.DataFrame()

    idx = pd.DatetimeIndex(sorted(all_dates))
    df = pd.DataFrame(index=idx)

    # Fed Funds Rate (daily)
    if len(raw.get("fed_funds", [])) > 0:
        df["Fed_Funds_Rate"] = raw["fed_funds"].reindex(idx).ffill()
        df["Delta_Fed_Funds"] = df["Fed_Funds_Rate"].diff(20)  # 1-month change
    else:
        df["Fed_Funds_Rate"] = np.nan
        df["Delta_Fed_Funds"] = np.nan

    # Fed Balance Sheet (weekly → daily ffill)
    if len(raw.get("fed_balance", [])) > 0:
        bal = raw["fed_balance"].reindex(idx).ffill()
        df["Fed_Balance_Change"] = bal.pct_change(4) * 100  # 4-week % change
    else:
        df["Fed_Balance_Change"] = np.nan

    # CPI year-over-year (monthly → daily ffill)
    if len(raw.get("cpi", [])) > 0:
        cpi = raw["cpi"].reindex(idx).ffill()
        df["CPI_YoY"] = cpi.pct_change(12 * 21) * 100  # ~12 months of trading days
    else:
        df["CPI_YoY"] = np.nan

    # PCE year-over-year
    if len(raw.get("pce", [])) > 0:
        pce = raw["pce"].reindex(idx).ffill()
        df["PCE_YoY"] = pce.pct_change(12 * 21) * 100
    else:
        df["PCE_YoY"] = np.nan

    # Unemployment rate
    if len(raw.get("unemployment", [])) > 0:
        df["Unemployment"] = raw["unemployment"].reindex(idx).ffill()
    else:
        df["Unemployment"] = np.nan

    # Term spread (10Y − 2Y)
    t10 = raw.get("treasury_10y", pd.Series(dtype=float))
    t2 = raw.get("treasury_2y", pd.Series(dtype=float))
    if len(t10) > 0 and len(t2) > 0:
        df["Term_Spread"] = (
            t10.reindex(idx).ffill() - t2.reindex(idx).ffill()
        )
    else:
        df["Term_Spread"] = np.nan

    # BAA corporate spread
    if len(raw.get("baa_spread", [])) > 0:
        df["BAA_Spread"] = raw["baa_spread"].reindex(idx).ffill()
    else:
        df["BAA_Spread"] = np.nan

    # M2 growth (monthly → daily, YoY %)
    if len(raw.get("m2", [])) > 0:
        m2 = raw["m2"].reindex(idx).ffill()
        df["M2_Growth"] = m2.pct_change(12 * 21) * 100
    else:
        df["M2_Growth"] = np.nan

    df = df.ffill().bfill()
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3.  FOMC calendar & rate-decision flags
# ═══════════════════════════════════════════════════════════════════════

# Hardcoded FOMC meeting dates (2015–2026) — last day of each meeting
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

# Approximate rate decisions (bps) — positive = hike, negative = cut
# Only major moves listed; 0 = hold or unmapped
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
      • FOMC_Decision_Flag   — 1 on FOMC meeting day, 0 otherwise
      • FOMC_Rate_Surprise   — rate change in bps (held for 5 days after meeting)
      • Policy_Stance        — cumulative stance: +1 hawkish, −1 dovish, 0 neutral
    """
    fomc_set = set(pd.Timestamp(d) for d in FOMC_MEETING_DATES)

    flag = np.zeros(len(date_index), dtype=np.float32)
    surprise = np.zeros(len(date_index), dtype=np.float32)
    stance = np.zeros(len(date_index), dtype=np.float32)

    # Build mapping: date → rate decision
    rate_map: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): bps / 100.0
        for d, bps in FOMC_RATE_DECISIONS_BPS.items()
    }

    # Running policy stance (cumulative direction of rate moves)
    cum_stance = 0.0
    last_surprise = 0.0
    surprise_countdown = 0

    for i, dt in enumerate(date_index):
        dt_day = pd.Timestamp(dt.date())

        if dt_day in fomc_set:
            flag[i] = 1.0
            decision = rate_map.get(dt_day, 0.0)
            last_surprise = decision
            surprise_countdown = 5  # hold surprise for 5 trading days
            if decision > 0:
                cum_stance = min(cum_stance + 0.25, 1.0)
            elif decision < 0:
                cum_stance = max(cum_stance - 0.25, -1.0)

        if surprise_countdown > 0:
            surprise[i] = last_surprise
            surprise_countdown -= 1

        stance[i] = cum_stance
        # Mean revert stance slowly when no meeting
        cum_stance *= 0.995

    return pd.DataFrame({
        "FOMC_Decision_Flag": flag,
        "FOMC_Rate_Surprise": surprise,
        "Policy_Stance": stance,
    }, index=date_index)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Market-context fetcher (Yahoo)
# ═══════════════════════════════════════════════════════════════════════
def fetch_market_context(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetch benchmark / market tickers from Yahoo:
      SP500_Ret, NASDAQ_Ret, VIX, BOND_10Y, DXY
    """
    import yfinance as yf
    from backend.v2.config import (
        BENCHMARK_TICKER, NASDAQ_TICKER, VIX_TICKER,
        BOND_10Y_TICKER, DXY_TICKER,
    )

    tickers = {
        "SP500":   BENCHMARK_TICKER,
        "NASDAQ":  NASDAQ_TICKER,
        "VIX":     VIX_TICKER,
        "BOND_10Y": BOND_10Y_TICKER,
        "DXY":     DXY_TICKER,
    }

    frames = {}
    for name, tkr in tickers.items():
        try:
            df = yf.download(tkr, start=start, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                frames[name] = df["Close"]
        except Exception as e:
            logger.warning("Market context fetch %s (%s) failed: %s", name, tkr, e)

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames)

    result = pd.DataFrame(index=combined.index)
    if "SP500" in combined.columns:
        result["SP500_Ret"] = combined["SP500"].pct_change(fill_method=None) * 100
    else:
        result["SP500_Ret"] = 0.0

    if "NASDAQ" in combined.columns:
        result["NASDAQ_Ret"] = combined["NASDAQ"].pct_change(fill_method=None) * 100
    else:
        result["NASDAQ_Ret"] = 0.0

    if "VIX" in combined.columns:
        result["VIX"] = combined["VIX"]
    else:
        result["VIX"] = 20.0  # long-run average

    if "BOND_10Y" in combined.columns:
        result["BOND_10Y"] = combined["BOND_10Y"]
    else:
        result["BOND_10Y"] = 3.0

    if "DXY" in combined.columns:
        result["DXY"] = combined["DXY"]
    else:
        result["DXY"] = 100.0

    result = result.ffill().bfill()
    return result


def fetch_sector_commodity(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch SOXX, SMH, Gold, Oil returns."""
    import yfinance as yf
    from backend.v2.config import SECTOR_TICKERS, COMMODITY_TICKERS

    all_tickers = {}
    all_tickers.update(SECTOR_TICKERS)
    all_tickers.update(COMMODITY_TICKERS)

    frames = {}
    for name, tkr in all_tickers.items():
        try:
            df = yf.download(tkr, start=start, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                frames[name] = df["Close"]
        except Exception as e:
            logger.warning("Sector/commodity fetch %s failed: %s", name, e)

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames)
    result = pd.DataFrame(index=combined.index)

    for name in ["SOXX", "SMH", "Gold", "Oil"]:
        col = f"{name}_Ret"
        if name in combined.columns:
            result[col] = combined[name].pct_change(fill_method=None) * 100
        else:
            result[col] = 0.0

    return result.ffill().bfill()


def fetch_nasdaq_close(start: str = "2000-01-01") -> pd.Series:
    """Return raw NASDAQ close series for excess-return computation."""
    import yfinance as yf
    from backend.v2.config import NASDAQ_TICKER
    try:
        df = yf.download(NASDAQ_TICKER, start=start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)
