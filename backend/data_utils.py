"""
Data fetching, caching, and preprocessing utilities.
Uses yfinance with an in-memory TTL cache to minimize API calls.
"""
import time, io, csv, os, json, re, shutil
import importlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import logging
import hashlib

logger = logging.getLogger("vizigenesis.data_utils")

TvDatafeed = None
Interval = None
for _tv_mod in ("tvDatafeed", "tvdatafeed"):
    try:
        _loaded = importlib.import_module(_tv_mod)
        TvDatafeed = getattr(_loaded, "TvDatafeed", None)
        Interval = getattr(_loaded, "Interval", None)
        if TvDatafeed is not None and Interval is not None:
            break
    except Exception:
        continue
try:
    import financedatabase as fd
except Exception:
    fd = None

from backend.config import (
    CACHE_TTL_REALTIME, CACHE_TTL_HISTORY,
    FEATURE_COLS, TARGET_COL, SEQUENCE_LENGTH, DATA_DIR,
    TIME_WEIGHT_MAP, QUANT_FEATURE_COLS,
)

try:
    from backend.macro_data import (
        fetch_macro_data, merge_macro_with_stock, MACRO_FEATURE_NAMES,
        fetch_quant_extra_data, merge_quant_extra_with_stock, QUANT_EXTRA_FEATURE_NAMES,
        fetch_nasdaq_close,
    )
except ImportError:
    fetch_macro_data = None        # type: ignore[assignment]
    merge_macro_with_stock = None  # type: ignore[assignment]
    MACRO_FEATURE_NAMES = []       # type: ignore[assignment]
    fetch_quant_extra_data = None  # type: ignore[assignment]
    merge_quant_extra_with_stock = None  # type: ignore[assignment]
    QUANT_EXTRA_FEATURE_NAMES = []  # type: ignore[assignment]
    fetch_nasdaq_close = None      # type: ignore[assignment]

try:
    from backend.quant_features import (
        add_quant_technical_indicators,
        generate_quant_targets,
    )
except ImportError:
    add_quant_technical_indicators = None  # type: ignore[assignment]
    generate_quant_targets = None          # type: ignore[assignment]

# ── Feature sets ──────────────────────────────────────────────────────
PRO_FEATURE_COLS = [
    # OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Classic technical indicators
    "MA20", "MA50", "EMA20", "RSI", "MACD", "Bollinger_Band", "OBV",
    # Additional engineered features
    "Volume_Change", "Volatility", "ATR",
    # FED / Macro-economic context
    "SP500", "NASDAQ", "VIX", "BOND_10Y", "INFLATION_PROXY",
]

AI_MODE_CONFIG = {
    "simple": {
        "period": "2y",
        "default_epochs": 80,
        "feature_cols": FEATURE_COLS,
    },
    "pro": {
        "period": "10y",
        "default_epochs": 220,
        "feature_cols": PRO_FEATURE_COLS,
    },
    "quant": {
        "period": "10y",
        "default_epochs": 150,
        "feature_cols": QUANT_FEATURE_COLS,
    },
}


def normalize_ai_mode(mode: str) -> str:
    raw = (mode or "simple").strip().lower()
    if raw in {"pro", "professional", "advanced"}:
        return "pro"
    if raw in {"quant", "quantitative", "institutional", "hedge"}:
        return "quant"
    return "simple"


def get_mode_config(mode: str) -> dict:
    return AI_MODE_CONFIG[normalize_ai_mode(mode)]


def get_feature_columns(mode: str = "simple") -> List[str]:
    cfg = get_mode_config(mode)
    return list(cfg["feature_cols"])


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    vol = out["Volume"].astype(float)

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
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    out["Bollinger_Band"] = ((close - bb_lower) / bb_width).clip(lower=0, upper=1)

    out["OBV"] = (np.sign(close.diff().fillna(0)) * vol).cumsum()

    # ── Additional engineered features ────────────────────────────────
    # Volume change (%)
    out["Volume_Change"] = vol.pct_change().fillna(0).clip(-5, 5)

    # Volatility — rolling coefficient of variation (20-day)
    returns = close.pct_change().fillna(0)
    out["Volatility"] = returns.rolling(20, min_periods=20).std().fillna(0)

    # ATR — Average True Range (14-period)
    high_col = out["High"].astype(float)
    low_col  = out["Low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high_col - low_col,
        (high_col - prev_close).abs(),
        (low_col  - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=14).mean()

    return out


def prepare_feature_dataframe(df: pd.DataFrame, mode: str = "simple") -> pd.DataFrame:
    selected_mode = normalize_ai_mode(mode)
    feat_cols = get_feature_columns(selected_mode)

    if selected_mode == "quant":
        # Quant mode: use advanced feature engineering
        if add_quant_technical_indicators is not None:
            out = add_quant_technical_indicators(df)
        else:
            out = add_technical_indicators(df)

        # Merge classic macro features
        macro_cols = [c for c in feat_cols if c in
                      (MACRO_FEATURE_NAMES if MACRO_FEATURE_NAMES else [])]
        if macro_cols and fetch_macro_data is not None:
            try:
                macro_df = fetch_macro_data(period="10y")
                out = merge_macro_with_stock(out, macro_df)
            except Exception as exc:
                logger.warning("Macro merge skipped (quant): %s", exc)
                for col in macro_cols:
                    if col not in out.columns:
                        out[col] = 0.0
        else:
            for col in macro_cols:
                if col not in out.columns:
                    out[col] = 0.0

        # Merge quant extra features (SOXX, SMH, Gold, Oil, DXY)
        quant_extra_cols = [c for c in feat_cols if c in
                           (QUANT_EXTRA_FEATURE_NAMES if QUANT_EXTRA_FEATURE_NAMES else [])]
        if quant_extra_cols and fetch_quant_extra_data is not None:
            try:
                quant_df = fetch_quant_extra_data(period="10y")
                out = merge_quant_extra_with_stock(out, quant_df)
            except Exception as exc:
                logger.warning("Quant extra merge skipped: %s", exc)
                for col in quant_extra_cols:
                    if col not in out.columns:
                        out[col] = 0.0
        else:
            for col in quant_extra_cols:
                if col not in out.columns:
                    out[col] = 0.0

    elif selected_mode == "pro":
        out = add_technical_indicators(df)

        # ── Merge macro-economic features (FED data) ─────────────────
        macro_cols = [c for c in feat_cols if c in
                      (MACRO_FEATURE_NAMES if MACRO_FEATURE_NAMES else [])]
        if macro_cols and fetch_macro_data is not None:
            try:
                macro_df = fetch_macro_data(period="10y")
                out = merge_macro_with_stock(out, macro_df)
            except Exception as exc:
                logger.warning("Macro merge skipped: %s", exc)
                for col in macro_cols:
                    if col not in out.columns:
                        out[col] = 0.0
        else:
            for col in macro_cols:
                if col not in out.columns:
                    out[col] = 0.0
    else:
        out = df.copy()

    for col in feat_cols:
        if col not in out.columns:
            out[col] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[feat_cols].copy()
    out = out.bfill().ffill().dropna()
    return out


def prepare_quant_targets_from_raw(
    df_raw: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare quant target variables aligned to features_df index.
    Fetches NASDAQ close for excess return calculation.
    """
    if generate_quant_targets is None:
        raise RuntimeError("quant_features module not available")

    nasdaq_close = None
    if fetch_nasdaq_close is not None:
        try:
            nasdaq_close = fetch_nasdaq_close(period="10y")
        except Exception as exc:
            logger.warning("NASDAQ close fetch failed for excess return: %s", exc)

    # Use the features_df which has proper index alignment
    targets = generate_quant_targets(features_df, nasdaq_close=nasdaq_close)

    # Align to features_df index
    targets = targets.reindex(features_df.index)
    return targets

# ── Simple TTL cache ──────────────────────────────────────────────────
_cache: Dict[str, Tuple[float, object]] = {}
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
_symbol_catalog_cache: Optional[List[dict]] = None
_tv_client = None


def _get_cached(key: str, ttl: int):
    """Return cached value if still fresh, else None."""
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < ttl:
            return val
    return None


def _set_cached(key: str, val):
    _cache[key] = (time.time(), val)


def _get_tv_client():
    global _tv_client
    if TvDatafeed is None:
        return None
    if _tv_client is None:
        try:
            tv_user = os.getenv("TV_USERNAME", "").strip()
            tv_pass = os.getenv("TV_PASSWORD", "").strip()
            if tv_user and tv_pass:
                _tv_client = TvDatafeed(username=tv_user, password=tv_pass)
            else:
                _tv_client = TvDatafeed()
        except Exception:
            _tv_client = None
    return _tv_client


def _resolve_tv_candidates(symbol: str) -> List[Tuple[str, str]]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return []

    if "." in sym:
        base, market = sym.split(".", 1)
        if market == "VN":
            return [(base, "HOSE"), (base, "HNX"), (base, "UPCOM")]
        if market == "HK":
            return [(base, "HKEX")]
        if market == "TO":
            return [(base, "TSX")]

    return [(sym, "NASDAQ"), (sym, "NYSE"), (sym, "AMEX")]


def _period_to_n_bars(period: str) -> int:
    raw = (period or "2y").strip().lower()
    mapping = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "max": 5000,
    }
    if raw in mapping:
        return mapping[raw]

    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", raw)
    if not m:
        return 730

    num = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        return max(20, num)
    if unit == "wk":
        return max(30, num * 7)
    if unit == "mo":
        return max(30, num * 30)
    if unit == "y":
        return max(365, num * 365)
    return 730


def _normalize_yahoo_range(period: str) -> str:
    raw = (period or "2y").strip().lower()
    allowed = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if raw in allowed:
        return raw
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", raw)
    if not m:
        return "2y"
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "wk":
        return f"{max(1, n)}wk"
    return f"{max(1, n)}{unit}"


def _get_historical_data_from_yahoo_chart(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch OHLCV using Yahoo chart endpoint directly (no yfinance wrappers)."""
    rng = _normalize_yahoo_range(period)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}"
        f"?range={quote_plus(rng)}&interval=1d&includePrePost=false&events=div%2Csplits"
    )
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None

    chart = (payload or {}).get("chart", {})
    result_list = chart.get("result") or []
    if not result_list:
        return None

    result = result_list[0] or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    n = min(len(timestamps), len(opens), len(highs), len(lows), len(closes), len(volumes))
    if n <= 0:
        return None

    rows = []
    for i in range(n):
        o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]
        if None in (o, h, l, c, v):
            continue
        try:
            rows.append({
                "Date": datetime.utcfromtimestamp(int(timestamps[i])),
                "Open": float(o),
                "High": float(h),
                "Low": float(l),
                "Close": float(c),
                "Volume": float(v),
            })
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    return df


def _get_historical_data_from_tv(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    if TvDatafeed is None or Interval is None:
        return None

    tv = _get_tv_client()
    if tv is None:
        return None

    n_bars = _period_to_n_bars(period)
    candidates = _resolve_tv_candidates(symbol)

    for candidate_symbol, exchange in candidates:
        try:
            df = tv.get_hist(
                symbol=candidate_symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars,
            )
            if df is None or df.empty:
                continue

            out = df.copy()
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            out = out.rename(columns=rename_map)
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in out.columns for col in required):
                continue

            out = out[required].copy()
            out.index = pd.to_datetime(out.index)
            out = out.sort_index().dropna()
            if len(out) >= 60:
                return out
        except Exception:
            continue

    return None


def _resolve_tv_scan_context(symbol: str) -> Tuple[str, List[str]]:
    sym = (symbol or "").upper().strip()
    if "." in sym:
        base, market = sym.split(".", 1)
        if market == "VN":
            return "vietnam", [f"HOSE:{base}", f"HNX:{base}", f"UPCOM:{base}"]
        if market == "HK":
            return "global", [f"HKEX:{base}"]
        if market == "TO":
            return "america", [f"TSX:{base}"]

    return "america", [f"NASDAQ:{sym}", f"NYSE:{sym}", f"AMEX:{sym}"]


def _scan_tv_quotes(screener: str, tickers: List[str]) -> List[dict]:
    if not tickers:
        return []

    url = f"https://scanner.tradingview.com/{screener}/scan"
    payload = {
        "symbols": {
            "tickers": tickers,
            "query": {"types": []},
        },
        "columns": [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "market_cap_basic",
            "change_abs",
            "change",
            "description",
        ],
    }

    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )
        with urlopen(req, timeout=4) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return (raw or {}).get("data", []) or []
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []
    except Exception:
        return []


def _row_to_quote_from_tv_scan(row: dict, symbol: str, source: str = "tradingview_scan") -> Optional[dict]:
    values = row.get("d") or []
    if len(values) < 2:
        return None

    close = values[0]
    open_v = values[1] if len(values) > 1 else None
    high = values[2] if len(values) > 2 else None
    low = values[3] if len(values) > 3 else None
    volume = values[4] if len(values) > 4 else None
    market_cap = values[5] if len(values) > 5 else None
    change_abs = values[6] if len(values) > 6 else None

    if close is None:
        return None

    try:
        close_f = float(close)
    except Exception:
        return None

    prev_close = None
    try:
        if change_abs is not None:
            prev_close = round(close_f - float(change_abs), 2)
    except Exception:
        prev_close = None

    exchange_symbol = str(row.get("s") or "")
    exchange = exchange_symbol.split(":", 1)[0] if ":" in exchange_symbol else None

    return {
        "symbol": symbol.upper(),
        "price": round(close_f, 2),
        "open": round(float(open_v), 2) if open_v is not None else None,
        "high": round(float(high), 2) if high is not None else None,
        "low": round(float(low), 2) if low is not None else None,
        "volume": int(volume) if volume is not None else None,
        "prev_close": prev_close,
        "market_cap": float(market_cap) if market_cap is not None else None,
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "exchange": exchange,
    }


def _get_realtime_quotes_from_tv_scan(symbols: List[str]) -> Dict[str, dict]:
    by_screener: Dict[str, List[str]] = {}
    ticker_to_symbol: Dict[str, str] = {}

    for symbol in symbols:
        screener, tickers = _resolve_tv_scan_context(symbol)
        for ticker in tickers:
            ticker_up = ticker.upper()
            ticker_to_symbol[ticker_up] = symbol.upper()
            by_screener.setdefault(screener, []).append(ticker_up)

    out: Dict[str, dict] = {}
    for screener, tickers in by_screener.items():
        unique_tickers = list(dict.fromkeys(tickers))
        rows = _scan_tv_quotes(screener, unique_tickers)
        for row in rows:
            row_symbol = str(row.get("s") or "").upper()
            mapped_symbol = ticker_to_symbol.get(row_symbol)
            if not mapped_symbol:
                continue
            if mapped_symbol in out:
                continue
            quote = _row_to_quote_from_tv_scan(row, mapped_symbol, source="tradingview_scan_batch")
            if quote is not None:
                out[mapped_symbol] = quote

    return out


def _get_realtime_price_from_tv_scan(symbol: str) -> Optional[dict]:
    screener, tickers = _resolve_tv_scan_context(symbol)
    if not tickers:
        return None

    rows = _scan_tv_quotes(screener, tickers)
    if not rows:
        return None

    for row in rows:
        quote = _row_to_quote_from_tv_scan(row, symbol, source="tradingview_scan")
        if quote is not None:
            return quote

    return None


def _parse_tv_df(symbol: str, exchange: str, df: pd.DataFrame) -> Optional[dict]:
    if df is None or df.empty:
        return None

    rows = df.dropna()
    if rows.empty:
        return None

    last_row = rows.iloc[-1]
    prev_row = rows.iloc[-2] if len(rows) >= 2 else rows.iloc[-1]

    ts = rows.index[-1]
    timestamp = ts.isoformat() if hasattr(ts, "isoformat") else datetime.utcnow().isoformat()

    if pd.isna(last_row.get("close")):
        return None

    return {
        "symbol": symbol.upper(),
        "price": round(float(last_row.get("close")), 2),
        "open": round(float(last_row.get("open")), 2) if pd.notna(last_row.get("open")) else None,
        "high": round(float(last_row.get("high")), 2) if pd.notna(last_row.get("high")) else None,
        "low": round(float(last_row.get("low")), 2) if pd.notna(last_row.get("low")) else None,
        "volume": int(last_row.get("volume")) if pd.notna(last_row.get("volume")) else None,
        "prev_close": round(float(prev_row.get("close")), 2) if pd.notna(prev_row.get("close")) else None,
        "market_cap": None,
        "timestamp": timestamp,
        "source": "tradingview_tvdatafeed",
        "exchange": exchange,
    }


def _get_realtime_price_from_tv(symbol: str) -> Optional[dict]:
    if TvDatafeed is None or Interval is None:
        return None

    tv = _get_tv_client()
    if tv is None:
        return None

    candidates = _resolve_tv_candidates(symbol)
    for candidate_symbol, exchange in candidates:
        try:
            df = tv.get_hist(
                symbol=candidate_symbol,
                exchange=exchange,
                interval=Interval.in_1_min,
                n_bars=3,
            )
            parsed = _parse_tv_df(symbol, exchange, df)
            if parsed:
                return parsed
        except Exception:
            pass

        try:
            df = tv.get_hist(
                symbol=candidate_symbol,
                exchange=exchange,
                interval=Interval.in_1_hour,
                n_bars=3,
            )
            parsed = _parse_tv_df(symbol, exchange, df)
            if parsed:
                return parsed
        except Exception:
            pass

    return None


def _build_symbol_catalog() -> List[dict]:
    """Load tradable symbols from financedatabase once and keep in memory."""
    global _symbol_catalog_cache
    if _symbol_catalog_cache is not None:
        return _symbol_catalog_cache

    catalog: List[dict] = []
    if fd is not None:
        try:
            equities = fd.Equities()
            table = equities.select()
            if isinstance(table, pd.DataFrame) and not table.empty:
                records = table.reset_index().to_dict("records")
                for row in records:
                    symbol = str(row.get("symbol") or row.get("index") or row.get("ticker") or "").upper().strip()
                    if not symbol:
                        continue
                    name = str(row.get("name") or row.get("short_name") or row.get("long_name") or symbol).strip()
                    exchange = str(row.get("exchange") or row.get("mic") or row.get("market") or "").upper().strip()
                    country = str(row.get("country") or "").strip()
                    if symbol.endswith(".VN"):
                        symbol_variant = symbol
                    elif country.lower() in {"vietnam", "vn"} and "." not in symbol:
                        symbol_variant = f"{symbol}.VN"
                    else:
                        symbol_variant = symbol
                    catalog.append({
                        "symbol": symbol_variant,
                        "name": name,
                        "exchange": exchange,
                        "country": country,
                    })
        except Exception:
            catalog = []

    if not catalog:
        catalog = [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "country": "United States"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "country": "United States"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "country": "United States"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ", "country": "United States"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "country": "United States"},
            {"symbol": "VIC.VN", "name": "Vingroup", "exchange": "HOSE", "country": "Vietnam"},
            {"symbol": "VNM.VN", "name": "Vinamilk", "exchange": "HOSE", "country": "Vietnam"},
            {"symbol": "HPG.VN", "name": "Hoa Phat Group", "exchange": "HOSE", "country": "Vietnam"},
        ]

    uniq: Dict[str, dict] = {}
    for item in catalog:
        symbol = item["symbol"].upper()
        if symbol not in uniq:
            uniq[symbol] = item

    _symbol_catalog_cache = list(uniq.values())
    return _symbol_catalog_cache


def search_symbol_catalog(query: str, limit: int = 12) -> List[dict]:
    """Find matching symbols for autocomplete."""
    q = (query or "").strip().upper()
    if len(q) < 1:
        return []

    limit = max(1, min(int(limit), 30))
    catalog = _build_symbol_catalog()

    ranked = []
    for item in catalog:
        symbol = item.get("symbol", "").upper()
        name = item.get("name", "")
        name_up = name.upper()

        if symbol.startswith(q):
            score = 0
        elif q in symbol:
            score = 1
        elif name_up.startswith(q):
            score = 2
        elif q in name_up:
            score = 3
        else:
            continue

        ranked.append((score, len(symbol), symbol, item))

    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    return [
        {
            "symbol": r[3].get("symbol"),
            "name": r[3].get("name"),
            "exchange": r[3].get("exchange"),
            "country": r[3].get("country"),
        }
        for r in ranked[:limit]
    ]


# ── Real-time price ──────────────────────────────────────────────────
def get_realtime_price(symbol: str) -> dict:
    """Fetch current quote from Yahoo Finance (cached 60 s)."""
    key = f"rt_{symbol.upper()}"
    cached = _get_cached(key, CACHE_TTL_REALTIME)
    if cached and cached.get("source") != "local_sample_csv":
        return cached

    tv_scan_result = _get_realtime_price_from_tv_scan(symbol)
    if tv_scan_result is not None:
        _set_cached(key, tv_scan_result)
        return tv_scan_result

    tv_result = _get_realtime_price_from_tv(symbol)
    if tv_result is not None:
        _set_cached(key, tv_result)
        return tv_result

    # Try Yahoo realtime first
    for _ in range(2):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            last_price = float(info.last_price) if info.last_price else None
            if last_price is None:
                raise ValueError("Yahoo realtime không có giá hợp lệ")

            result = {
                "symbol": symbol.upper(),
                "price": round(last_price, 2),
                "open": round(float(info.open), 2) if info.open else None,
                "high": round(float(info.day_high), 2) if info.day_high else None,
                "low": round(float(info.day_low), 2) if info.day_low else None,
                "volume": int(info.last_volume) if info.last_volume else None,
                "prev_close": round(float(info.previous_close), 2) if info.previous_close else None,
                "market_cap": float(info.market_cap) if info.market_cap else None,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "yahoo_finance",
            }
            _set_cached(key, result)
            return result
        except Exception:
            time.sleep(0.8)

    # Fallback from Yahoo recent daily bars (still from Yahoo, less real-time than quote feed).
    try:
        ticker = yf.Ticker(symbol)
        recent = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if isinstance(recent, pd.DataFrame) and not recent.empty:
            recent = recent.dropna()
            if len(recent) >= 1:
                last_row = recent.iloc[-1]
                prev_row = recent.iloc[-2] if len(recent) >= 2 else recent.iloc[-1]
                result = {
                    "symbol": symbol.upper(),
                    "price": round(float(last_row["Close"]), 2),
                    "open": round(float(last_row["Open"]), 2) if pd.notna(last_row.get("Open")) else None,
                    "high": round(float(last_row["High"]), 2) if pd.notna(last_row.get("High")) else None,
                    "low": round(float(last_row["Low"]), 2) if pd.notna(last_row.get("Low")) else None,
                    "volume": int(last_row["Volume"]) if pd.notna(last_row.get("Volume")) else None,
                    "prev_close": round(float(prev_row["Close"]), 2) if pd.notna(prev_row.get("Close")) else None,
                    "market_cap": None,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "yahoo_history_fallback",
                }
                _set_cached(key, result)
                return result
    except Exception:
        pass

    raise ValueError(
        f"Không thể lấy giá realtime cho {symbol.upper()}. "
        "Yahoo Finance đang lỗi mạng hoặc mã chưa có dữ liệu realtime hợp lệ."
    )


def get_symbol_news(symbol: str, limit: int = 8) -> List[dict]:
    """Fetch recent related news for a symbol from Yahoo Finance with graceful fallback."""
    key = f"news_{symbol.upper()}_{limit}"
    cached = _get_cached(key, 900)
    if cached is not None:
        return cached

    def _canonical_link(item: dict, content: dict) -> str:
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            return canonical.get("url") or item.get("link") or "#"
        if isinstance(canonical, str) and canonical:
            return canonical
        return item.get("link") or item.get("url") or "#"

    def _publisher(item: dict, content: dict) -> str:
        provider = content.get("provider")
        if isinstance(provider, dict):
            return provider.get("displayName") or item.get("publisher") or "Yahoo Finance"
        if isinstance(provider, str) and provider:
            return provider
        return item.get("publisher") or "Yahoo Finance"

    def _news_from_rss(sym: str, max_items: int) -> List[dict]:
        try:
            rss_url = (
                "https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={quote_plus(sym.upper())}&region=US&lang=en-US"
            )
            req = Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=6) as resp:
                xml_raw = resp.read()

            root = ET.fromstring(xml_raw)
            rows: List[dict] = []
            for node in root.findall("./channel/item")[:max_items]:
                title = (node.findtext("title") or f"Tin mới về {sym.upper()}").strip()
                link = (node.findtext("link") or "#").strip()
                summary = (node.findtext("description") or "Cập nhật thị trường liên quan đến mã đang theo dõi.").strip()
                published = (node.findtext("pubDate") or "").strip() or datetime.utcnow().isoformat()
                source = (node.findtext("source") or "Yahoo Finance RSS").strip()
                rows.append({
                    "title": title,
                    "publisher": source,
                    "link": link,
                    "summary": summary,
                    "published_at": published,
                })
            return rows
        except Exception:
            return []

    news_items: List[dict] = []
    try:
        ticker = yf.Ticker(symbol)
        raw_news = getattr(ticker, "news", None) or []
        for item in raw_news[:limit]:
            content = item.get("content") or {}
            pub_ts = content.get("pubDate") or item.get("providerPublishTime")
            news_items.append({
                "title": content.get("title") or item.get("title") or f"Tin mới về {symbol.upper()}",
                "publisher": _publisher(item, content),
                "link": _canonical_link(item, content),
                "summary": content.get("summary") or item.get("summary") or "Cập nhật thị trường liên quan đến mã đang theo dõi.",
                "published_at": pub_ts,
            })
    except Exception:
        news_items = []

    if not news_items:
        news_items = _news_from_rss(symbol, limit)

    if news_items:
        seen = set()
        deduped: List[dict] = []
        for item in news_items:
            unique_key = (str(item.get("link") or "").strip(), str(item.get("title") or "").strip())
            if unique_key in seen:
                continue
            seen.add(unique_key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        news_items = deduped

    if not news_items:
        news_items = [
            {
                "title": f"Đang theo dõi diễn biến mới nhất của {symbol.upper()}",
                "publisher": "ViziGenesis",
                "link": "#",
                "summary": f"Hiện chưa lấy được feed tin tức trực tiếp cho {symbol.upper()}. Hệ thống vẫn tiếp tục theo dõi giá, xu hướng và tín hiệu kỹ thuật theo thời gian thực.",
                "published_at": datetime.utcnow().isoformat(),
            }
        ]

    _set_cached(key, news_items)
    return news_items


def get_realtime_quotes(symbols: List[str]) -> List[dict]:
    """Fetch multi-symbol realtime quotes from Yahoo in a single request."""
    clean_symbols = []
    seen = set()
    for symbol in symbols or []:
        sym = str(symbol or "").upper().strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        clean_symbols.append(sym)

    if not clean_symbols:
        return []

    key = f"rt_bulk_{','.join(clean_symbols)}"
    cached_bulk = _get_cached(key, CACHE_TTL_REALTIME)
    if cached_bulk is not None:
        return cached_bulk

    joined = ",".join(clean_symbols)
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={quote_plus(joined)}"

    result_map: Dict[str, dict] = {}
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))

        quote_rows = (payload or {}).get("quoteResponse", {}).get("result", []) or []
        for row in quote_rows:
            sym = str(row.get("symbol") or "").upper().strip()
            price = row.get("regularMarketPrice")
            if not sym or price is None:
                continue

            prev_close = row.get("regularMarketPreviousClose")
            out = {
                "symbol": sym,
                "price": round(float(price), 2),
                "open": round(float(row.get("regularMarketOpen")), 2) if row.get("regularMarketOpen") is not None else None,
                "high": round(float(row.get("regularMarketDayHigh")), 2) if row.get("regularMarketDayHigh") is not None else None,
                "low": round(float(row.get("regularMarketDayLow")), 2) if row.get("regularMarketDayLow") is not None else None,
                "volume": int(row.get("regularMarketVolume")) if row.get("regularMarketVolume") is not None else None,
                "prev_close": round(float(prev_close), 2) if prev_close is not None else None,
                "market_cap": float(row.get("marketCap")) if row.get("marketCap") is not None else None,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "yahoo_quote_batch",
            }
            result_map[sym] = out
            _set_cached(f"rt_{sym}", out)
    except Exception:
        result_map = {}

    missing = [sym for sym in clean_symbols if sym not in result_map]
    if missing:
        tv_batch = _get_realtime_quotes_from_tv_scan(missing)
        for sym, quote in tv_batch.items():
            result_map[sym] = quote
            _set_cached(f"rt_{sym}", quote)

    # Fast path: for missing tickers, use existing cache only (no extra network calls).
    for sym in clean_symbols:
        if sym in result_map:
            continue
        cached_single = _get_cached(f"rt_{sym}", CACHE_TTL_REALTIME)
        if cached_single is not None and cached_single.get("source") != "local_sample_csv":
            result_map[sym] = cached_single

    ordered = [result_map[sym] for sym in clean_symbols if sym in result_map]
    _set_cached(key, ordered)
    return ordered


# ── Historical OHLCV ─────────────────────────────────────────────────
def get_historical_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download historical daily OHLCV (cached 1 h)."""
    key = f"hist_{symbol.upper()}_{period}"
    cached = _get_cached(key, CACHE_TTL_HISTORY)
    if cached is not None:
        return cached

    df = pd.DataFrame()
    last_error = None

    # Try Yahoo Chart API directly first (usually more stable than wrappers).
    try:
        direct_df = _get_historical_data_from_yahoo_chart(symbol, period=period)
        if direct_df is not None and not direct_df.empty:
            _set_cached(key, direct_df)
            return direct_df
    except Exception as exc:
        last_error = exc

    # Retry with yf.download first
    for _ in range(3):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
                actions=False,
            )
            if not df.empty:
                break
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)

    # Fallback to ticker.history if download endpoint is flaky
    if df.empty:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d", auto_adjust=False)
        except Exception as exc:
            last_error = exc

    if df.empty:
        if last_error:
            raise ValueError(
                f"Không có dữ liệu cho mã {symbol.upper()} từ Yahoo Finance. "
                f"Chi tiết: {last_error}"
            )
        raise ValueError(f"Không có dữ liệu cho mã {symbol.upper()}.")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    _set_cached(key, df)
    return df


# ── Preprocessing for LSTM ───────────────────────────────────────────
def prepare_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    train_ratio: float = 0.8,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale features, create sliding-window sequences, split into train/val.
    Returns (X_train, y_train, X_val, y_val, scaler).
    """
    selected_features = feature_cols or FEATURE_COLS
    data = df[selected_features].values.astype(np.float32)
    scaler = MinMaxScaler()

    split_idx = int(len(data) * train_ratio)
    split_idx = max(split_idx, seq_len + 5)
    split_idx = min(split_idx, len(data) - 1)

    scaler.fit(data[:split_idx])
    scaled = scaler.transform(data)

    # Target is the 'Close' column index inside FEATURE_COLS
    close_idx = selected_features.index(TARGET_COL)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len : i])
        y.append(scaled[i, close_idx])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * train_ratio)
    split = max(split, 1)
    split = min(split, len(X) - 1)
    return X[:split], y[:split], X[split:], y[split:], scaler


# ── Time-weight utilities for training ────────────────────────────────
def get_training_sample_dates(
    df: pd.DataFrame,
    n_train_samples: int,
    seq_len: int = SEQUENCE_LENGTH,
) -> pd.DatetimeIndex:
    """Return the target dates corresponding to each training sample from prepare_sequences."""
    end = min(seq_len + n_train_samples, len(df))
    return pd.to_datetime(df.index[seq_len:end])


def compute_time_weights(
    n_samples: int,
    dates: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """
    Compute per-sample weights that emphasise recent data.

    When *dates* are provided, uses a year-based scheme:
        2024–present → 1.0
        2020–2023    → 0.8
        2015–2019    → 0.5
        2008–2014    → 0.3
        earlier      → 0.2

    Otherwise, falls back to a linear ramp from 0.3 → 1.0.
    """
    if n_samples <= 1:
        return np.ones(n_samples, dtype=np.float32)

    if dates is not None and len(dates) >= n_samples:
        weights = np.ones(n_samples, dtype=np.float32)
        for i in range(n_samples):
            dt = dates[i]
            year = dt.year if hasattr(dt, "year") else 2024
            # Walk TIME_WEIGHT_MAP thresholds (highest first)
            assigned = False
            for thr_year in sorted(TIME_WEIGHT_MAP.keys(), reverse=True):
                if year >= thr_year:
                    weights[i] = TIME_WEIGHT_MAP[thr_year]
                    assigned = True
                    break
            if not assigned:
                weights[i] = 0.2
        return weights

    # Fallback: linear decay from 0.3 (oldest) to 1.0 (newest)
    return np.linspace(0.3, 1.0, n_samples).astype(np.float32)


# ── CSV export helper ─────────────────────────────────────────────────
def predictions_to_csv(dates: List[str], actual: List[float],
                       predicted: List[float]) -> str:
    """Build an in-memory CSV string from prediction results."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Actual", "Predicted"])
    for d, a, p in zip(dates, actual, predicted):
        writer.writerow([d, round(a, 2), round(p, 2)])
    return buf.getvalue()


def _safe_symbol(symbol: str) -> str:
    return (symbol or "").upper().replace("/", "_").replace("\\", "_").strip()


def symbol_data_dir(symbol: str) -> str:
    path = os.path.join(DATA_DIR, _safe_symbol(symbol))
    os.makedirs(path, exist_ok=True)
    return path


def _data_file_path(symbol: str, kind: str) -> str:
    filename_map = {
        "sample": "sample.csv",
        "yahoo": "yahoo.csv",
        "local": "local.csv",
        "tradingview": "tradingview.csv",
    }
    filename = filename_map.get(kind, f"{kind}.csv")
    return os.path.join(symbol_data_dir(symbol), filename)


def _legacy_data_file_path(symbol: str, kind: str) -> str:
    safe_symbol = _safe_symbol(symbol)
    return os.path.join(DATA_DIR, f"{safe_symbol}_{kind}.csv")


def migrate_legacy_data_files() -> Dict[str, object]:
    """
    Move old flat CSV files in data/ into per-symbol folders:
      data/SYMBOL_sample.csv -> data/SYMBOL/sample.csv
      data/SYMBOL_yahoo.csv  -> data/SYMBOL/yahoo.csv
      data/SYMBOL_local.csv  -> data/SYMBOL/local.csv
    """
    suffix_to_kind = {
        "_sample.csv": "sample",
        "_yahoo.csv": "yahoo",
        "_local.csv": "local",
        "_tradingview.csv": "tradingview",
    }

    moved = []
    if not os.path.exists(DATA_DIR):
        return {"moved_count": 0, "moved": moved}

    for name in os.listdir(DATA_DIR):
        src = os.path.join(DATA_DIR, name)
        if not os.path.isfile(src):
            continue

        matched_suffix = None
        for suffix in suffix_to_kind:
            if name.endswith(suffix):
                matched_suffix = suffix
                break
        if not matched_suffix:
            continue

        symbol = name[: -len(matched_suffix)].strip().upper()
        if not symbol:
            continue

        kind = suffix_to_kind[matched_suffix]
        dst = _data_file_path(symbol, kind)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue

        if os.path.exists(dst):
            os.remove(src)
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved.append({"from": src, "to": dst})

    return {"moved_count": len(moved), "moved": moved}


# ── Save sample CSV for testing ───────────────────────────────────────
def save_sample_data(symbol: str = "AAPL"):
    """Download and persist a sample CSV so the app works offline too."""
    path = _data_file_path(symbol, "sample")
    if os.path.exists(path):
        return path
    try:
        df = get_historical_data(symbol, period="2y")
        df.to_csv(path)
        return path
    except Exception:
        return None


def save_downloaded_history_csv(symbol: str, df: pd.DataFrame, source: str = "yahoo_finance") -> str:
    """Persist historical dataframe to CSV by symbol for audit/retrain workflow."""
    suffix = "yahoo" if source == "yahoo_finance" else "local"
    path = _data_file_path(symbol, suffix)
    df.to_csv(path)
    return path


def load_local_sample_csv(symbol: str) -> Optional[pd.DataFrame]:
    """Load local sample CSV if available."""
    path = _data_file_path(symbol, "sample")
    if not os.path.exists(path):
        legacy_path = _legacy_data_file_path(symbol, "sample")
        path = legacy_path if os.path.exists(legacy_path) else path
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df if not df.empty else None


def _symbol_seed(symbol: str) -> int:
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _estimate_start_price(symbol: str) -> float:
    # Deterministic pseudo start-price in a realistic range for fallback data.
    seed = _symbol_seed(symbol)
    return round(20 + (seed % 380), 2)


def generate_symbol_sample_csv(symbol: str, days: int = 520) -> str:
    """Generate deterministic synthetic OHLCV fallback CSV for any symbol."""
    rng = np.random.default_rng(_symbol_seed(symbol))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    start_price = _estimate_start_price(symbol)
    closes = [start_price]
    for _ in range(days - 1):
        drift = 0.0005
        vol = 0.018
        change = rng.normal(drift, vol) * closes[-1]
        closes.append(max(closes[-1] + change, 1.0))

    close = np.array(closes)
    high = close * (1 + np.abs(rng.normal(0, 0.008, days)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, days)))
    opn = low + (high - low) * rng.uniform(0.3, 0.7, days)
    volume = rng.integers(5_000_000, 150_000_000, days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": np.round(opn, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close, 2),
        "Volume": volume,
    }).set_index("Date")

    path = _data_file_path(symbol, "sample")
    df.to_csv(path)
    return path


def get_historical_data_with_fallback(symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, str]:
    """
    Prefer Yahoo Finance data; fallback to local sample CSV if Yahoo fails.
    Returns (dataframe, source).
    """
    try:
        df = get_historical_data(symbol, period=period)
        return df, "yahoo_finance"
    except Exception:
        tv_df = _get_historical_data_from_tv(symbol, period=period)
        if tv_df is not None:
            return tv_df, "tradingview_history"

        local_df = load_local_sample_csv(symbol)
        if local_df is not None:
            return local_df, "local_sample_csv"

        # Last fallback: generate synthetic sample dynamically for unknown symbols.
        generate_symbol_sample_csv(symbol)
        generated_df = load_local_sample_csv(symbol)
        if generated_df is not None:
            return generated_df, "generated_sample_csv"
        raise
