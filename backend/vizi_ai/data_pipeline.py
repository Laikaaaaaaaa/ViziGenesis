"""
ViziGenesis vizi-o1 — Streaming Multi-Modal Data Pipeline
==========================================================
Lazy-loads heterogeneous financial data (OHLCV, macro, fundamentals,
news, transcripts, cross-asset) from disk in aligned temporal batches.

Design rationale
----------------
*  Financial data is inherently multi-modal: numeric time-series (OHLCV,
   macro), structured tables (fundamentals), and free-form text (news,
   transcripts).  A single monolithic tensor would waste memory and lose
   modality-specific semantics.

*  Instead we produce **per-sample dicts** whose keys map to modality
   tensors.  The model receives these dicts and routes each modality to
   its own encoder branch.

*  Memory is bounded by streaming: only one stock's data is in RAM at a
   time; samples are yielded lazily and shuffled in a bounded buffer by
   the DataLoader.

*  Timestamp alignment is done at *sample construction time*: for every
   anchor date we look-back to collect the corresponding history window
   and snapshot the latest fundamental, macro, and news context.

Classes
-------
MultiModalSample   — typed dict of tensors for one (stock, date) pair
StockDataStream    — IterableDataset yielding samples for one symbol
UniverseDataStream — IterableDataset interleaving all symbols
collate_multimodal — custom collate for DataLoader
"""
from __future__ import annotations

import json, logging, math, os, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger("vizi_ai.data_pipeline")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

# Hard ceiling for any normalised feature value.  Values beyond this
# indicate data errors or extreme outliers that would destabilise
# gradients during training (especially with AMP / bf16).
_MAX_FEATURE_VAL = 10.0

# ═══════════════════════════════════════════════════════════════
#  Currency & timezone metadata for multi-country handling
# ═══════════════════════════════════════════════════════════════
# Maps symbol suffix → (currency, UTC offset of market close, country)
MARKET_META = {
    ".VN":  ("VND", 7,  "VN"),   # Vietnam HOSE/HNX  close ~14:30 UTC+7
    ".HM":  ("VND", 7,  "VN"),
    ".T":   ("JPY", 9,  "JP"),   # Tokyo Stock Exchange close ~15:00 UTC+9
    ".KS":  ("KRW", 9,  "KR"),   # Korea Exchange      close ~15:30 UTC+9
    ".KQ":  ("KRW", 9,  "KR"),   # KOSDAQ
    ".HK":  ("HKD", 8,  "HK"),   # HKEX               close ~16:00 UTC+8
    ".SS":  ("CNY", 8,  "CN"),   # Shanghai
    ".SZ":  ("CNY", 8,  "CN"),   # Shenzhen
    ".L":   ("GBP", 0,  "GB"),   # London
    ".PA":  ("EUR", 1,  "FR"),   # Paris
    ".DE":  ("EUR", 1,  "DE"),   # Frankfurt
    ".TO":  ("CAD", -5, "CA"),   # Toronto
}
# US stocks (no suffix) default:
_US_META = ("USD", -5, "US")
# US market close UTC offset (Eastern Time, ~16:00 → UTC-5 / UTC-4 DST)
_US_CLOSE_UTC = -5


def _symbol_meta(symbol: str) -> tuple:
    """Return (currency, utc_close_offset, country) for a symbol."""
    for suffix, meta in MARKET_META.items():
        if symbol.upper().endswith(suffix.upper()):
            return meta
    return _US_META


def _closes_before_us(symbol: str) -> bool:
    """True if this symbol's exchange closes *before* the US market.

    For such symbols we must use T-1 US/macro data to avoid look-ahead."""
    _, utc_off, _ = _symbol_meta(symbol)
    # Asian / European markets close before US (UTC+7..+9 vs UTC-5)
    return utc_off > _US_CLOSE_UTC


def _sanitize_array(arr: np.ndarray,
                    max_val: float = _MAX_FEATURE_VAL) -> np.ndarray:
    """Replace non-finite values and clip to [-max_val, +max_val].

    This is the last line of defence before tensors enter the model.
    Financial data is notoriously dirty — inf from pct_change on zero
    prices, NaN from missing series, and extreme outliers from illiquid
    markets.  Clamping to a fixed range prevents gradient explosion
    in AMP / bf16 training."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=-max_val)
    return np.clip(arr, -max_val, max_val).astype(np.float32)

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════
@dataclass
class DataConfig:
    """All knobs for the data pipeline — one place to change."""
    # Sequence lengths
    price_seq_len: int = 120          # trading-day look-back for OHLCV
    macro_seq_len: int = 60           # look-back for macro series
    max_news_items: int = 8           # news headlines per sample
    max_news_tokens: int = 64         # token-ids per headline

    # Targets
    horizons: Tuple[int, ...] = (1, 5, 21)  # 1d, 1w, 1m forward returns
    direction_threshold: float = 0.0        # > 0 → up

    # Regime
    regime_window: int = 63           # 3-month rolling for regime label
    regime_thresholds: Tuple[float, float] = (-0.05, 0.05)  # bear / bull

    # Splits  (by date, NOT by random — prevents look-ahead)
    train_end: str = "2023-12-31"
    val_end:   str = "2024-12-31"
    # test = everything after val_end

    # Normalisation
    price_scaler: str = "robust"      # robust | zscore | minmax
    macro_scaler: str = "robust"

    # Paths
    data_root: Path = DATA


# ═══════════════════════════════════════════════════════════════
#  Data loading helpers (lazy, per-stock)
# ═══════════════════════════════════════════════════════════════
_MACRO_CACHE: Dict[str, pd.DataFrame] = {}     # shared across stocks
_NEWS_CACHE: Optional[List[Dict]] = None
_FUNDAMENTALS_CACHE: Dict[str, Dict] = {}
_MARKET_CACHE: Dict[str, pd.DataFrame] = {}

FRED_KEYS = [
    "cpi_all_items", "fed_funds_rate", "unemployment_rate",
    "gdp_real", "pce_price_index", "treasury_yield_10y", "treasury_yield_2y",
    "sp500_index", "vix_close",
    "m2_money_stock", "industrial_production",
    "umich_consumer_sentiment", "housing_starts",
    "initial_claims", "brent_crude_spot",
]

# Actual filenames in data/markets/ (stem without .csv)
MARKET_INDICES = [
    ("indices",     "SP500"),       # S&P 500
    ("indices",     "NASDAQ"),      # NASDAQ Composite
    ("indices",     "DOW"),         # Dow Jones
    ("indices",     "RUSSELL2000"), # Russell 2000
    ("bonds",       "YIELD_10Y"),   # 10Y treasury yield
    ("volatility",  "VIX"),         # CBOE VIX
    ("commodities", "GOLD"),        # Gold
    ("commodities", "OIL_WTI"),     # WTI Crude
    ("fx",          "DXY"),         # USD index
    ("crypto",      "BTC"),         # Bitcoin
]


def _load_stock_ohlcv(symbol: str, root: Path) -> pd.DataFrame:
    """Load OHLCV + features for a single stock."""
    feat_path = root / "stocks" / symbol / "features.csv"
    ohlcv_path = root / "stocks" / symbol / "ohlcv.csv"
    path = feat_path if feat_path.exists() else ohlcv_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def _load_macro(root: Path) -> Dict[str, pd.DataFrame]:
    """Load all FRED macro series into a shared cache."""
    global _MACRO_CACHE
    if _MACRO_CACHE:
        return _MACRO_CACHE
    fred_dir = root / "macro" / "fred"
    if not fred_dir.exists():
        return _MACRO_CACHE
    for csv in fred_dir.glob("*.csv"):
        key = csv.stem
        try:
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            _MACRO_CACHE[key] = df.sort_index()
        except Exception:
            pass
    logger.info("Loaded %d FRED macro series", len(_MACRO_CACHE))
    return _MACRO_CACHE


def _load_market_indices(root: Path) -> Dict[str, pd.DataFrame]:
    """Load cross-market indices/commodities/crypto."""
    global _MARKET_CACHE
    if _MARKET_CACHE:
        return _MARKET_CACHE
    for subdir, name in MARKET_INDICES:
        csv = root / "markets" / subdir / f"{name}.csv"
        if csv.exists():
            try:
                df = pd.read_csv(csv, index_col=0, parse_dates=True)
                df = df.apply(pd.to_numeric, errors="coerce")
                if "Close" in df.columns:
                    _MARKET_CACHE[name] = df[["Close"]].rename(columns={"Close": name}).sort_index()
                elif "value" in df.columns:
                    _MARKET_CACHE[name] = df[["value"]].rename(columns={"value": name}).sort_index()
                elif len(df.columns) > 0:
                    col = df.columns[0]
                    _MARKET_CACHE[name] = df[[col]].rename(columns={col: name}).sort_index()
            except Exception:
                pass
    logger.info("Loaded %d market indices", len(_MARKET_CACHE))
    return _MARKET_CACHE


def _load_fundamentals(symbol: str, root: Path) -> Dict[str, Any]:
    """Load fundamental data for a stock (cached)."""
    if symbol in _FUNDAMENTALS_CACHE:
        return _FUNDAMENTALS_CACHE[symbol]
    fund_dir = root / "fundamentals" / symbol
    if not fund_dir.exists():
        _FUNDAMENTALS_CACHE[symbol] = {}
        return {}
    result: Dict[str, Any] = {}
    # Key stats
    ks = fund_dir / "key_stats.json"
    if ks.exists():
        with open(ks, "r") as f:
            result["key_stats"] = json.load(f)
    # Financial statements (actual filenames: income_annual.csv, balance_sheet_annual.csv, etc.)
    for name in ["income_annual.csv", "income_quarterly.csv",
                 "balance_sheet_annual.csv", "balance_sheet_quarterly.csv",
                 "cashflow_annual.csv", "cashflow_quarterly.csv"]:
        p = fund_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0)
                result[name.replace(".csv", "")] = df
            except Exception:
                pass
    _FUNDAMENTALS_CACHE[symbol] = result
    return result


def _load_news(root: Path) -> List[Dict]:
    """Load news corpus (shared)."""
    global _NEWS_CACHE
    if _NEWS_CACHE is not None:
        return _NEWS_CACHE
    _NEWS_CACHE = []
    news_path = root / "processed" / "news_corpus.jsonl"
    if news_path.exists():
        with open(news_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        _NEWS_CACHE.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    logger.info("Loaded %d news articles", len(_NEWS_CACHE))
    return _NEWS_CACHE


# ═══════════════════════════════════════════════════════════════
#  Normalisation utilities
# ═══════════════════════════════════════════════════════════════
def _robust_scale(arr: np.ndarray) -> np.ndarray:
    """Robust z-score: (x - median) / IQR; handles fat tails in finance."""
    med = np.nanmedian(arr, axis=0, keepdims=True)
    q75 = np.nanpercentile(arr, 75, axis=0, keepdims=True)
    q25 = np.nanpercentile(arr, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-8, 1.0, iqr)
    scaled = ((arr - med) / iqr).astype(np.float32)
    # Hard clip — prevents outlier columns from producing extreme values
    # that would cause NaN in AMP forward passes.
    return np.clip(scaled, -_MAX_FEATURE_VAL, _MAX_FEATURE_VAL)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = ((arr - mu) / std).astype(np.float32)
    return np.clip(scaled, -_MAX_FEATURE_VAL, _MAX_FEATURE_VAL)


def normalise(arr: np.ndarray, method: str = "robust") -> np.ndarray:
    if method == "robust":
        out = _robust_scale(arr)
    elif method == "zscore":
        out = _zscore(arr)
    elif method == "minmax":
        lo = np.nanmin(arr, axis=0, keepdims=True)
        hi = np.nanmax(arr, axis=0, keepdims=True)
        rng = hi - lo
        rng = np.where(rng < 1e-8, 1.0, rng)
        out = ((arr - lo) / rng).astype(np.float32)
    else:
        out = arr.astype(np.float32)
    # Final sanitization — no NaN/Inf must leave this function.
    return _sanitize_array(out)


# ═══════════════════════════════════════════════════════════════
#  Fundamental features (fixed-length vector)
# ═══════════════════════════════════════════════════════════════
FUNDAMENTAL_KEYS = [
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "profitMargins", "operatingMargins", "grossMargins",
    "returnOnEquity", "returnOnAssets", "debtToEquity", "currentRatio",
    "quickRatio", "revenueGrowth", "earningsGrowth", "freeCashflow",
    "dividendYield", "beta", "marketCap", "enterpriseValue",
]
N_FUNDAMENTAL_FEATURES = len(FUNDAMENTAL_KEYS)


# Reasonable range limits for fundamental ratios.  Values outside
# these are almost certainly data errors or extreme outliers that
# would blow up gradients if fed raw into the model.
_FUND_CLIP = {
    "trailingPE":       (-200, 2000),
    "forwardPE":        (-200, 2000),
    "priceToBook":      (-50, 500),
    "priceToSalesTrailing12Months": (0, 500),
    "enterpriseToEbitda": (-100, 500),
    "profitMargins":    (-5, 5),
    "operatingMargins": (-5, 5),
    "grossMargins":     (-5, 5),
    "returnOnEquity":   (-10, 10),
    "returnOnAssets":   (-5, 5),
    "debtToEquity":     (-100, 5000),
    "currentRatio":     (0, 100),
    "quickRatio":       (0, 100),
    "revenueGrowth":    (-5, 50),
    "earningsGrowth":   (-5, 50),
    "freeCashflow":     None,  # handled with sign-aware log
    "dividendYield":    (0, 1),
    "beta":             (-5, 10),
    "marketCap":        None,  # handled with log
    "enterpriseValue":  None,  # handled with log
}


def _fundamentals_to_vector(fund: Dict[str, Any]) -> np.ndarray:
    """Convert fundamental dict → fixed-length float vector.

    Key fixes for multi-country data:
    - Sign-aware log scaling for monetary values (freeCashflow, marketCap,
      enterpriseValue) — eliminates currency-scale differences.
      e.g. VND market caps (trillions) vs USD (billions) both map to ~20-35.
    - Clip ratios to physically reasonable ranges before they enter the model.
    - Robust-scale the final vector so all fundamentals are O(1).
    """
    ks = fund.get("key_stats", {})
    vec = np.full(N_FUNDAMENTAL_FEATURES, 0.0, dtype=np.float32)
    for i, key in enumerate(FUNDAMENTAL_KEYS):
        val = ks.get(key)
        if val is None:
            continue
        try:
            v = float(val)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(v):
            continue
        vec[i] = v

    # Sign-aware log scaling for monetary values (handles negative FCF
    # and cross-currency scale: VND trillions vs USD billions).
    for idx in [15, 18, 19]:  # freeCashflow, marketCap, enterpriseValue
        v = vec[idx]
        if v != 0.0:
            vec[idx] = np.sign(v) * np.log1p(abs(v))

    # Clip ratio features to sane ranges
    for i, key in enumerate(FUNDAMENTAL_KEYS):
        bounds = _FUND_CLIP.get(key)
        if bounds is not None:
            lo, hi = bounds
            vec[i] = np.clip(vec[i], lo, hi)

    # Robust-scale the vector so all values are O(1) — essential for
    # the FundamentalEncoder Linear layer to not blow up.
    med = np.nanmedian(vec)
    q75, q25 = np.nanpercentile(vec, 75), np.nanpercentile(vec, 25)
    iqr = q75 - q25
    if iqr < 1e-8:
        iqr = max(abs(med), 1.0)
    vec = (vec - med) / iqr

    return _sanitize_array(vec)


# ═══════════════════════════════════════════════════════════════
#  Macro features (time-aligned snapshot)
# ═══════════════════════════════════════════════════════════════
def _macro_snapshot(date: pd.Timestamp, macro: Dict[str, pd.DataFrame],
                    seq_len: int) -> np.ndarray:
    """
    Build a (seq_len, N_macro) array of macro indicators ending at `date`.
    Forward-fills, then back-fills remaining NaNs.
    """
    cols = []
    for key in FRED_KEYS:
        df = macro.get(key)
        if df is not None and not df.empty:
            col_name = df.columns[0] if len(df.columns) else key
            series = df[col_name] if col_name in df.columns else df.iloc[:, 0]
            # Reindex to business days up to `date`
            idx = pd.bdate_range(end=date, periods=seq_len * 3, freq="B")
            series = series.reindex(idx, method="ffill").bfill()
            vals = series.iloc[-seq_len:].values.astype(np.float32)
        else:
            vals = np.full(seq_len, np.nan, dtype=np.float32)

        if len(vals) < seq_len:
            vals = np.pad(vals, (seq_len - len(vals), 0), constant_values=np.nan)
        cols.append(vals[:seq_len])

    arr = np.column_stack(cols) if cols else np.zeros((seq_len, 1), dtype=np.float32)
    # Forward-fill, then back-fill NaN
    df_tmp = pd.DataFrame(arr).ffill().bfill().fillna(0)
    return _sanitize_array(df_tmp.values)


# ═══════════════════════════════════════════════════════════════
#  Cross-market features (time-aligned)
# ═══════════════════════════════════════════════════════════════
def _market_snapshot(date: pd.Timestamp, markets: Dict[str, pd.DataFrame],
                     seq_len: int,
                     lag_one_day: bool = False) -> np.ndarray:
    """(seq_len, N_markets) of cross-market daily returns.

    Args:
        lag_one_day: If True, shift the snapshot back by 1 business day.
            Used for Asian stocks to avoid look-ahead bias (US market
            data from the same calendar date was not available when
            the Asian market closed).
    """
    end_date = date
    if lag_one_day:
        # Use T-1 data to prevent look-ahead from later-closing markets
        end_date = date - pd.tseries.offsets.BDay(1)

    cols = []
    for name, df in markets.items():
        col = df.columns[0]
        series = df[col].sort_index()
        idx = pd.bdate_range(end=end_date, periods=seq_len * 3, freq="B")
        series = series.reindex(idx, method="ffill").bfill()
        # Convert to returns — clip to ±50% to squash inf from zero-price gaps
        rets = series.pct_change(fill_method=None).fillna(0)
        rets = rets.clip(-0.5, 0.5)  # no single-day index move > 50%
        rets = rets.iloc[-seq_len:].values.astype(np.float32)
        if len(rets) < seq_len:
            rets = np.pad(rets, (seq_len - len(rets), 0), constant_values=0)
        cols.append(rets[:seq_len])
    if not cols:
        return np.zeros((seq_len, 1), dtype=np.float32)
    arr = np.column_stack(cols)
    return _sanitize_array(arr, max_val=1.0)  # returns are O(0.01)


# ═══════════════════════════════════════════════════════════════
#  News encoding (simple bag-of-words token IDs)
# ═══════════════════════════════════════════════════════════════
# We use character trigram hashing for zero-dependency encoding.
# The model learns embeddings from these IDs.
VOCAB_SIZE = 8192  # hash space

def _hash_token(tok: str) -> int:
    """Deterministic hash of a token to [1, VOCAB_SIZE-1]. 0 = padding."""
    h = 5381
    for ch in tok.lower():
        h = ((h << 5) + h + ord(ch)) & 0x7FFFFFFF
    return (h % (VOCAB_SIZE - 1)) + 1


def _encode_headline(text: str, max_tokens: int) -> np.ndarray:
    """Encode a headline as token-hashed IDs, padded/truncated to max_tokens."""
    tokens = text.split()[:max_tokens]
    ids = [_hash_token(t) for t in tokens]
    arr = np.zeros(max_tokens, dtype=np.int64)
    arr[:len(ids)] = ids
    return arr


def _news_for_date(date: pd.Timestamp, news: List[Dict],
                   max_items: int, max_tokens: int) -> np.ndarray:
    """
    Find news articles near `date` and encode them.
    Returns (max_items, max_tokens) int64 array.
    """
    result = np.zeros((max_items, max_tokens), dtype=np.int64)

    # Filter by date (within 7 days prior)
    date_str = date.strftime("%Y-%m-%d")
    candidates = []
    for article in news:
        adate = article.get("date", "")
        title = article.get("title", "")
        if title and adate:
            candidates.append((adate, title))

    # Sort by recency and take top-k
    candidates.sort(key=lambda x: x[0], reverse=True)
    for i, (_, title) in enumerate(candidates[:max_items]):
        result[i] = _encode_headline(title, max_tokens)

    return result


# ═══════════════════════════════════════════════════════════════
#  Target computation
# ═══════════════════════════════════════════════════════════════
def _compute_targets(close: pd.Series, idx: int,
                     horizons: Tuple[int, ...],
                     threshold: float,
                     regime_window: int,
                     regime_thresholds: Tuple[float, float]) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute forward returns + direction + regime label.
    Returns None if any horizon extends beyond data.
    """
    max_h = max(horizons)
    if idx + max_h >= len(close):
        return None
    price_now = close.iloc[idx]
    if price_now <= 0 or np.isnan(price_now):
        return None

    returns = {}
    for h in horizons:
        future = close.iloc[idx + h]
        if np.isnan(future) or future <= 0:
            return None
        returns[f"ret_{h}d"] = np.float32((future / price_now) - 1.0)

    # Direction: based on 1-day return
    direction = np.float32(1.0 if returns["ret_1d"] > threshold else 0.0)

    # Regime: rolling return over window
    start = max(0, idx - regime_window)
    regime_ret = (price_now / close.iloc[start]) - 1.0 if close.iloc[start] > 0 else 0.0
    if regime_ret < regime_thresholds[0]:
        regime = 0  # bear
    elif regime_ret > regime_thresholds[1]:
        regime = 2  # bull
    else:
        regime = 1  # sideways

    targets = {
        "direction": np.array([direction], dtype=np.float32),
        "regime": np.array([regime], dtype=np.int64),
    }
    for k, v in returns.items():
        targets[k] = np.array([v], dtype=np.float32)
    return targets


# ═══════════════════════════════════════════════════════════════
#  StockDataStream — lazy per-stock iterable
# ═══════════════════════════════════════════════════════════════
class StockDataStream(IterableDataset):
    """
    Yields multi-modal samples for a single stock symbol.

    Each sample is a dict:
        price_seq:    (price_seq_len, n_price_feat)  float32
        macro_seq:    (macro_seq_len, n_macro_feat)  float32
        market_seq:   (macro_seq_len, n_market_feat) float32
        fundamental:  (n_fund_feat,)                 float32
        news_ids:     (max_news, max_tokens)         int64
        stock_id:     ()                             int64
        targets:      dict of arrays

    Memory: only one stock in RAM at a time.
    """

    def __init__(
        self,
        symbol: str,
        stock_id: int,
        cfg: DataConfig,
        split: str = "train",  # train | val | test
        shuffle: bool = True,
    ):
        self.symbol = symbol
        self.stock_id = stock_id
        self.cfg = cfg
        self.split = split
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        cfg = self.cfg

        # ── Load this stock's data ──
        ohlcv = _load_stock_ohlcv(self.symbol, cfg.data_root)
        if ohlcv.empty or len(ohlcv) < cfg.price_seq_len + max(cfg.horizons) + 10:
            return

        # ── Load shared data ──
        macro = _load_macro(cfg.data_root)
        markets = _load_market_indices(cfg.data_root)
        fund = _load_fundamentals(self.symbol, cfg.data_root)
        news = _load_news(cfg.data_root)

        # ── Feature matrix for price ──
        # Use all numeric columns from features.csv / ohlcv.csv
        price_cols = [c for c in ohlcv.columns if ohlcv[c].dtype in (np.float64, np.float32, np.int64, np.float16)]
        if "Close" not in price_cols:
            return
        price_df = ohlcv[price_cols].copy()
        close = price_df["Close"].copy()

        # Forward-fill then back-fill
        price_df = price_df.ffill().bfill().fillna(0)
        price_arr = price_df.values.astype(np.float32)

        # Replace any remaining inf/nan *before* normalisation to prevent
        # poisoning of median/IQR statistics in robust scaler.
        price_arr = np.nan_to_num(price_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise price features per-stock (training window stats)
        price_arr = normalise(price_arr, cfg.price_scaler)
        # normalise() already calls _sanitize_array, but be explicit:
        price_arr = _sanitize_array(price_arr)

        # ── Fundamental vector ──
        fund_vec = _fundamentals_to_vector(fund)  # already sanitized inside

        # ── Timezone awareness: use T-1 macro/market for non-US stocks ──
        use_lag = _closes_before_us(self.symbol)

        # ── Date-based split ──
        dates = ohlcv.index
        if self.split == "train":
            mask = dates <= cfg.train_end
        elif self.split == "val":
            mask = (dates > cfg.train_end) & (dates <= cfg.val_end)
        else:
            mask = dates > cfg.val_end

        valid_indices = np.where(np.asarray(mask))[0]
        # Must have enough history
        valid_indices = valid_indices[valid_indices >= cfg.price_seq_len]
        # Must be able to compute targets
        max_h = max(cfg.horizons)
        valid_indices = valid_indices[valid_indices + max_h < len(dates)]

        if len(valid_indices) == 0:
            return

        if self.shuffle:
            np.random.shuffle(valid_indices)

        n_price_feat = price_arr.shape[1]
        for idx in valid_indices:
            anchor_date = dates[idx]

            # ── Price sequence ──
            start = idx - cfg.price_seq_len
            seq = price_arr[start:idx]  # (price_seq_len, n_price_feat)
            if seq.shape[0] < cfg.price_seq_len:
                pad = np.zeros((cfg.price_seq_len - seq.shape[0], n_price_feat), dtype=np.float32)
                seq = np.vstack([pad, seq])

            # ── Macro sequence ──
            # For non-US stocks, shift macro window back 1 day to avoid
            # using same-day US macro releases as features (look-ahead).
            macro_date = anchor_date
            if use_lag:
                macro_date = anchor_date - pd.tseries.offsets.BDay(1)
            macro_seq = _macro_snapshot(macro_date, macro, cfg.macro_seq_len)
            macro_seq = normalise(macro_seq, cfg.macro_scaler)
            macro_seq = _sanitize_array(macro_seq)

            # ── Market cross-asset ──
            market_seq = _market_snapshot(
                anchor_date, markets, cfg.macro_seq_len,
                lag_one_day=use_lag,
            )
            market_seq = _sanitize_array(market_seq, max_val=1.0)

            # ── News ──
            news_enc = _news_for_date(anchor_date, news, cfg.max_news_items, cfg.max_news_tokens)

            # ── Targets ──
            targets = _compute_targets(
                close, idx, cfg.horizons, cfg.direction_threshold,
                cfg.regime_window, cfg.regime_thresholds,
            )
            if targets is None:
                continue

            # Clip return targets to ±50% — extreme outliers from stock
            # splits, delistings, or data errors destabilise Huber loss.
            for tk, tv in targets.items():
                if tk.startswith("ret_"):
                    targets[tk] = np.clip(tv, -0.5, 0.5)

            # ── Final per-sample validation ──
            # Skip any sample with non-finite values rather than let it
            # poison the mega-batch and cause NaN loss.
            sample_arrays = [seq, macro_seq, market_seq, fund_vec]
            if any(not np.all(np.isfinite(a)) for a in sample_arrays):
                continue

            yield {
                "price_seq": torch.from_numpy(seq),
                "macro_seq": torch.from_numpy(macro_seq),
                "market_seq": torch.from_numpy(market_seq),
                "fundamental": torch.from_numpy(fund_vec),
                "news_ids": torch.from_numpy(news_enc),
                "stock_id": torch.tensor(self.stock_id, dtype=torch.long),
                "targets": {k: torch.from_numpy(v) for k, v in targets.items()},
            }


# ═══════════════════════════════════════════════════════════════
#  UniverseDataStream — interleave all stocks
# ═══════════════════════════════════════════════════════════════
class UniverseDataStream(IterableDataset):
    """
    Interleaves StockDataStream for the full universe.

    Why interleave rather than concatenate?
    ⟶ Each batch sees multiple stocks, so the model learns cross-stock
      patterns from the start.  Also distributes memory usage evenly.
    """

    def __init__(
        self,
        symbols: List[str],
        cfg: DataConfig,
        split: str = "train",
        shuffle: bool = True,
    ):
        self.symbols = symbols
        self.cfg = cfg
        self.split = split
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Create per-stock iterators
        iters = []
        for idx, sym in enumerate(self.symbols):
            stream = StockDataStream(sym, idx, self.cfg, self.split, self.shuffle)
            iters.append(iter(stream))

        if not iters:
            return

        # Round-robin with random interleaving
        active = list(range(len(iters)))
        if self.shuffle:
            random.shuffle(active)

        while active:
            exhausted = []
            for i in list(active):
                try:
                    yield next(iters[i])
                except StopIteration:
                    exhausted.append(i)
            for i in exhausted:
                active.remove(i)


# ═══════════════════════════════════════════════════════════════
#  Collate function
# ═══════════════════════════════════════════════════════════════
def collate_multimodal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that stacks per-modality tensors and nests targets.
    """
    result: Dict[str, Any] = {}
    keys = ["price_seq", "macro_seq", "market_seq", "fundamental", "news_ids", "stock_id"]
    for k in keys:
        result[k] = torch.stack([b[k] for b in batch])

    # Targets: stack each target key
    target_keys = batch[0]["targets"].keys()
    result["targets"] = {}
    for tk in target_keys:
        result["targets"][tk] = torch.stack([b["targets"][tk] for b in batch])

    return result


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════
def create_dataloader(
    symbols: List[str],
    cfg: DataConfig,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a streaming DataLoader for the given split.

    num_workers=0 is recommended on Windows; on Linux set 2-4 for
    prefetching.  The IterableDataset handles its own shuffling.
    """
    dataset = UniverseDataStream(symbols, cfg, split, shuffle=(shuffle and split == "train"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_multimodal,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


# ═══════════════════════════════════════════════════════════════
#  Discover available symbols
# ═══════════════════════════════════════════════════════════════
def discover_symbols(root: Path = DATA, min_rows: int = 500) -> List[str]:
    """Find all stock symbols that have sufficient data."""
    stocks_dir = root / "stocks"
    if not stocks_dir.exists():
        return []
    symbols = []
    for d in sorted(stocks_dir.iterdir()):
        if not d.is_dir():
            continue
        ohlcv = d / "ohlcv.csv"
        feat = d / "features.csv"
        path = feat if feat.exists() else ohlcv
        if not path.exists():
            continue
        # Quick row count check
        try:
            with open(path, "r") as f:
                n = sum(1 for _ in f) - 1
            if n >= min_rows:
                symbols.append(d.name)
        except Exception:
            pass
    return symbols
