"""
Bulk downloader for open market datasets.

Layout:
- data/market_data/    -> macro-only datasets
- data/stocks/<SYMBOL> -> OHLCV + technical indicators per stock
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "market_data"
STOCKS_OUT = ROOT / "data" / "stocks"
MACRO_YF_OUT = OUT / "macro_yahoo"
FRED_OUT = OUT / "fred"
WB_OUT = OUT / "data360"

# Large, liquid multi-sector universe + extra growth names + international + VN symbols.
STOCK_TICKERS: List[str] = [
    # Mega cap / tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "INTC", "AMD", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MU", "ANET",
    "PANW", "NOW", "SNOW", "DDOG", "PLTR", "SHOP", "UBER", "ABNB", "NFLX", "SPOT",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK", "LLY", "ABBV", "TMO", "DHR", "ABT", "ISRG", "GILD", "BMY",
    # Consumer
    "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "CMCSA", "PEP", "KO", "PG",
    "PM", "MO", "EL", "TGT", "ROST", "TJX",
    # Industrials / materials / energy
    "BA", "CAT", "DE", "UNP", "HON", "GE", "RTX", "LMT", "NOC", "GD", "ETN", "PH",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO",
    # Utilities / REIT / telecom
    "NEE", "DUK", "SO", "AEP", "EXC", "AMT", "PLD", "EQIX", "CCI", "O", "T", "VZ",
    # Growth / AI / EV / biotech
    "ARM", "SMCI", "CRWD", "NET", "MDB", "ZS", "DOCU", "RIVN", "LCID", "NIO", "XPEV", "LI",
    "MRNA", "VRTX", "REGN", "BIIB", "ILMN", "WDAY", "TEAM", "SQ", "COIN",
    # Vietnam examples
    "VIC.VN", "VNM.VN", "HPG.VN", "FPT.VN", "VCB.VN", "SSI.VN", "MBB.VN", "ACB.VN", "MWG.VN", "VHM.VN",
]

EXTRA_MARKET_TICKERS: Dict[str, str] = {
    # Indices
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "RUSSELL2000": "^RUT",
    "NASDAQ100": "^NDX",
    "NYSE_COMP": "^NYA",
    # Volatility
    "VIX": "^VIX",
    "VXN": "^VXN",
    # Rates
    "YIELD_13W": "^IRX",
    "YIELD_5Y": "^FVX",
    "YIELD_10Y": "^TNX",
    "YIELD_30Y": "^TYX",
    # Bonds
    "TLT": "TLT",
    "IEF": "IEF",
    "SHY": "SHY",
    "HYG": "HYG",
    "LQD": "LQD",
    "TIP": "TIP",
    "AGG": "AGG",
    # Sectors
    "XLK": "XLK",
    "XLF": "XLF",
    "XLE": "XLE",
    "XLV": "XLV",
    "XLI": "XLI",
    "XLP": "XLP",
    "XLY": "XLY",
    "XLB": "XLB",
    "XLU": "XLU",
    "XLRE": "XLRE",
    "XLC": "XLC",
    "SOXX": "SOXX",
    "SMH": "SMH",
    # Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL_WTI": "CL=F",
    "OIL_BRENT": "BZ=F",
    "NATGAS": "NG=F",
    "COPPER": "HG=F",
    # FX
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCNY": "USDCNY=X",
    # Crypto
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    # International ETFs
    "EEM": "EEM",
    "EFA": "EFA",
    "FXI": "FXI",
    "EWJ": "EWJ",
    "EWZ": "EWZ",
    "INDA": "INDA",
    "EWG": "EWG",
}

FRED_SERIES: Dict[str, str] = {
    "DFF": "fed_funds_rate",
    "WALCL": "fed_balance_sheet_total_assets",
    "M2SL": "m2_money_stock",
    "GS2": "treasury_2y",
    "GS10": "treasury_10y",
    "TB3MS": "treasury_3m",
    "UNRATE": "unemployment_rate",
    "PAYEMS": "nonfarm_payrolls",
    "CPIAUCSL": "cpi_all_items",
    "PCEPILFE": "core_pce_price_index",
    "BAA10Y": "baa_minus_10y_spread",
    "T10Y2Y": "term_spread_10y_2y",
    "T10Y3M": "term_spread_10y_3m",
    "VIXCLS": "vix_close",
    "BAMLH0A0HYM2": "hy_oad_spread",
    "DEXUSEU": "usd_eur",
    "DEXJPUS": "usd_jpy",
    "DCOILWTICO": "wti_spot",
    "NASDAQCOM": "nasdaq_composite",
    "SP500": "sp500_index",
}

DATA360_INDICATORS: Dict[str, Dict[str, str]] = {
    "WB_GDP_GROWTH": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "USA", "FREQ": "A"},
    "WB_GDP_PER_CAPITA": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_PCAP_CD", "REF_AREA": "USA", "FREQ": "A"},
    "WB_CPI": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL", "REF_AREA": "USA", "FREQ": "A"},
    "WB_INFLATION": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL_ZG", "REF_AREA": "USA", "FREQ": "A"},
    "WB_UNEMPLOYMENT": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_SL_UEM_TOTL_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_LENDING_RATE": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FR_INR_LEND", "REF_AREA": "USA", "FREQ": "A"},
    "WB_M2_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FM_LBL_BMNY_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_TRADE_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NE_TRD_GNFS_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_FDI_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_BX_KLT_DINV_WD_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_GOV_DEBT_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_GC_DOD_TOTL_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_CURR_ACCOUNT_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_BN_CAB_XOKA_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_STOCKS_TRADED_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_CM_MKT_TRAD_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_MCAP_GDP": {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_CM_MKT_LCAP_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
}

DATA360_URL = "https://data360api.worldbank.org/data360/data"


def ensure_dirs() -> None:
    for p in [OUT, STOCKS_OUT, MACRO_YF_OUT, FRED_OUT, WB_OUT]:
        p.mkdir(parents=True, exist_ok=True)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def build_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    req = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in ohlcv.columns for c in req):
        return pd.DataFrame(index=ohlcv.index)

    close = pd.to_numeric(ohlcv["Close"], errors="coerce")
    high = pd.to_numeric(ohlcv["High"], errors="coerce")
    low = pd.to_numeric(ohlcv["Low"], errors="coerce")
    vol = pd.to_numeric(ohlcv["Volume"], errors="coerce").fillna(0)

    ind = pd.DataFrame(index=ohlcv.index)
    ind["Return_1d"] = close.pct_change(fill_method=None) * 100
    ind["Return_5d"] = close.pct_change(5, fill_method=None) * 100
    ind["Return_20d"] = close.pct_change(20, fill_method=None) * 100
    ind["SMA_20"] = close.rolling(20, min_periods=5).mean()
    ind["SMA_50"] = close.rolling(50, min_periods=10).mean()
    ind["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    ind["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    ind["RSI_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["MACD"] = ema12 - ema26
    ind["MACD_Signal"] = ind["MACD"].ewm(span=9, adjust=False).mean()
    ind["MACD_Hist"] = ind["MACD"] - ind["MACD_Signal"]

    bb_mid = close.rolling(20, min_periods=5).mean()
    bb_std = close.rolling(20, min_periods=5).std(ddof=0)
    ind["BB_Upper"] = bb_mid + 2 * bb_std
    ind["BB_Lower"] = bb_mid - 2 * bb_std
    ind["BB_Width"] = (ind["BB_Upper"] - ind["BB_Lower"]) / bb_mid.replace(0, pd.NA)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    ind["ATR_14"] = tr.rolling(14, min_periods=5).mean()

    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    ind["OBV"] = (direction * vol).cumsum()
    ind["ROC_10"] = close.pct_change(10, fill_method=None) * 100
    ind["ROC_20"] = close.pct_change(20, fill_method=None) * 100

    ret = close.pct_change(fill_method=None)
    ind["HistVol_20"] = ret.rolling(20, min_periods=5).std(ddof=0) * (252 ** 0.5) * 100
    ind["HistVol_60"] = ret.rolling(60, min_periods=10).std(ddof=0) * (252 ** 0.5) * 100
    ind["VWAP"] = ((high + low + close) / 3 * vol).cumsum() / vol.cumsum().replace(0, pd.NA)

    return ind


def download_yahoo_symbol(symbol: str, period: str = "max", retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            df = _normalize_index(df)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(0.8 + attempt * 0.6)
    return pd.DataFrame()


def download_many_yahoo_stocks(symbols: Iterable[str]) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for i, sym in enumerate(symbols, start=1):
        df = download_yahoo_symbol(sym, period="max")
        if df.empty:
            rows[sym] = 0
            continue

        stock_dir = STOCKS_OUT / sym
        stock_dir.mkdir(parents=True, exist_ok=True)

        req = ["Open", "High", "Low", "Close", "Volume"]
        ohlcv = df.copy()
        for col in req:
            if col not in ohlcv.columns:
                ohlcv[col] = pd.NA
        ohlcv = ohlcv[req].dropna(how="all")

        indicators = build_technical_indicators(ohlcv)
        features = pd.concat([ohlcv, indicators], axis=1)

        ohlcv.to_csv(stock_dir / "ohlcv.csv")
        indicators.to_csv(stock_dir / "technical_indicators.csv")
        features.to_csv(stock_dir / "features.csv")

        rows[sym] = len(df)
        if i % 10 == 0:
            time.sleep(0.4)
    return rows


def download_extra_markets(ticker_map: Dict[str, str]) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for i, (name, ticker) in enumerate(ticker_map.items(), start=1):
        df = download_yahoo_symbol(ticker, period="max")
        if df.empty:
            rows[name] = 0
            continue
        out = MACRO_YF_OUT / f"{name}.csv"
        df.to_csv(out)
        rows[name] = len(df)
        if i % 10 == 0:
            time.sleep(0.4)
    return rows


def download_fred_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        if len(df.columns) >= 2:
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            return df
    except Exception:
        pass
    return pd.DataFrame()


def download_fred_series(series_map: Dict[str, str]) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for sid, name in series_map.items():
        df = download_fred_csv(sid)
        if df.empty:
            rows[name] = 0
            continue
        df.to_csv(FRED_OUT / f"{name}.csv")
        rows[name] = len(df)
    return rows


def fetch_data360_indicator(cfg: Dict[str, str], from_year: str = "1960") -> pd.DataFrame:
    params = {
        "DATABASE_ID": cfg.get("DATABASE_ID", ""),
        "INDICATOR": cfg.get("INDICATOR", ""),
        "timePeriodFrom": from_year,
        "format": "json",
        "top": 1000,
        "skip": 0,
    }
    if cfg.get("REF_AREA"):
        params["REF_AREA"] = cfg["REF_AREA"]
    if cfg.get("FREQ"):
        params["FREQ"] = cfg["FREQ"]

    dates: List[pd.Timestamp] = []
    vals: List[float] = []
    expected: Optional[int] = None

    while True:
        req = Request(f"{DATA360_URL}?{urlencode(params)}", headers={"User-Agent": "ViziGenesis/2.0"})
        with urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        rows = payload.get("value", []) if isinstance(payload, dict) else []
        if expected is None and isinstance(payload, dict):
            try:
                expected = int(payload.get("count", 0))
            except Exception:
                expected = 0

        if not rows:
            break

        for r in rows:
            raw_t = str(r.get("TIME_PERIOD", "")).strip()
            raw_v = r.get("OBS_VALUE")
            t = pd.to_datetime(raw_t, errors="coerce")
            try:
                v = float(raw_v)
            except Exception:
                continue
            if pd.isna(t):
                continue
            dates.append(t)
            vals.append(v)

        params["skip"] = int(params.get("skip", 0)) + int(params.get("top", 1000))
        if len(rows) < int(params.get("top", 1000)):
            break
        if expected is not None and int(params["skip"]) >= expected:
            break

    if not dates:
        return pd.DataFrame()

    df = pd.DataFrame({"value": vals}, index=pd.DatetimeIndex(dates)).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def download_data360(indicators: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for name, cfg in indicators.items():
        try:
            df = fetch_data360_indicator(cfg)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            rows[name] = 0
            continue
        df.to_csv(WB_OUT / f"{name}.csv")
        rows[name] = len(df)
    return rows


def main() -> None:
    ensure_dirs()

    started = datetime.utcnow().isoformat() + "Z"

    stock_rows = download_many_yahoo_stocks(STOCK_TICKERS)
    market_rows = download_extra_markets(EXTRA_MARKET_TICKERS)
    fred_rows = download_fred_series(FRED_SERIES)
    wb_rows = download_data360(DATA360_INDICATORS)

    summary = {
        "created_at": started,
        "output_root": str(ROOT / "data"),
        "layout": {
            "market_data": str(OUT),
            "stocks": str(STOCKS_OUT),
        },
        "counts": {
            "yahoo_stocks_total": len(STOCK_TICKERS),
            "yahoo_stocks_ok": sum(1 for v in stock_rows.values() if v > 0),
            "yahoo_markets_total": len(EXTRA_MARKET_TICKERS),
            "yahoo_markets_ok": sum(1 for v in market_rows.values() if v > 0),
            "fred_total": len(FRED_SERIES),
            "fred_ok": sum(1 for v in fred_rows.values() if v > 0),
            "data360_total": len(DATA360_INDICATORS),
            "data360_ok": sum(1 for v in wb_rows.values() if v > 0),
        },
        "rows": {
            "yahoo_stocks": stock_rows,
            "yahoo_markets": market_rows,
            "fred": fred_rows,
            "data360": wb_rows,
        },
    }

    with open(ROOT / "data" / "manifest_reorganized.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["counts"], indent=2))
    print(f"Saved datasets under: {ROOT / 'data'}")


if __name__ == "__main__":
    main()
