"""
ViziGenesis V2 — Central Configuration
========================================
All hyper-parameters, feature lists, asset universes, and operational
constants for the institutional-grade quant pipeline.
"""
import os, torch

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
V2_DIR      = os.path.join(MODEL_DIR, "_v2")          # V2 panel artifacts

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(V2_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()       # mixed-precision on GPU

# ═══════════════════════════════════════════════════════════════════════
# Asset universe
# ═══════════════════════════════════════════════════════════════════════
PILOT_TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"]

# Benchmark / market context tickers
BENCHMARK_TICKER   = "^GSPC"              # S&P 500
NASDAQ_TICKER      = "^IXIC"              # NASDAQ Composite
VIX_TICKER         = "^VIX"
BOND_10Y_TICKER    = "^TNX"
DXY_TICKER         = "DX-Y.NYB"

# Sector / cross-asset
SECTOR_TICKERS = {
    "SOXX": "SOXX",       # semiconductor ETF
    "SMH":  "SMH",        # VanEck semiconductor
}
COMMODITY_TICKERS = {
    "Gold": "GC=F",
    "Oil":  "CL=F",
}

# ═══════════════════════════════════════════════════════════════════════
# FRED series IDs  (free — no API key for most)
# ═══════════════════════════════════════════════════════════════════════
FRED_SERIES = {
    # ── Federal Reserve / Monetary Policy ──
    "fed_funds":       "DFF",            # effective federal-funds rate (daily)
    "fed_balance":     "WALCL",          # Fed total assets (weekly)
    "m2":              "M2SL",           # M2 money supply (monthly)

    # ── Treasury yields / term structure ──
    "treasury_2y":     "DGS2",           # 2-year Treasury yield (daily)
    "treasury_10y":    "DGS10",          # 10-year Treasury yield (daily)

    # ── Inflation ──
    "cpi":             "CPIAUCSL",       # CPI urban consumers (monthly)
    "pce":             "PCEPI",          # PCE price index (monthly)
    "core_pce":        "PCEPILFE",       # core PCE ex food & energy

    # ── Employment ──
    "unemployment":    "UNRATE",         # civilian unemployment rate (monthly)
    "nonfarm_payroll": "PAYEMS",         # nonfarm payrolls (monthly)

    # ── Credit & financial conditions ──
    "baa_spread":      "BAAFFM",         # BAA corporate bond spread (monthly)
    "ted_spread":      "TEDRATE",        # TED spread (discontinued → fallback)
}

# FOMC meeting dates (approximate) — updated periodically
FOMC_DATES_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# ═══════════════════════════════════════════════════════════════════════
# Hyper-parameters — Model
# ═══════════════════════════════════════════════════════════════════════
SEQ_LEN         = 60          # configurable lookback window
D_MODEL         = 192         # hidden dimension (shared encoder)
N_HEADS         = 8           # multi-head attention heads
N_LAYERS        = 3           # transformer / RNN layers
DROPOUT         = 0.30        # regularisation
LEARNING_RATE   = 1e-4        # AdamW
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 512         # panel training
MAX_EPOCHS      = 220
PATIENCE        = 10          # early-stop patience
GRAD_CLIP       = 1.0         # gradient clipping norm

# ═══════════════════════════════════════════════════════════════════════
# Hyper-parameters — Training strategy
# ═══════════════════════════════════════════════════════════════════════
TRAIN_PERIOD    = "max"       # fetch maximum history from Yahoo
MIN_ROWS        = 500         # minimum rows per stock to include

# Walk-forward CV
WF_MIN_TRAIN_YEARS = 5
WF_VAL_YEARS       = 1
WF_STEP_YEARS      = 1
WF_N_FOLDS_MAX     = 10       # cap number of folds

# ═══════════════════════════════════════════════════════════════════════
# Loss weights  ( regression + α*classification + β*regime )
# ═══════════════════════════════════════════════════════════════════════
LOSS_W_DIRECTION  = 2.0       # α  —  direction BCE
LOSS_W_RET_1D     = 1.0
LOSS_W_RET_5D     = 0.5
LOSS_W_RET_30D    = 0.3
LOSS_W_EXCESS     = 0.5
LOSS_W_REGIME     = 0.3       # β  —  regime CE

# ═══════════════════════════════════════════════════════════════════════
# Horizons & targets
# ═══════════════════════════════════════════════════════════════════════
HORIZONS = [1, 5, 30]         # forecast horizons (trading days)

# Regime classes
REGIME_CLASSES = ["bull", "bear", "sideways"]
N_REGIMES = len(REGIME_CLASSES)

# ═══════════════════════════════════════════════════════════════════════
# Backtest parameters
# ═══════════════════════════════════════════════════════════════════════
BT_SIGNAL_LONG     = 0.60     # go long when P(up) > this
BT_SIGNAL_SHORT    = 0.60     # go short when P(down) > this
BT_KELLY_CAP       = 0.10     # max Kelly fraction per trade
BT_MAX_EXPOSURE    = 0.25     # max portfolio exposure per stock
BT_INITIAL_CAPITAL = 100_000
BT_COMMISSION_BPS  = 5        # round-trip commission
BT_SLIPPAGE_BPS    = 3        # estimated slippage
BT_STOP_LOSS_PCT   = 0.05     # 5% trailing stop-loss

# Crisis periods for stress tests
CRISIS_PERIODS = {
    "GFC_2008":       ("2007-10-01", "2009-03-31"),
    "COVID_2020":     ("2020-02-01", "2020-06-30"),
    "RATE_HIKE_2022": ("2022-01-01", "2022-12-31"),
}

# ═══════════════════════════════════════════════════════════════════════
# Feature column definitions  (ordered — model sees this exact layout)
# ═══════════════════════════════════════════════════════════════════════

# Group 1: OHLCV (5)
FEAT_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Group 2: Classic technicals (8)
FEAT_TECH_CLASSIC = [
    "MA20", "MA50", "EMA20", "EMA50", "RSI", "MACD",
    "Bollinger_Band", "OBV",
]

# Group 3: Engineered (3)
FEAT_ENGINEERED = ["Volume_Change", "Volatility", "ATR"]

# Group 4: Advanced technicals (9)
FEAT_TECH_ADV = [
    "Stochastic_RSI", "ROC_10", "ROC_20",
    "Momentum_10", "Momentum_20",
    "VWAP", "BB_Width", "Hist_Vol_20", "Hist_Vol_60",
]

# Group 5: Lag returns (6)
FEAT_LAG = [
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    "Return_Lag_5", "Return_Lag_10", "Return_Lag_20",
]

# Group 6: FRED macro (9)
FEAT_FRED = [
    "Fed_Funds_Rate", "Delta_Fed_Funds",
    "Fed_Balance_Change",
    "CPI_YoY", "PCE_YoY",
    "Unemployment",
    "Term_Spread",     # 10y − 2y
    "BAA_Spread",
    "M2_Growth",
]

# Group 7: FOMC / policy events (3)
FEAT_FOMC = [
    "FOMC_Decision_Flag",     # 1 on FOMC day, 0 otherwise
    "FOMC_Rate_Surprise",     # actual − expected (continuous)
    "Policy_Stance",          # −1 dovish, 0 neutral, +1 hawkish
]

# Group 8: Market context (5)
FEAT_MARKET = ["SP500_Ret", "NASDAQ_Ret", "VIX", "BOND_10Y", "DXY"]

# Group 9: Sector / cross-asset (4)
FEAT_SECTOR = ["SOXX_Ret", "SMH_Ret", "Gold_Ret", "Oil_Ret"]

# Group 10: Sentiment (4)
FEAT_SENTIMENT = [
    "News_Sentiment",          # rolling avg headline sentiment
    "Social_Sentiment",        # social media momentum
    "Put_Call_Proxy",          # VIX/realised vol ratio as proxy
    "Fear_Greed_Proxy",        # composite fear/greed
]

# Group 11: Regime proxies (3)
FEAT_REGIME_PROXY = [
    "Realised_Vol_20",         # 20-day realised volatility
    "VIX_Regime",              # z-scored VIX
    "Market_Breadth_Proxy",    # SP500 momentum as breadth proxy
]

# Full V2 feature list (59 features)
V2_FEATURE_COLS = (
    FEAT_OHLCV
    + FEAT_TECH_CLASSIC
    + FEAT_ENGINEERED
    + FEAT_TECH_ADV
    + FEAT_LAG
    + FEAT_FRED
    + FEAT_FOMC
    + FEAT_MARKET
    + FEAT_SECTOR
    + FEAT_SENTIMENT
    + FEAT_REGIME_PROXY
)

N_FEATURES = len(V2_FEATURE_COLS)   # should be 59

# ═══════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════
CALIBRATION_METHODS = ["isotonic", "platt"]
RELIABILITY_BINS = 10

# ═══════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════
SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# Data augmentation
# ═══════════════════════════════════════════════════════════════════════
AUGMENT_NOISE_STD   = 0.002
AUGMENT_SCALE_JITTER = 0.015

# ═══════════════════════════════════════════════════════════════════════
# Time-weighted training (year → weight)
# ═══════════════════════════════════════════════════════════════════════
TIME_WEIGHT_MAP = {
    2024: 1.0,
    2020: 0.8,
    2015: 0.5,
    2008: 0.3,
    0:    0.2,
}
