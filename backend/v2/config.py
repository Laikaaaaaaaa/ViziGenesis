"""
ViziGenesis V2 — Central Configuration
========================================
All hyper-parameters, feature lists, asset universes, and operational
constants for the institutional-grade quant pipeline.

Optimized for RTX 4090 24 GB — large model, deep training, 150+ features,
multi-million row datasets from 60+ open market data sources.
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
# Asset universe — expanded to 25 diverse, liquid US stocks
# ═══════════════════════════════════════════════════════════════════════
PILOT_TICKERS = [
    # Mega-cap Tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    # Semiconductors
    "AMD", "INTC", "AVGO",
    # Finance
    "JPM", "GS", "BAC",
    # Healthcare
    "UNH", "JNJ", "PFE",
    # Energy
    "XOM", "CVX",
    # Consumer
    "WMT", "HD",
    # Industrials
    "CAT", "BA",
    # Diversified
    "BRK-B", "DIS", "NFLX",
]

# Benchmark / market context tickers
BENCHMARK_TICKER   = "^GSPC"              # S&P 500
NASDAQ_TICKER      = "^IXIC"              # NASDAQ Composite
VIX_TICKER         = "^VIX"
BOND_10Y_TICKER    = "^TNX"
DXY_TICKER         = "DX-Y.NYB"

# Macro feature keys (backward-compatible identity mapping)
MACRO_SERIES = {
    "fed_funds":       "fed_funds",
    "fed_balance":     "fed_balance",
    "m2":              "m2",
    "treasury_2y":     "treasury_2y",
    "treasury_10y":    "treasury_10y",
    "cpi":             "cpi",
    "pce":             "pce",
    "core_pce":        "core_pce",
    "unemployment":    "unemployment",
    "nonfarm_payroll": "nonfarm_payroll",
    "baa_spread":      "baa_spread",
    "ted_spread":      "ted_spread",
}
FRED_SERIES = MACRO_SERIES  # backward compatibility alias

# ═══════════════════════════════════════════════════════════════════════
# Hyper-parameters — Model (RTX 4090 24 GB optimized)
# ═══════════════════════════════════════════════════════════════════════
SEQ_LEN         = 120         # 6-month lookback (more context)
D_MODEL         = 384         # large hidden dimension
N_HEADS         = 12          # attention heads (384 / 12 = 32 per head)
N_LAYERS        = 6           # deep transformer / RNN stack
DROPOUT         = 0.25        # regularisation (slightly less for bigger model)
LEARNING_RATE   = 5e-5        # lower LR for larger model
WEIGHT_DECAY    = 1e-4        # stronger weight decay
BATCH_SIZE      = 768         # RTX 4090 can handle this with AMP
MAX_EPOCHS      = 800         # deep training — RTX 4090 handles this
PATIENCE        = 25          # more patience for deeper convergence
GRAD_CLIP       = 1.0         # gradient clipping norm
GRAD_ACCUM_STEPS = 2          # effective batch = 768 × 2 = 1536

# ═══════════════════════════════════════════════════════════════════════
# Hyper-parameters — Training strategy
# ═══════════════════════════════════════════════════════════════════════
TRAIN_PERIOD    = "max"       # fetch maximum history from Yahoo
MIN_ROWS        = 500         # minimum rows per stock to include

# Walk-forward CV
WF_MIN_TRAIN_YEARS = 5
WF_VAL_YEARS       = 1
WF_STEP_YEARS      = 1
WF_N_FOLDS_MAX     = 15       # more folds for better evaluation

# ═══════════════════════════════════════════════════════════════════════
# Loss weights  ( regression + α*classification + β*regime )
# ═══════════════════════════════════════════════════════════════════════
LOSS_W_DIRECTION  = 2.5       # α  —  direction BCE (primary objective)
LOSS_W_RET_1D     = 1.0
LOSS_W_RET_5D     = 0.8
LOSS_W_RET_30D    = 0.5
LOSS_W_EXCESS     = 0.7
LOSS_W_REGIME     = 0.5       # β  —  regime CE (important for context)

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
    "BANKING_2023":   ("2023-03-01", "2023-05-31"),
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

# Group 6: Macro economic (9) — legacy column names preserved
FEAT_MACRO = [
    "Fed_Funds_Rate", "Delta_Fed_Funds",
    "Fed_Balance_Change",
    "CPI_YoY", "PCE_YoY",
    "Unemployment",
    "Term_Spread",     # 10y − 3m
    "BAA_Spread",
    "M2_Growth",
]
FEAT_FRED = FEAT_MACRO  # backward compatibility alias

# Group 7: FOMC / policy events (3)
FEAT_FOMC = [
    "FOMC_Decision_Flag",
    "FOMC_Rate_Surprise",
    "Policy_Stance",
]

# Group 8: Market context (5)
FEAT_MARKET = ["SP500_Ret", "NASDAQ_Ret", "VIX", "BOND_10Y", "DXY"]

# Group 9: Sector / cross-asset legacy (4) — kept for compatibility
FEAT_SECTOR = ["SOXX_Ret", "SMH_Ret", "Gold_Ret", "Oil_Ret"]

# Group 10: Sentiment (4)
FEAT_SENTIMENT = [
    "News_Sentiment",
    "Social_Sentiment",
    "Put_Call_Proxy",
    "Fear_Greed_Proxy",
]

# Group 11: Regime proxies (3)
FEAT_REGIME_PROXY = [
    "Realised_Vol_20",
    "VIX_Regime",
    "Market_Breadth_Proxy",
]

# ── NEW Group 12: Yield Curve & Interest Rate Structure (8) ──────────
FEAT_YIELD_CURVE = [
    "Yield_10Y_3M_Spread",
    "Yield_10Y_5Y_Spread",
    "Yield_30Y_10Y_Spread",
    "Yield_Curve_Curvature",
    "Yield_Level",
    "Yield_10Y_Change_5d",
    "Yield_10Y_Change_20d",
    "TIPS_vs_Treasury",
]

# ── NEW Group 13: Credit & Bond Market (4) ───────────────────────────
FEAT_CREDIT = [
    "HY_IG_Spread",
    "Credit_Risk_Appetite",
    "TLT_Ret_5d",
    "TLT_Ret_20d",
]

# ── NEW Group 14: Sector Rotation (26) ───────────────────────────────
FEAT_SECTOR_ROTATION = [
    "XLK_Ret_5d", "XLK_Ret_20d",
    "XLF_Ret_5d", "XLF_Ret_20d",
    "XLE_Ret_5d", "XLE_Ret_20d",
    "XLV_Ret_5d", "XLV_Ret_20d",
    "XLI_Ret_5d", "XLI_Ret_20d",
    "XLP_Ret_5d", "XLP_Ret_20d",
    "XLY_Ret_5d", "XLY_Ret_20d",
    "XLB_Ret_5d", "XLB_Ret_20d",
    "XLU_Ret_5d", "XLU_Ret_20d",
    "XLRE_Ret_5d", "XLRE_Ret_20d",
    "XLC_Ret_5d", "XLC_Ret_20d",
    "Risk_On_Off",
    "Consumer_Cyclical_vs_Defensive",
    "Sector_Dispersion",
    "SOXX_Ret",   # semiconductor from sector module (not legacy)
]

# ── NEW Group 15: Commodities (14) ───────────────────────────────────
FEAT_COMMODITIES = [
    "Gold_Ret", "Gold_Ret_20d",
    "Silver_Ret", "Silver_Ret_20d",
    "Oil_Ret", "Oil_Ret_20d",
    "Brent_Ret", "Brent_Ret_20d",
    "NatGas_Ret", "NatGas_Ret_20d",
    "Copper_Ret", "Copper_Ret_20d",
    "Gold_Oil_Ratio",
    "Copper_Gold_Ratio",
]

# ── NEW Group 16: Currencies (11) ────────────────────────────────────
FEAT_CURRENCIES = [
    "DXY_Change", "DXY_Change_20d", "DXY_Momentum",
    "EURUSD_Change", "EURUSD_Change_20d",
    "GBPUSD_Change", "GBPUSD_Change_20d",
    "USDJPY_Change", "USDJPY_Change_20d",
    "USDCNY_Change", "USDCNY_Change_20d",
]

# ── NEW Group 17: Crypto Risk Appetite (5) ───────────────────────────
FEAT_CRYPTO = [
    "BTC_Ret", "BTC_Ret_20d", "BTC_Vol_20d",
    "ETH_Ret",
    "Crypto_Risk_Appetite",
]

# ── NEW Group 18: International Markets (16) ─────────────────────────
FEAT_INTERNATIONAL = [
    "EM_Ret", "EM_Ret_20d",
    "EAFE_Ret", "EAFE_Ret_20d",
    "China_Ret", "China_Ret_20d",
    "Japan_Ret", "Japan_Ret_20d",
    "Brazil_Ret", "Brazil_Ret_20d",
    "India_Ret", "India_Ret_20d",
    "Europe_Ret", "Europe_Ret_20d",
    "US_vs_Intl",
    "EM_vs_DM",
]

# ── NEW Group 19: Volatility Structure (6) ───────────────────────────
FEAT_VOLATILITY = [
    "VIX_Level", "VIX_Change_5d", "VIX_ZScore",
    "VIX_RealizedVol_Gap",
    "VXN_Level",
    "VIX_VXN_Spread",
]

# ── NEW Group 20: Market Breadth & Structure (10) ────────────────────
FEAT_BREADTH = [
    "SP500_Ret", "SP500_Ret_20d",
    "NASDAQ_Ret", "NASDAQ_Ret_20d",
    "DOW_Ret", "DOW_Ret_20d",
    "RUSSELL2000_Ret", "RUSSELL2000_Ret_20d",
    "LargeCap_vs_SmallCap",
    "Narrow_vs_Broad",
]

# ── NEW Group 21: Calendar Features (5) ──────────────────────────────
FEAT_CALENDAR = [
    "DayOfWeek",
    "MonthOfYear",
    "QuarterOfYear",
    "IsMonthEnd",
    "DaysToExpiry",          # options expiry proximity (3rd Friday)
]

# ── NEW Group 22: World Bank Macro Fundamentals (5) ──────────────────
FEAT_WORLDBANK = [
    "WB_GDP_Growth",
    "WB_Inflation",
    "WB_Unemployment",
    "WB_Trade_GDP",
    "WB_Gov_Debt_GDP",
]

# Full V2 feature list (de-duplicated, preserving order)
# Some groups share column names (e.g. Gold_Ret in FEAT_SECTOR & FEAT_COMMODITIES).
# dict.fromkeys removes duplicates while keeping first-seen order.
_ALL_FEATURE_GROUPS = (
    FEAT_OHLCV                # 5
    + FEAT_TECH_CLASSIC       # 8
    + FEAT_ENGINEERED         # 3
    + FEAT_TECH_ADV           # 9
    + FEAT_LAG                # 6
    + FEAT_MACRO              # 9
    + FEAT_FOMC               # 3
    + FEAT_MARKET             # 5
    + FEAT_SECTOR             # 4
    + FEAT_SENTIMENT          # 4
    + FEAT_REGIME_PROXY       # 3
    + FEAT_YIELD_CURVE        # 8
    + FEAT_CREDIT             # 4
    + FEAT_SECTOR_ROTATION    # 26
    + FEAT_COMMODITIES        # 14
    + FEAT_CURRENCIES         # 11
    + FEAT_CRYPTO             # 5
    + FEAT_INTERNATIONAL      # 16
    + FEAT_VOLATILITY         # 6
    + FEAT_BREADTH            # 10
    + FEAT_CALENDAR           # 5
    + FEAT_WORLDBANK          # 5
)
V2_FEATURE_COLS = list(dict.fromkeys(_ALL_FEATURE_GROUPS))  # ~164 unique

N_FEATURES = len(V2_FEATURE_COLS)

# ═══════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════
CALIBRATION_METHODS = ["isotonic", "platt"]
RELIABILITY_BINS = 15

# ═══════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════
SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# Data augmentation (more aggressive for larger model)
# ═══════════════════════════════════════════════════════════════════════
AUGMENT_NOISE_STD    = 0.003
AUGMENT_SCALE_JITTER = 0.02
AUGMENT_TIME_WARP    = 0.01      # slight temporal warping
AUGMENT_MIXUP_ALPHA  = 0.2       # mixup regularization

# ═══════════════════════════════════════════════════════════════════════
# Time-weighted training (year → weight)
# ═══════════════════════════════════════════════════════════════════════
TIME_WEIGHT_MAP = {
    2026: 1.0,
    2025: 1.0,
    2024: 1.0,
    2023: 0.95,
    2022: 0.90,
    2021: 0.85,
    2020: 0.80,
    2019: 0.70,
    2018: 0.60,
    2015: 0.50,
    2010: 0.40,
    2008: 0.35,      # GFC — rare but educational
    0:    0.25,
}
