"""
Configuration settings for ViziGenesis stock prediction platform.

Modes:
  - simple  → 5 features, basic LSTM
  - pro     → 20 features, LSTM+GRU ensemble
  - quant   → 41 features, TFT+BiLSTM+GRU hybrid with walk-forward validation,
               calibrated probabilities, multi-horizon predictions & backtest engine
"""
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Device (GPU if available) ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model hyper-parameters ─────────────────────────────────────────────
SEQUENCE_LENGTH = 60          # lookback window in trading days
HIDDEN_SIZE = 128             # LSTM hidden units
NUM_LAYERS = 2                # stacked LSTM layers
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOP_PATIENCE = 7      # stop if val loss doesn't improve

# ── Yahoo Finance cache TTL (seconds) ─────────────────────────────────
CACHE_TTL_REALTIME = 60       # real-time price: 1 min
CACHE_TTL_HISTORY = 3600      # historical data: 1 hour

# ── Feature columns used for training ─────────────────────────────────
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL = "Close"

# ── Data augmentation (time-series) ───────────────────────────────────
AUGMENT_NOISE_STD = 0.003      # Gaussian noise injection σ
AUGMENT_SCALE_JITTER = 0.02    # volatility scaling ±2 %

# ── Time-weighted training (year → weight) ────────────────────────────
TIME_WEIGHT_MAP = {
    2024: 1.0,   # 2024–present
    2020: 0.8,   # 2020–2023
    2015: 0.5,   # 2015–2019
    2008: 0.3,   # 2008–2014
    0:    0.2,   # earlier
}


# ═══════════════════════════════════════════════════════════════════════
# QUANT MODE — Institutional-grade configuration
# ═══════════════════════════════════════════════════════════════════════

# ── Quant hyper-parameters ─────────────────────────────────────────────
QUANT_SEQUENCE_LENGTH = 60    # same lookback for consistency
QUANT_HIDDEN_SIZE = 128       # hidden dimension for all branches
QUANT_NUM_HEADS = 4           # multi-head attention heads
QUANT_NUM_LAYERS = 2          # depth per branch
QUANT_DROPOUT = 0.25          # slightly higher for regularisation
QUANT_LEARNING_RATE = 5e-4    # smaller LR for hybrid model
QUANT_BATCH_SIZE = 64
QUANT_EPOCHS = 150            # more epochs with early stop
QUANT_EARLY_STOP = 12         # more patience for complex model

# ── Walk-forward validation ────────────────────────────────────────────
WF_MIN_TRAIN_YEARS = 3        # minimum training window
WF_VAL_YEARS = 1              # validation window per fold
WF_STEP_YEARS = 1             # slide step

# ── Multi-horizon targets ──────────────────────────────────────────────
HORIZONS = [1, 5, 30]         # forecast horizons (trading days)

# ── Backtest parameters ───────────────────────────────────────────────
BT_SIGNAL_THRESHOLD = 0.60    # P(up) > 0.60 to open long
BT_KELLY_CAP = 0.10           # max Kelly fraction (10 % of capital)
BT_INITIAL_CAPITAL = 100_000  # USD
BT_COMMISSION_BPS = 5         # 5 bps round-trip

# ── Loss weights for multi-task training ───────────────────────────────
LOSS_W_DIRECTION = 2.0        # binary cross-entropy weight
LOSS_W_RET_1D = 1.0           # 1-day return MSE
LOSS_W_RET_5D = 0.5           # 5-day return MSE
LOSS_W_RET_30D = 0.3          # 30-day return MSE
LOSS_W_EXCESS = 0.5           # excess return vs benchmark MSE

# ── Quant feature columns (41 features) ───────────────────────────────
QUANT_FEATURE_COLS = [
    # OHLCV (5)
    "Open", "High", "Low", "Close", "Volume",
    # Classic technicals (8)
    "MA20", "MA50", "EMA20", "EMA50", "RSI", "MACD", "Bollinger_Band", "OBV",
    # Engineered features (3)
    "Volume_Change", "Volatility", "ATR",
    # New quant technicals (9)
    "Stochastic_RSI", "ROC_10", "ROC_20",
    "Momentum_10", "Momentum_20",
    "VWAP", "BB_Width", "Hist_Vol_20", "Hist_Vol_60",
    # Lag features — past returns (6)
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    "Return_Lag_5", "Return_Lag_10", "Return_Lag_20",
    # Macro-economic (5)
    "SP500", "NASDAQ", "VIX", "BOND_10Y", "INFLATION_PROXY",
    # Sector ETF returns (2)
    "SOXX_Return", "SMH_Return",
    # Commodity & currency (3)
    "Gold_Return", "Oil_Return", "DXY_Level",
]

QUANT_MACRO_EXTRA_NAMES = [
    "SOXX_Return", "SMH_Return", "Gold_Return", "Oil_Return", "DXY_Level",
]
