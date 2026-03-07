"""
ViziGenesis V2 — Tree-Based Models (LightGBM + XGBoost)
=========================================================
Tabular heads that complement neural branches:
  • LightGBM   — gradient-boosted decision tree for direction + returns
  • XGBoost    — alternative GBDT ensemble
  • Stacking   — combines neural + tree predictions

These models are trained on flattened features (last row of sequence)
+ hand-crafted cross-sectional features.
"""
import logging, os, json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from backend.v2.config import V2_DIR, SEED, N_FEATURES

logger = logging.getLogger("vizigenesis.v2.tree_models")

# ── Lazy imports (graceful fallback) ──────────────────────────────────
_lgb = None
_xgb = None


def _import_lightgbm():
    global _lgb
    if _lgb is None:
        try:
            import lightgbm as lgb
            _lgb = lgb
        except ImportError:
            logger.warning("LightGBM not installed — tree model disabled")
    return _lgb


def _import_xgboost():
    global _xgb
    if _xgb is None:
        try:
            import xgboost as xgb
            _xgb = xgb
        except ImportError:
            logger.warning("XGBoost not installed — tree model disabled")
    return _xgb


# ═══════════════════════════════════════════════════════════════════════
# 1.  Flatten sequence features for tabular model
# ═══════════════════════════════════════════════════════════════════════
def flatten_sequences(X: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    """
    Convert (N, T, F) sequence matrix to (N, F*3) tabular matrix.
    Uses last row, mean, and std across time dimension as features.
    """
    # X: (N, T, F)
    last = X[:, -1, :]             # (N, F) — most recent values
    mean = X.mean(axis=1)          # (N, F) — rolling average
    std = X.std(axis=1)            # (N, F) — rolling vol

    cols = []
    data_parts = []

    for part, suffix in [(last, "_last"), (mean, "_mean"), (std, "_std")]:
        for i, col in enumerate(feature_cols):
            cols.append(f"{col}{suffix}")
        data_parts.append(part)

    flat = np.hstack(data_parts)   # (N, F*3)
    return pd.DataFrame(flat, columns=cols)


# ═══════════════════════════════════════════════════════════════════════
# 2.  LightGBM direction classifier + return regressors
# ═══════════════════════════════════════════════════════════════════════
def train_lightgbm(
    X_train: np.ndarray,
    y_dir_train: np.ndarray,
    y_ret_train: np.ndarray,
    X_val: np.ndarray,
    y_dir_val: np.ndarray,
    y_ret_val: np.ndarray,
    feature_cols: List[str],
) -> Optional[Dict]:
    """
    Train LightGBM classifier (direction) and regressors (returns).
    Returns dict: {"clf": model, "reg_1d": model, ...}
    """
    lgb = _import_lightgbm()
    if lgb is None:
        return None

    df_train = flatten_sequences(X_train, feature_cols)
    df_val = flatten_sequences(X_val, feature_cols)

    models = {}

    # Direction classifier
    clf_params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": 8,
        "min_child_samples": 50,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": SEED,
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(
        df_train, y_dir_train,
        eval_set=[(df_val, y_dir_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    models["clf"] = clf

    # Return regressors
    reg_params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": 8,
        "min_child_samples": 50,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "random_state": SEED,
        "verbose": -1,
    }

    for i, name in enumerate(["reg_1d", "reg_5d", "reg_30d", "reg_excess"]):
        reg = lgb.LGBMRegressor(**reg_params)
        y_train_i = y_ret_train[:, i]
        y_val_i = y_ret_val[:, i]
        # Filter NaN targets
        mask_train = np.isfinite(y_train_i)
        mask_val = np.isfinite(y_val_i)
        if mask_train.sum() < 100:
            continue
        reg.fit(
            df_train[mask_train], y_train_i[mask_train],
            eval_set=[(df_val[mask_val], y_val_i[mask_val])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        models[name] = reg

    logger.info("LightGBM trained: %s", list(models.keys()))
    return models


# ═══════════════════════════════════════════════════════════════════════
# 3.  XGBoost models
# ═══════════════════════════════════════════════════════════════════════
def train_xgboost(
    X_train: np.ndarray,
    y_dir_train: np.ndarray,
    y_ret_train: np.ndarray,
    X_val: np.ndarray,
    y_dir_val: np.ndarray,
    y_ret_val: np.ndarray,
    feature_cols: List[str],
) -> Optional[Dict]:
    """Train XGBoost classifier and regressors."""
    xgb = _import_xgboost()
    if xgb is None:
        return None

    df_train = flatten_sequences(X_train, feature_cols)
    df_val = flatten_sequences(X_val, feature_cols)

    models = {}

    # Direction classifier
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        verbosity=0,
        early_stopping_rounds=30,
    )
    clf.fit(
        df_train, y_dir_train,
        eval_set=[(df_val, y_dir_val)],
        verbose=False,
    )
    models["clf"] = clf

    # Return regressors
    for i, name in enumerate(["reg_1d", "reg_5d", "reg_30d", "reg_excess"]):
        reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            colsample_bytree=0.7,
            subsample=0.8,
            random_state=SEED,
            verbosity=0,
            early_stopping_rounds=30,
        )
        y_train_i = y_ret_train[:, i]
        y_val_i = y_ret_val[:, i]
        mask_train = np.isfinite(y_train_i)
        mask_val = np.isfinite(y_val_i)
        if mask_train.sum() < 100:
            continue
        reg.fit(
            df_train[mask_train], y_train_i[mask_train],
            eval_set=[(df_val[mask_val], y_val_i[mask_val])],
            verbose=False,
        )
        models[name] = reg

    logger.info("XGBoost trained: %s", list(models.keys()))
    return models


# ═══════════════════════════════════════════════════════════════════════
# 4.  Tree model inference
# ═══════════════════════════════════════════════════════════════════════
def predict_tree_models(
    models: Dict,
    X: np.ndarray,
    feature_cols: List[str],
) -> Dict[str, np.ndarray]:
    """Run inference with tree models."""
    df = flatten_sequences(X, feature_cols)
    result = {}

    if "clf" in models:
        result["direction_prob"] = models["clf"].predict_proba(df)[:, 1]
    if "reg_1d" in models:
        result["return_1d"] = models["reg_1d"].predict(df)
    if "reg_5d" in models:
        result["return_5d"] = models["reg_5d"].predict(df)
    if "reg_30d" in models:
        result["return_30d"] = models["reg_30d"].predict(df)
    if "reg_excess" in models:
        result["excess"] = models["reg_excess"].predict(df)

    return result


# ═══════════════════════════════════════════════════════════════════════
# 5.  Persistence
# ═══════════════════════════════════════════════════════════════════════
def save_tree_models(models: Dict, label: str = "lgbm"):
    d = os.path.join(V2_DIR, f"tree_{label}")
    os.makedirs(d, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(d, f"{name}.pkl"))
    logger.info("Tree models (%s) saved: %s", label, list(models.keys()))


def load_tree_models(label: str = "lgbm") -> Optional[Dict]:
    d = os.path.join(V2_DIR, f"tree_{label}")
    if not os.path.isdir(d):
        return None
    models = {}
    for f in os.listdir(d):
        if f.endswith(".pkl"):
            name = f[:-4]
            models[name] = joblib.load(os.path.join(d, f))
    return models if models else None
