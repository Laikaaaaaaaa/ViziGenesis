"""
ViziGenesis V2 — Walk-Forward Cross-Validation
=================================================
Time-aware expanding-window cross-validation for panel data.

Features:
  • Expanding-window folds (min 5yr train, 1yr val, 1yr step)
  • Per-fold model training + metric collection
  • Pooled validation predictions for calibration fitting
  • Final model trained on all data
  • ROC-AUC and Brier score as primary stopping metrics
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.v2.config import (
    WF_MIN_TRAIN_YEARS, WF_VAL_YEARS, WF_STEP_YEARS,
    WF_N_FOLDS_MAX, SEQ_LEN, N_FEATURES, MAX_EPOCHS, SEED,
)

logger = logging.getLogger("vizigenesis.v2.walk_forward")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Generate walk-forward folds
# ═══════════════════════════════════════════════════════════════════════
def generate_walk_forward_folds(
    n_samples: int,
    dates: List,
    min_train_years: float = WF_MIN_TRAIN_YEARS,
    val_years: float = WF_VAL_YEARS,
    step_years: float = WF_STEP_YEARS,
    max_folds: int = WF_N_FOLDS_MAX,
) -> List[Dict]:
    """
    Generate expanding-window fold indices.
    Returns list of dicts: {train_start, train_end, val_start, val_end}
    """
    if not dates or n_samples == 0:
        return []

    dates = pd.DatetimeIndex(dates)
    days_per_year = 252
    min_train_days = int(min_train_years * days_per_year)
    val_days = int(val_years * days_per_year)
    step_days = int(step_years * days_per_year)

    folds = []
    train_end = min_train_days

    while train_end + val_days <= n_samples and len(folds) < max_folds:
        val_end = min(train_end + val_days, n_samples)
        folds.append({
            "fold": len(folds) + 1,
            "train_start": 0,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": val_end,
            "train_size": train_end,
            "val_size": val_end - train_end,
        })
        train_end += step_days

    logger.info("Generated %d walk-forward folds (min_train=%d, val=%d, step=%d)",
                len(folds), min_train_days, val_days, step_days)
    return folds


# ═══════════════════════════════════════════════════════════════════════
# 2.  Run walk-forward validation
# ═══════════════════════════════════════════════════════════════════════
def run_walk_forward_validation(
    X: np.ndarray,             # (N, T, F)
    y_direction: np.ndarray,   # (N,)
    y_returns: np.ndarray,     # (N, 4)
    y_regime: np.ndarray,      # (N,)
    stock_ids: np.ndarray,     # (N,)
    dates: List,
    n_stocks: int = 5,
    n_features: int = N_FEATURES,
    epochs_per_fold: int = 50,
    callback: Optional[Callable] = None,
) -> Dict:
    """
    Full walk-forward pipeline:
      1. Generate folds
      2. Train per-fold
      3. Collect pooled val predictions
      4. Fit calibrator
      5. Train final model on all data
      6. Compute aggregate metrics

    Returns: {
      final_model, final_history, calibrator,
      aggregate_metrics, pooled_val_probs, pooled_val_labels,
      per_fold_metrics
    }
    """
    from backend.v2.model import train_hybrid_model, predict_hybrid
    from backend.v2.calibration import (
        CombinedCalibrator, compute_classification_metrics,
        compute_reliability_diagram,
    )

    folds = generate_walk_forward_folds(len(X), dates)
    if not folds:
        logger.warning("No valid walk-forward folds — training on full data")
        # Train directly on all data with 85/15 split
        n_val = max(int(len(X) * 0.15), 100)
        n_train = len(X) - n_val
        model, history = train_hybrid_model(
            X[:n_train], y_direction[:n_train], y_returns[:n_train],
            y_regime[:n_train],
            X[n_train:], y_direction[n_train:], y_returns[n_train:],
            y_regime[n_train:],
            stock_ids_train=stock_ids[:n_train],
            stock_ids_val=stock_ids[n_train:],
            n_stocks=n_stocks, n_features=n_features,
            epochs=epochs_per_fold,
        )
        return {
            "final_model": model,
            "final_history": history,
            "calibrator": None,
            "aggregate_metrics": {"n_folds": 0},
            "pooled_val_probs": np.array([]),
            "pooled_val_labels": np.array([]),
            "per_fold_metrics": [],
        }

    # ── Per-fold training ──────────────────────────────────────────────
    pooled_probs = []
    pooled_labels = []
    per_fold_metrics = []

    for fold in folds:
        fi = fold["fold"]
        ts, te = fold["train_start"], fold["train_end"]
        vs, ve = fold["val_start"], fold["val_end"]

        logger.info("Walk-forward fold %d: train[%d:%d] val[%d:%d]", fi, ts, te, vs, ve)

        if callback:
            callback(f"fold_{fi}", 0, 0, 0)

        def fold_callback(epoch, train_loss, val_loss, detail):
            if callback:
                callback(f"fold_{fi}", epoch, train_loss, val_loss)

        model, history = train_hybrid_model(
            X[ts:te], y_direction[ts:te], y_returns[ts:te], y_regime[ts:te],
            X[vs:ve], y_direction[vs:ve], y_returns[vs:ve], y_regime[vs:ve],
            stock_ids_train=stock_ids[ts:te],
            stock_ids_val=stock_ids[vs:ve],
            n_stocks=n_stocks, n_features=n_features,
            epochs=min(epochs_per_fold, 80),
            callback=fold_callback,
        )

        # Validation predictions
        val_preds = predict_hybrid(model, X[vs:ve], stock_ids[vs:ve])
        val_dir_probs = val_preds["direction"].squeeze()
        val_labels = y_direction[vs:ve]

        pooled_probs.append(val_dir_probs)
        pooled_labels.append(val_labels)

        # Per-fold metrics
        fold_metrics = compute_classification_metrics(val_labels, val_dir_probs)
        fold_metrics["fold"] = fi
        fold_metrics["train_size"] = fold["train_size"]
        fold_metrics["val_size"] = fold["val_size"]
        per_fold_metrics.append(fold_metrics)

        logger.info(
            "Fold %d: AUC=%.4f Brier=%.4f Acc=%.4f",
            fi,
            fold_metrics["raw"]["auc_roc"],
            fold_metrics["raw"]["brier"],
            fold_metrics["raw"]["accuracy"],
        )

    # ── Pool validation predictions ────────────────────────────────────
    all_probs = np.concatenate(pooled_probs)
    all_labels = np.concatenate(pooled_labels)

    # ── Fit calibrator ─────────────────────────────────────────────────
    calibrator = CombinedCalibrator()
    if len(all_probs) > 50:
        calibrator.fit(all_probs, all_labels)
        cal_probs = calibrator.predict(all_probs)
    else:
        cal_probs = all_probs

    # ── Aggregate metrics ──────────────────────────────────────────────
    aggregate = compute_classification_metrics(all_labels, all_probs, cal_probs)
    aggregate["n_folds"] = len(folds)
    aggregate["total_val_samples"] = len(all_probs)
    aggregate["reliability_diagram"] = {
        "raw": compute_reliability_diagram(all_labels, all_probs),
        "calibrated": compute_reliability_diagram(all_labels, cal_probs),
    }

    logger.info(
        "Aggregate: AUC=%.4f Brier_raw=%.4f Brier_cal=%.4f Acc=%.4f",
        aggregate["raw"]["auc_roc"],
        aggregate["raw"]["brier"],
        aggregate.get("calibrated", {}).get("brier", 0),
        aggregate["raw"]["accuracy"],
    )

    # ── Train final model on ALL data ──────────────────────────────────
    logger.info("Training final model on all %d samples...", len(X))
    n_val_final = max(int(len(X) * 0.10), 100)
    n_train_final = len(X) - n_val_final

    def final_callback(epoch, train_loss, val_loss, detail):
        if callback:
            callback("final", epoch, train_loss, val_loss)

    final_model, final_history = train_hybrid_model(
        X[:n_train_final], y_direction[:n_train_final],
        y_returns[:n_train_final], y_regime[:n_train_final],
        X[n_train_final:], y_direction[n_train_final:],
        y_returns[n_train_final:], y_regime[n_train_final:],
        stock_ids_train=stock_ids[:n_train_final],
        stock_ids_val=stock_ids[n_train_final:],
        n_stocks=n_stocks, n_features=n_features,
        epochs=epochs_per_fold,
        callback=final_callback,
    )

    return {
        "final_model": final_model,
        "final_history": final_history,
        "calibrator": calibrator,
        "aggregate_metrics": aggregate,
        "pooled_val_probs": all_probs,
        "pooled_val_labels": all_labels,
        "per_fold_metrics": per_fold_metrics,
    }
