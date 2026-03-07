"""
Walk-Forward Validation + Isotonic Regression Probability Calibration.

Walk-forward splits data into expanding training windows:
  Fold 1: Train 2014–2017 → Val 2018
  Fold 2: Train 2014–2018 → Val 2019
  ...
  Fold N: Train 2014–(T-1) → Val T

After all folds, calibrates direction probabilities using Isotonic Regression
on pooled validation predictions.
"""
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

from backend.config import (
    DEVICE, QUANT_SEQUENCE_LENGTH, QUANT_FEATURE_COLS,
    WF_MIN_TRAIN_YEARS, WF_VAL_YEARS, WF_STEP_YEARS,
    QUANT_EPOCHS, QUANT_LEARNING_RATE, QUANT_BATCH_SIZE,
    MODEL_DIR,
)

logger = logging.getLogger("vizigenesis.walk_forward")


# ═══════════════════════════════════════════════════════════════════════
# 1. Walk-forward fold generator
# ═══════════════════════════════════════════════════════════════════════

def generate_walk_forward_folds(
    df: pd.DataFrame,
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    val_years: int = WF_VAL_YEARS,
    step_years: int = WF_STEP_YEARS,
) -> List[Dict]:
    """
    Generate expanding-window walk-forward folds.

    Each fold is a dict: {
        "fold": int,
        "train_start": Timestamp,
        "train_end": Timestamp,
        "val_start": Timestamp,
        "val_end": Timestamp,
        "train_idx": (start_row, end_row),
        "val_idx": (start_row, end_row),
    }
    """
    dates = pd.to_datetime(df.index)
    start_year = dates.min().year
    end_year = dates.max().year

    folds = []
    fold_num = 0

    # First validation year starts after min_train_years
    first_val_year = start_year + min_train_years

    for val_start_year in range(first_val_year, end_year + 1, step_years):
        val_end_year = val_start_year + val_years - 1
        if val_end_year > end_year:
            break

        train_mask = dates.year < val_start_year
        val_mask = (dates.year >= val_start_year) & (dates.year <= val_end_year)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        if len(train_indices) < QUANT_SEQUENCE_LENGTH + 10 or len(val_indices) < 10:
            continue

        fold_num += 1
        folds.append({
            "fold": fold_num,
            "train_start": dates[train_indices[0]],
            "train_end": dates[train_indices[-1]],
            "val_start": dates[val_indices[0]],
            "val_end": dates[val_indices[-1]],
            "train_idx": (int(train_indices[0]), int(train_indices[-1]) + 1),
            "val_idx": (int(val_indices[0]), int(val_indices[-1]) + 1),
        })

    logger.info("Generated %d walk-forward folds from %d to %d", len(folds), start_year, end_year)
    return folds


# ═══════════════════════════════════════════════════════════════════════
# 2. Walk-forward training engine
# ═══════════════════════════════════════════════════════════════════════

def run_walk_forward_validation(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = QUANT_SEQUENCE_LENGTH,
    epochs_per_fold: int = None,
    callback=None,
) -> Dict:
    """
    Run full walk-forward validation cycle.

    Returns dict with:
      - folds: list of per-fold metrics
      - pooled_val_probs: all validation direction probabilities
      - pooled_val_labels: all validation direction labels
      - pooled_val_returns: dict of pooled validation predictions per target
      - calibrator: fitted IsotonicRegression
      - aggregate_metrics: summary across all folds
      - final_model: model trained on all data (for deployment)
    """
    from sklearn.preprocessing import MinMaxScaler
    from backend.quant_model import (
        QuantHybridModel, train_quant_model, predict_quant,
    )

    epochs_per_fold = epochs_per_fold or max(QUANT_EPOCHS // 2, 50)

    folds = generate_walk_forward_folds(features_df)
    if not folds:
        raise ValueError("Not enough data for walk-forward validation. Need at least "
                         f"{WF_MIN_TRAIN_YEARS + WF_VAL_YEARS} years of history.")

    data = features_df[feature_cols].values.astype(np.float32)
    target_names = list(targets_df.columns)
    target_data = {col: targets_df[col].values.astype(np.float32) for col in target_names}

    n_features = len(feature_cols)

    # Collect all validation predictions
    all_val_probs = []
    all_val_labels = []
    all_val_preds = {t: [] for t in target_names}
    all_val_actuals = {t: [] for t in target_names}
    fold_results = []

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_start, train_end = fold_info["train_idx"]
        val_start, val_end = fold_info["val_idx"]

        logger.info(
            "Fold %d: Train %s → %s | Val %s → %s",
            fold_num,
            fold_info["train_start"].strftime("%Y-%m-%d"),
            fold_info["train_end"].strftime("%Y-%m-%d"),
            fold_info["val_start"].strftime("%Y-%m-%d"),
            fold_info["val_end"].strftime("%Y-%m-%d"),
        )

        # Fit scaler on training data only
        scaler = MinMaxScaler()
        scaler.fit(data[train_start:train_end])
        scaled = scaler.transform(data)

        # Build sequences for this fold
        X_train_list, y_train_dict, m_train_dict = [], {t: [] for t in target_names}, {t: [] for t in target_names}
        X_val_list, y_val_dict, m_val_dict = [], {t: [] for t in target_names}, {t: [] for t in target_names}

        # Training sequences (from within training window)
        for i in range(train_start + seq_len, train_end):
            X_train_list.append(scaled[i - seq_len:i])
            for t in target_names:
                val = target_data[t][i]
                y_train_dict[t].append(val if np.isfinite(val) else 0.0)
                m_train_dict[t].append(np.isfinite(target_data[t][i]))

        # Validation sequences (need seq_len history before each val point)
        for i in range(max(val_start, seq_len), val_end):
            X_val_list.append(scaled[i - seq_len:i])
            for t in target_names:
                val = target_data[t][i]
                y_val_dict[t].append(val if np.isfinite(val) else 0.0)
                m_val_dict[t].append(np.isfinite(target_data[t][i]))

        if len(X_train_list) < 20 or len(X_val_list) < 5:
            logger.warning("Fold %d: insufficient data, skipping", fold_num)
            continue

        X_train = np.array(X_train_list, dtype=np.float32)
        X_val = np.array(X_val_list, dtype=np.float32)
        y_train = {t: np.array(v, dtype=np.float32) for t, v in y_train_dict.items()}
        y_val = {t: np.array(v, dtype=np.float32) for t, v in y_val_dict.items()}
        masks_train = {t: np.array(v, dtype=bool) for t, v in m_train_dict.items()}
        masks_val = {t: np.array(v, dtype=bool) for t, v in m_val_dict.items()}

        # Train model for this fold
        fold_callback = None
        if callback:
            def fold_callback(epoch, train_loss, val_loss, _fold=fold_num):
                callback(f"fold_{_fold}", epoch, train_loss, val_loss)

        model, history = train_quant_model(
            X_train, y_train, masks_train,
            X_val, y_val, masks_val,
            n_features=n_features,
            epochs=epochs_per_fold,
            callback=fold_callback,
        )

        # Evaluate on validation set
        val_preds = predict_quant(model, X_val)

        # Collect direction probabilities for calibration
        dir_probs = val_preds["direction"].flatten()
        dir_labels = y_val["Direction"]
        dir_mask = masks_val["Direction"]

        valid_probs = dir_probs[dir_mask]
        valid_labels = dir_labels[dir_mask]

        if len(valid_probs) > 0:
            all_val_probs.extend(valid_probs.tolist())
            all_val_labels.extend(valid_labels.tolist())

        # Collect other targets
        for t in target_names:
            mask = masks_val[t]
            pred_key = {
                "Direction": "direction",
                "Return_1d": "return_1d",
                "Return_5d": "return_5d",
                "Return_30d": "return_30d",
                "Excess_Return": "excess",
            }.get(t)
            if pred_key and pred_key in val_preds:
                all_val_preds[t].extend(val_preds[pred_key].flatten()[mask].tolist())
                all_val_actuals[t].extend(y_val[t][mask].tolist())

        # Per-fold metrics
        fold_metrics = _compute_fold_metrics(valid_probs, valid_labels)
        fold_metrics["fold"] = fold_num
        fold_metrics["train_samples"] = len(X_train)
        fold_metrics["val_samples"] = len(X_val)
        fold_metrics["epochs_run"] = len(history)
        fold_metrics["final_val_loss"] = history[-1]["val_loss"] if history else None
        fold_results.append(fold_metrics)

        logger.info("Fold %d metrics: %s", fold_num, fold_metrics)

    # ── Calibration via Isotonic Regression ───────────────────────────
    calibrator = None
    if len(all_val_probs) >= 20:
        calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        calibrator.fit(all_val_probs, all_val_labels)
        logger.info("Isotonic calibrator fitted on %d pooled validation samples", len(all_val_probs))
    else:
        logger.warning("Not enough validation data for calibration (%d samples)", len(all_val_probs))

    # ── Aggregate metrics ─────────────────────────────────────────────
    aggregate = _compute_aggregate_metrics(
        np.array(all_val_probs), np.array(all_val_labels), calibrator
    ) if len(all_val_probs) >= 10 else {}
    aggregate["n_folds"] = len(fold_results)
    aggregate["total_val_samples"] = len(all_val_probs)

    # ── Train final model on ALL data ─────────────────────────────────
    logger.info("Training final model on all available data...")
    from backend.quant_features import prepare_quant_sequences

    final_data = prepare_quant_sequences(
        features_df, targets_df, feature_cols,
        seq_len=seq_len, train_ratio=0.9,
    )

    final_model, final_history = train_quant_model(
        final_data["X_train"], final_data["y_train"], final_data["masks_train"],
        final_data["X_val"], final_data["y_val"], final_data["masks_val"],
        n_features=n_features,
        epochs=QUANT_EPOCHS,
        callback=callback,
    )

    return {
        "folds": fold_results,
        "pooled_val_probs": np.array(all_val_probs),
        "pooled_val_labels": np.array(all_val_labels),
        "calibrator": calibrator,
        "aggregate_metrics": aggregate,
        "final_model": final_model,
        "final_scaler": final_data["scaler"],
        "final_history": final_history,
    }


def _compute_fold_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute classification metrics for a single fold."""
    if len(probs) == 0:
        return {}

    preds_binary = (probs > 0.5).astype(int)
    labels_int = labels.astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(labels_int, preds_binary)), 4),
    }

    # Protect against single-class validation sets
    try:
        metrics["auc_roc"] = round(float(roc_auc_score(labels_int, probs)), 4)
    except ValueError:
        metrics["auc_roc"] = None

    try:
        metrics["brier_score"] = round(float(brier_score_loss(labels_int, probs)), 6)
    except Exception:
        metrics["brier_score"] = None

    metrics["precision"] = round(float(precision_score(labels_int, preds_binary, zero_division=0)), 4)
    metrics["recall"] = round(float(recall_score(labels_int, preds_binary, zero_division=0)), 4)
    metrics["f1"] = round(float(f1_score(labels_int, preds_binary, zero_division=0)), 4)

    return metrics


def _compute_aggregate_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    calibrator: Optional[IsotonicRegression],
) -> Dict:
    """Compute aggregate metrics across all walk-forward folds."""
    raw_metrics = _compute_fold_metrics(probs, labels)

    result = {"raw": raw_metrics}

    if calibrator is not None:
        cal_probs = calibrator.predict(probs)
        cal_metrics = _compute_fold_metrics(cal_probs, labels)
        result["calibrated"] = cal_metrics

        # Reliability diagram data (10 bins)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        reliability = []
        for i in range(n_bins):
            mask = (cal_probs >= bin_edges[i]) & (cal_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                reliability.append({
                    "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2),
                    "mean_predicted": round(float(cal_probs[mask].mean()), 4),
                    "fraction_positive": round(float(labels[mask].mean()), 4),
                    "count": int(mask.sum()),
                })
        result["reliability_diagram"] = reliability

    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. Calibrator persistence
# ═══════════════════════════════════════════════════════════════════════

def save_calibrator(calibrator: IsotonicRegression, symbol: str):
    """Save the isotonic calibrator to disk."""
    import joblib
    safe = symbol.upper().replace("/", "_").replace("\\", "_").strip()
    d = os.path.join(MODEL_DIR, safe)
    os.makedirs(d, exist_ok=True)
    joblib.dump(calibrator, os.path.join(d, "quant_calibrator.pkl"))


def load_calibrator(symbol: str) -> Optional[IsotonicRegression]:
    """Load a previously saved calibrator."""
    import joblib
    safe = symbol.upper().replace("/", "_").replace("\\", "_").strip()
    path = os.path.join(MODEL_DIR, safe, "quant_calibrator.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None
