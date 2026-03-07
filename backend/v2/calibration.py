"""
ViziGenesis V2 — Probability Calibration
==========================================
Converts raw model outputs to calibrated probabilities using:
  1. Isotonic Regression  (non-parametric, preferred for NN)
  2. Platt Scaling        (logistic calibration)

Evaluation:
  • Brier score
  • Reliability diagrams (calibration curves)
  • AUC-ROC, precision, recall, F1

Also includes concept-drift detection for triggering retraining.
"""
import logging, os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_score,
    recall_score, f1_score, accuracy_score,
    log_loss, confusion_matrix,
)

from backend.v2.config import V2_DIR, RELIABILITY_BINS

logger = logging.getLogger("vizigenesis.v2.calibration")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Isotonic Regression calibrator
# ═══════════════════════════════════════════════════════════════════════
def fit_isotonic_calibrator(
    raw_probs: np.ndarray,
    true_labels: np.ndarray,
) -> IsotonicRegression:
    """
    Fit Isotonic Regression calibrator.
    raw_probs:  model P(up) in [0, 1]
    true_labels: actual direction 0/1
    """
    mask = np.isfinite(raw_probs) & np.isfinite(true_labels)
    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    cal.fit(raw_probs[mask], true_labels[mask])
    return cal


def calibrate_isotonic(calibrator: IsotonicRegression, raw_probs: np.ndarray) -> np.ndarray:
    """Apply isotonic calibration."""
    return calibrator.predict(np.clip(raw_probs, 0, 1))


# ═══════════════════════════════════════════════════════════════════════
# 2.  Platt Scaling calibrator
# ═══════════════════════════════════════════════════════════════════════
def fit_platt_calibrator(
    raw_probs: np.ndarray,
    true_labels: np.ndarray,
) -> LogisticRegression:
    """
    Fit Platt Scaling (logistic regression on logit of raw probs).
    """
    mask = np.isfinite(raw_probs) & np.isfinite(true_labels)
    # Convert to logits
    clipped = np.clip(raw_probs[mask], 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(logits, true_labels[mask].astype(int))
    return platt


def calibrate_platt(calibrator: LogisticRegression, raw_probs: np.ndarray) -> np.ndarray:
    """Apply Platt Scaling calibration."""
    clipped = np.clip(raw_probs, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


# ═══════════════════════════════════════════════════════════════════════
# 3.  Combined calibrator (best of Isotonic + Platt via Brier score)
# ═══════════════════════════════════════════════════════════════════════
class CombinedCalibrator:
    """
    Fits both Isotonic and Platt calibrators, selects the one with
    lower Brier score on validation data.
    """
    def __init__(self):
        self.isotonic = None
        self.platt = None
        self.best_method = "isotonic"
        self.brier_scores = {}

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray):
        self.isotonic = fit_isotonic_calibrator(raw_probs, true_labels)
        self.platt = fit_platt_calibrator(raw_probs, true_labels)

        # Compare on same data (ideally use held-out set)
        iso_cal = calibrate_isotonic(self.isotonic, raw_probs)
        platt_cal = calibrate_platt(self.platt, raw_probs)

        mask = np.isfinite(true_labels)
        brier_iso = brier_score_loss(true_labels[mask], iso_cal[mask])
        brier_platt = brier_score_loss(true_labels[mask], platt_cal[mask])

        self.brier_scores = {"isotonic": brier_iso, "platt": brier_platt}
        self.best_method = "isotonic" if brier_iso <= brier_platt else "platt"

        logger.info(
            "Calibration fitted — Brier isotonic=%.4f platt=%.4f → using %s",
            brier_iso, brier_platt, self.best_method,
        )

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.best_method == "platt" and self.platt is not None:
            return calibrate_platt(self.platt, raw_probs)
        elif self.isotonic is not None:
            return calibrate_isotonic(self.isotonic, raw_probs)
        return raw_probs

    def predict_both(self, raw_probs: np.ndarray) -> Dict[str, np.ndarray]:
        result = {"raw": raw_probs}
        if self.isotonic is not None:
            result["isotonic"] = calibrate_isotonic(self.isotonic, raw_probs)
        if self.platt is not None:
            result["platt"] = calibrate_platt(self.platt, raw_probs)
        return result


# ═══════════════════════════════════════════════════════════════════════
# 4.  Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════
def compute_classification_metrics(
    true_labels: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict:
    """Compute comprehensive classification metrics."""
    mask = np.isfinite(true_labels) & np.isfinite(raw_probs)
    y_true = true_labels[mask].astype(int)

    metrics = {}

    # Raw metrics
    raw_p = raw_probs[mask]
    raw_pred = (raw_p > threshold).astype(int)
    metrics["raw"] = {
        "accuracy": float(accuracy_score(y_true, raw_pred)),
        "precision": float(precision_score(y_true, raw_pred, zero_division=0)),
        "recall": float(recall_score(y_true, raw_pred, zero_division=0)),
        "f1": float(f1_score(y_true, raw_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, raw_p)) if len(np.unique(y_true)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_true, raw_p)),
        "log_loss": float(log_loss(y_true, np.clip(raw_p, 1e-7, 1 - 1e-7))),
    }

    # Calibrated metrics
    if calibrated_probs is not None:
        cal_p = calibrated_probs[mask]
        cal_pred = (cal_p > threshold).astype(int)
        metrics["calibrated"] = {
            "accuracy": float(accuracy_score(y_true, cal_pred)),
            "precision": float(precision_score(y_true, cal_pred, zero_division=0)),
            "recall": float(recall_score(y_true, cal_pred, zero_division=0)),
            "f1": float(f1_score(y_true, cal_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, cal_p)) if len(np.unique(y_true)) > 1 else 0.5,
            "brier": float(brier_score_loss(y_true, cal_p)),
            "log_loss": float(log_loss(y_true, np.clip(cal_p, 1e-7, 1 - 1e-7))),
        }

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, raw_pred).tolist()

    return metrics


def compute_reliability_diagram(
    true_labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = RELIABILITY_BINS,
) -> List[Dict]:
    """
    Compute reliability diagram data points.
    Returns list of {bin_center, predicted_prob, actual_freq, count}.
    """
    mask = np.isfinite(true_labels) & np.isfinite(probs)
    y = true_labels[mask]
    p = probs[mask]

    bins = np.linspace(0, 1, n_bins + 1)
    diagram = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_center = (lo + hi) / 2
        predicted = p[in_bin].mean()
        actual = y[in_bin].mean()
        diagram.append({
            "bin_center": round(bin_center, 3),
            "predicted_prob": round(float(predicted), 4),
            "actual_freq": round(float(actual), 4),
            "count": int(count),
        })

    return diagram


# ═══════════════════════════════════════════════════════════════════════
# 5.  Concept drift detection
# ═══════════════════════════════════════════════════════════════════════
def detect_concept_drift(
    recent_probs: np.ndarray,
    recent_labels: np.ndarray,
    baseline_brier: float,
    threshold_factor: float = 1.5,
) -> Dict:
    """
    Detect if model calibration has degraded (concept drift).
    If recent Brier score exceeds baseline by threshold_factor → trigger retrain.
    """
    mask = np.isfinite(recent_labels) & np.isfinite(recent_probs)
    if mask.sum() < 30:
        return {"drift_detected": False, "reason": "insufficient_data"}

    recent_brier = brier_score_loss(recent_labels[mask], recent_probs[mask])
    drift = recent_brier > baseline_brier * threshold_factor

    return {
        "drift_detected": bool(drift),
        "recent_brier": round(float(recent_brier), 4),
        "baseline_brier": round(float(baseline_brier), 4),
        "ratio": round(float(recent_brier / max(baseline_brier, 1e-6)), 3),
        "threshold": threshold_factor,
        "should_retrain": drift,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6.  Persistence
# ═══════════════════════════════════════════════════════════════════════
def save_calibrator(calibrator: CombinedCalibrator, label: str = "panel"):
    d = V2_DIR
    os.makedirs(d, exist_ok=True)
    joblib.dump(calibrator, os.path.join(d, f"calibrator_{label}.pkl"))
    logger.info("Calibrator saved: %s", label)


def load_calibrator(label: str = "panel") -> Optional[CombinedCalibrator]:
    path = os.path.join(V2_DIR, f"calibrator_{label}.pkl")
    return joblib.load(path) if os.path.exists(path) else None
