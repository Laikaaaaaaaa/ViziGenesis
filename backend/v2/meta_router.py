"""
ViziGenesis V2 — Meta-Router / Ensemble Selector
===================================================
Routes predictions through multiple model families based on
current market regime, uncertainty, and volatility.

Candidate models: TFT/BiLSTM/GRU (neural branches), LightGBM, XGBoost.

The meta-router learns (or uses rules) to assign weights:
  • Regime head output → which models are strong in each regime
  • Uncertainty (entropy of direction prediction) → hedge toward trees
  • VIX/vol threshold → reduce neural weight in high-vol

Can be:
  1. Rule-based (default — no extra training needed)
  2. Learned MLP router (trained on validation fold performance)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("vizigenesis.v2.meta_router")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Rule-based meta-router
# ═══════════════════════════════════════════════════════════════════════
def rule_based_weights(
    regime: int,
    uncertainty: float,
    vix_level: float = 20.0,
    has_lgbm: bool = False,
    has_xgb: bool = False,
) -> Dict[str, float]:
    """
    Assign model weights based on regime, uncertainty, VIX.

    Returns dict: {"neural": w1, "lgbm": w2, "xgb": w3}
    Weights sum to 1.0.

    Strategy:
      Bull + low uncertainty → neural heavy (good at momentum capture)
      Bear + high uncertainty → tree heavy (more robust to noise)
      Sideways → equal weight
      High VIX → reduce neural weight
    """
    n_models = 1 + int(has_lgbm) + int(has_xgb)

    # Base weights
    if regime == 0:  # Bull
        w_neural = 0.60
        w_lgbm = 0.25
        w_xgb = 0.15
    elif regime == 1:  # Bear
        w_neural = 0.35
        w_lgbm = 0.40
        w_xgb = 0.25
    else:  # Sideways
        w_neural = 0.45
        w_lgbm = 0.30
        w_xgb = 0.25

    # Uncertainty adjustment: high entropy → trust trees more
    if uncertainty > 0.8:
        w_neural *= 0.7
        w_lgbm *= 1.2
        w_xgb *= 1.1
    elif uncertainty < 0.3:
        w_neural *= 1.2
        w_lgbm *= 0.8
        w_xgb *= 0.8

    # VIX adjustment: panic → reduce neural
    if vix_level > 30:
        w_neural *= 0.8
        w_lgbm *= 1.15
        w_xgb *= 1.15
    elif vix_level > 40:
        w_neural *= 0.6
        w_lgbm *= 1.3
        w_xgb *= 1.3

    # Normalize
    weights = {}
    weights["neural"] = w_neural
    if has_lgbm:
        weights["lgbm"] = w_lgbm
    if has_xgb:
        weights["xgb"] = w_xgb

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


# ═══════════════════════════════════════════════════════════════════════
# 2.  Learned MLP meta-router
# ═══════════════════════════════════════════════════════════════════════
class LearnedMetaRouter:
    """
    Small MLP trained on validation-fold predictions to learn optimal
    model weights per-sample.

    Input features:
      - regime probabilities (3)
      - uncertainty (1)
      - VIX z-score (1)
      - realised vol (1)
      - validation accuracy of each model (n_models)

    Output: softmax weights over n_models
    """
    def __init__(self, n_models: int = 3):
        self.n_models = n_models
        self.model = None
        self._fitted = False

    def fit(
        self,
        regime_probs: np.ndarray,    # (N, 3)
        uncertainty: np.ndarray,      # (N,)
        vix_zscore: np.ndarray,       # (N,)
        realised_vol: np.ndarray,     # (N,)
        model_predictions: np.ndarray,  # (N, n_models) — direction probs from each
        y_true: np.ndarray,           # (N,) — actual direction
    ):
        """
        Train the meta-router. Uses logistic regression on per-model
        correctness weighted by context features.
        """
        from sklearn.linear_model import LogisticRegression

        N = len(y_true)
        if N < 100:
            logger.warning("Too few samples for learned router (%d)", N)
            return

        # For each model, compute correctness
        best_model_per_sample = np.zeros(N, dtype=int)
        for i in range(N):
            errors = []
            for m in range(self.n_models):
                pred_dir = 1.0 if model_predictions[i, m] > 0.5 else 0.0
                errors.append(abs(pred_dir - y_true[i]))
            best_model_per_sample[i] = np.argmin(errors)

        # Build feature matrix
        X = np.column_stack([
            regime_probs,
            uncertainty.reshape(-1, 1),
            vix_zscore.reshape(-1, 1),
            realised_vol.reshape(-1, 1),
        ])

        self.model = LogisticRegression(
            max_iter=500,
            multi_class="multinomial",
            random_state=42,
        )
        self.model.fit(X, best_model_per_sample)
        self._fitted = True
        logger.info("Learned meta-router trained on %d samples", N)

    def predict_weights(
        self,
        regime_probs: np.ndarray,
        uncertainty: np.ndarray,
        vix_zscore: np.ndarray,
        realised_vol: np.ndarray,
    ) -> np.ndarray:
        """Return per-sample model weights (N, n_models)."""
        if not self._fitted or self.model is None:
            # Fallback: equal weights
            N = len(uncertainty)
            return np.ones((N, self.n_models)) / self.n_models

        X = np.column_stack([
            regime_probs,
            uncertainty.reshape(-1, 1),
            vix_zscore.reshape(-1, 1),
            realised_vol.reshape(-1, 1),
        ])
        return self.model.predict_proba(X)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Ensemble combiner
# ═══════════════════════════════════════════════════════════════════════
def combine_predictions(
    neural_preds: Dict[str, np.ndarray],
    lgbm_preds: Optional[Dict[str, np.ndarray]] = None,
    xgb_preds: Optional[Dict[str, np.ndarray]] = None,
    weights: Optional[Dict[str, float]] = None,
    regime: int = 2,
    uncertainty: float = 0.5,
    vix_level: float = 20.0,
) -> Dict[str, np.ndarray]:
    """
    Combine predictions from neural + tree models using meta-router weights.
    Returns final blended predictions.
    """
    if weights is None:
        weights = rule_based_weights(
            regime=regime,
            uncertainty=uncertainty,
            vix_level=vix_level,
            has_lgbm=lgbm_preds is not None,
            has_xgb=xgb_preds is not None,
        )

    # Blend direction probability
    dir_blend = neural_preds["direction"].squeeze() * weights.get("neural", 1.0)
    if lgbm_preds and "direction_prob" in lgbm_preds:
        dir_blend = dir_blend + lgbm_preds["direction_prob"] * weights.get("lgbm", 0)
    if xgb_preds and "direction_prob" in xgb_preds:
        dir_blend = dir_blend + xgb_preds["direction_prob"] * weights.get("xgb", 0)

    # Blend returns
    result = {"direction": dir_blend}
    for key in ["return_1d", "return_5d", "return_30d", "excess"]:
        val = neural_preds.get(key, np.zeros(1)).squeeze() * weights.get("neural", 1.0)
        if lgbm_preds and key in lgbm_preds:
            val = val + lgbm_preds[key] * weights.get("lgbm", 0)
        if xgb_preds and key in xgb_preds:
            val = val + xgb_preds[key] * weights.get("xgb", 0)
        result[key] = val

    # Pass through regime and other neural-only outputs
    for key in ["regime", "branch_weights", "feat_weights"]:
        if key in neural_preds:
            result[key] = neural_preds[key]

    result["meta_weights"] = weights

    return result


def compute_uncertainty(direction_prob: float) -> float:
    """
    Shannon entropy of binary direction prediction.
    0 = certain, 1 = maximum uncertainty.
    """
    p = np.clip(direction_prob, 1e-7, 1 - 1e-7)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(entropy)
