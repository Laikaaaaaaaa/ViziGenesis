"""
ViziGenesis V2 — Regime Detection
====================================
Detects market regimes (bull / bear / sideways) using multiple methods:
  1. Rule-based: rolling returns + volatility thresholds
  2. Hidden Markov Model (HMM) inspired: GMM-based state detection
  3. VIX-threshold regime

Regime labels are used as:
  - Additional training target (regime classification head)
  - Meta-router input (model weighting per regime)
  - Feature-set modifier (emphasize macro in bear markets)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("vizigenesis.v2.regime")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Rule-based regime classification
# ═══════════════════════════════════════════════════════════════════════
def classify_regime_rules(
    close: pd.Series,
    vix: Optional[pd.Series] = None,
    window: int = 60,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
    vix_panic: float = 30.0,
) -> pd.Series:
    """
    Classify regime using rolling return + VIX thresholds.
      0 = bull   (positive momentum, moderate VIX)
      1 = bear   (negative momentum or VIX panic)
      2 = sideways (range-bound, high vol but no trend)
    """
    rolling_ret = close.pct_change(window).fillna(0)
    rolling_vol = close.pct_change().rolling(window, min_periods=10).std().fillna(0)

    regime = pd.Series(2, index=close.index, dtype=int)  # default sideways

    # Bull: strong positive returns
    regime[rolling_ret > bull_threshold] = 0

    # Bear: strong negative returns
    regime[rolling_ret < bear_threshold] = 1

    # VIX override
    if vix is not None:
        vix_aligned = vix.reindex(close.index).ffill().bfill()
        regime[vix_aligned > vix_panic] = 1  # VIX panic → bear

    # High vol + no trend → sideways
    vol_p85 = rolling_vol.quantile(0.85)
    choppy = (rolling_vol > vol_p85) & (rolling_ret.abs() < bull_threshold)
    regime[choppy] = 2

    regime.name = "Regime"
    return regime


# ═══════════════════════════════════════════════════════════════════════
# 2.  GMM-based regime detection (approximate HMM)
# ═══════════════════════════════════════════════════════════════════════
def classify_regime_gmm(
    close: pd.Series,
    n_regimes: int = 3,
    features_window: int = 20,
) -> pd.Series:
    """
    Use Gaussian Mixture Model on (return, volatility) features.
    Maps clusters to bull/bear/sideways by average return.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning("sklearn not available — falling back to rule-based regime")
        return classify_regime_rules(close)

    ret = close.pct_change().fillna(0)
    vol = ret.rolling(features_window, min_periods=5).std().fillna(0)
    mom = close.pct_change(features_window).fillna(0)

    X = np.column_stack([
        ret.values,
        vol.values,
        mom.values,
    ])

    # Replace NaN / inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(X)

    # Map clusters to regimes by average momentum
    cluster_avg_mom = {}
    for c in range(n_regimes):
        mask = labels == c
        cluster_avg_mom[c] = np.mean(mom.values[mask]) if mask.sum() > 0 else 0

    # Sort clusters by momentum: highest = bull, lowest = bear
    sorted_clusters = sorted(cluster_avg_mom.keys(), key=lambda c: cluster_avg_mom[c], reverse=True)
    cluster_to_regime = {}
    if n_regimes >= 3:
        cluster_to_regime[sorted_clusters[0]] = 0  # bull
        cluster_to_regime[sorted_clusters[-1]] = 1  # bear
        for c in sorted_clusters[1:-1]:
            cluster_to_regime[c] = 2  # sideways
    elif n_regimes == 2:
        cluster_to_regime[sorted_clusters[0]] = 0
        cluster_to_regime[sorted_clusters[1]] = 1
    else:
        cluster_to_regime[sorted_clusters[0]] = 2

    mapped = np.array([cluster_to_regime[l] for l in labels])
    regime = pd.Series(mapped, index=close.index, name="Regime", dtype=int)
    return regime


# ═══════════════════════════════════════════════════════════════════════
# 3.  Combined / ensemble regime detection
# ═══════════════════════════════════════════════════════════════════════
def detect_regime(
    stock_df: pd.DataFrame,
    vix: Optional[pd.Series] = None,
    method: str = "combined",
) -> pd.Series:
    """
    Main regime detection function.
    Methods: 'rules', 'gmm', 'combined' (majority vote).
    Returns Series with values 0, 1, 2 (bull, bear, sideways).
    """
    close = stock_df["Close"].astype(float)

    if method == "rules":
        return classify_regime_rules(close, vix=vix)
    elif method == "gmm":
        return classify_regime_gmm(close)
    elif method == "combined":
        # Majority vote of rules + GMM
        rules = classify_regime_rules(close, vix=vix)
        gmm = classify_regime_gmm(close)

        combined = pd.Series(2, index=close.index, dtype=int)
        for i in range(len(close)):
            votes = [rules.iloc[i], gmm.iloc[i]]
            # If both agree → use that
            if votes[0] == votes[1]:
                combined.iloc[i] = votes[0]
            else:
                # In disagreement, lean toward rules (more interpretable)
                combined.iloc[i] = votes[0]

        combined.name = "Regime"
        return combined
    else:
        return classify_regime_rules(close, vix=vix)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Regime statistics helper
# ═══════════════════════════════════════════════════════════════════════
def compute_regime_statistics(
    stock_df: pd.DataFrame,
    regime: pd.Series,
) -> Dict:
    """Compute statistics per regime (return, vol, Sharpe) for reporting."""
    close = stock_df["Close"].astype(float)
    ret = close.pct_change().fillna(0)

    stats = {}
    for r, name in enumerate(["bull", "bear", "sideways"]):
        mask = regime == r
        if mask.sum() == 0:
            stats[name] = {"days": 0, "avg_return": 0, "volatility": 0, "sharpe": 0}
            continue
        r_ret = ret[mask]
        avg = r_ret.mean() * 252
        vol = r_ret.std() * np.sqrt(252)
        sharpe = avg / vol if vol > 0 else 0
        stats[name] = {
            "days": int(mask.sum()),
            "pct_of_total": round(mask.sum() / len(regime) * 100, 1),
            "annualised_return": round(avg * 100, 2),
            "annualised_volatility": round(vol * 100, 2),
            "sharpe": round(sharpe, 3),
        }
    return stats


def get_current_regime(
    stock_df: pd.DataFrame,
    vix: Optional[pd.Series] = None,
) -> Dict:
    """Get the current (latest) regime and its confidence."""
    regime = detect_regime(stock_df, vix=vix)
    current = int(regime.iloc[-1])
    regime_name = ["bull", "bear", "sideways"][current]

    # Confidence: how long has this regime persisted?
    streak = 1
    for i in range(len(regime) - 2, -1, -1):
        if regime.iloc[i] == current:
            streak += 1
        else:
            break

    return {
        "regime": regime_name,
        "regime_id": current,
        "streak_days": streak,
        "confidence": min(streak / 20.0, 1.0),  # more days = more confident
    }
