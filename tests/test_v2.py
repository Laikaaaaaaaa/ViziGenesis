"""
ViziGenesis V2 — Test Suite
============================
Unit tests for the V2 pipeline modules.
Run: python -m pytest tests/ -v
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════
@pytest.fixture
def sample_ohlcv():
    """Generate 300-day synthetic OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 10)  # Ensure positive
    df = pd.DataFrame({
        "Open": close * (1 + np.random.randn(n) * 0.005),
        "High": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "Low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)
    return df


@pytest.fixture
def sample_sequences():
    """Generate synthetic sequences for model testing."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 60
    n_features = 59
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_dir = (np.random.rand(n_samples) > 0.5).astype(np.float32)
    y_ret = np.random.randn(n_samples, 4).astype(np.float32) * 2  # 1d, 5d, 30d, excess
    y_regime = np.random.randint(0, 3, n_samples).astype(np.int64)
    return X, y_dir, y_ret, y_regime


# ═══════════════════════════════════════════════════════════════════════
# 1. Config tests
# ═══════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_v2_config_loads(self):
        from backend.v2.config import (
            PILOT_TICKERS, V2_FEATURE_COLS, D_MODEL, N_HEADS,
            N_LAYERS, SEQ_LEN, BATCH_SIZE, LEARNING_RATE,
        )
        assert len(PILOT_TICKERS) == 5
        assert len(V2_FEATURE_COLS) >= 50
        assert D_MODEL == 192
        assert N_HEADS == 8
        assert N_LAYERS == 3
        assert SEQ_LEN == 60
        assert BATCH_SIZE == 512
        assert LEARNING_RATE == 1e-4

    def test_crisis_periods(self):
        from backend.v2.config import CRISIS_PERIODS
        assert "GFC_2008" in CRISIS_PERIODS
        assert "COVID_2020" in CRISIS_PERIODS
        assert len(CRISIS_PERIODS) >= 3


# ═══════════════════════════════════════════════════════════════════════
# 2. Feature Engineering tests
# ═══════════════════════════════════════════════════════════════════════
class TestFeatures:
    def test_add_technical_indicators(self, sample_ohlcv):
        from backend.v2.features import add_technical_indicators
        df = add_technical_indicators(sample_ohlcv.copy())
        # Should have new columns
        for col in ["RSI", "MACD", "MA20", "MA50", "EMA20", "OBV", "ATR"]:
            assert col in df.columns, f"Missing column: {col}"
        # No NaN explosion (after dropna, should have reasonable rows)
        assert len(df.dropna()) > 100

    def test_generate_targets(self, sample_ohlcv):
        from backend.v2.features import generate_targets
        df = generate_targets(sample_ohlcv.copy())
        assert "Direction" in df.columns
        assert "Return_1d" in df.columns
        assert "Return_5d" in df.columns
        assert "Return_30d" in df.columns
        assert df["Direction"].isin([0, 1]).all()

    def test_generate_regime_labels(self, sample_ohlcv):
        from backend.v2.features import generate_regime_labels
        labels = generate_regime_labels(sample_ohlcv.copy())
        assert len(labels) == len(sample_ohlcv)
        assert set(labels.dropna().unique()).issubset({0, 1, 2})  # bull, bear, sideways

    def test_rolling_zscore(self, sample_ohlcv):
        from backend.v2.features import rolling_zscore
        df = sample_ohlcv[["Close", "Volume"]].copy()
        z = rolling_zscore(df, 20)
        assert len(z) == len(df)
        # Z-scores should be mostly in [-5, 5]
        valid = z.dropna()
        assert (valid.abs() < 10).all().all()


# ═══════════════════════════════════════════════════════════════════════
# 3. Regime Detection tests
# ═══════════════════════════════════════════════════════════════════════
class TestRegime:
    def test_classify_regime_rules(self, sample_ohlcv):
        from backend.v2.regime import classify_regime_rules
        close = sample_ohlcv["Close"]
        labels = classify_regime_rules(close)
        assert len(labels) == len(sample_ohlcv)
        assert set(labels.dropna().unique()).issubset({0, 1, 2})  # int codes

    def test_detect_regime(self, sample_ohlcv):
        from backend.v2.regime import detect_regime
        labels = detect_regime(sample_ohlcv, method="rules")
        assert len(labels) == len(sample_ohlcv)

    def test_get_current_regime(self, sample_ohlcv):
        from backend.v2.regime import get_current_regime
        result = get_current_regime(sample_ohlcv)
        assert "regime" in result
        assert result["regime"] in {"bull", "bear", "sideways"}
        assert "confidence" in result


# ═══════════════════════════════════════════════════════════════════════
# 4. Model Architecture tests
# ═══════════════════════════════════════════════════════════════════════
class TestModel:
    def test_hybrid_forecaster_forward(self):
        """Test that HybridForecaster produces correct output shapes."""
        import torch
        from backend.v2.model import HybridForecaster

        n_features = 59
        model = HybridForecaster(
            n_features=n_features,
            d_model=64,  # Small for testing
            n_heads=4,
            n_layers=1,
            dropout=0.1,
            n_stocks=5,
        )
        model.eval()

        batch = 8
        seq_len = 60
        x = torch.randn(batch, seq_len, n_features)
        stock_ids = torch.zeros(batch, dtype=torch.long)

        with torch.no_grad():
            out = model(x, stock_ids)

        assert out["direction"].shape == (batch, 1)
        assert out["return_1d"].shape == (batch, 1)
        assert out["return_5d"].shape == (batch, 1)
        assert out["return_30d"].shape == (batch, 1)
        assert out["excess"].shape == (batch, 1)
        assert out["regime"].shape == (batch, 3)
        assert out["branch_weights"].shape == (batch, 3)
        assert out["feat_weights"].shape == (batch, n_features)

        # Direction should be in [0, 1] (sigmoid)
        assert (out["direction"] >= 0).all() and (out["direction"] <= 1).all()

    def test_multi_task_loss(self):
        """Test MultiTaskLoss computes without error."""
        import torch
        from backend.v2.model import MultiTaskLoss

        loss_fn = MultiTaskLoss()

        batch = 16
        preds = {
            "direction": torch.sigmoid(torch.randn(batch, 1)),
            "return_1d": torch.randn(batch, 1),
            "return_5d": torch.randn(batch, 1),
            "return_30d": torch.randn(batch, 1),
            "excess": torch.randn(batch, 1),
            "regime": torch.randn(batch, 3),
        }
        targets = {
            "direction": torch.randint(0, 2, (batch,)).float(),
            "return_1d": torch.randn(batch),
            "return_5d": torch.randn(batch),
            "return_30d": torch.randn(batch),
            "excess": torch.randn(batch),
            "regime": torch.randint(0, 3, (batch,)),
        }

        total, breakdown = loss_fn(preds, targets)
        assert total.shape == ()  # scalar
        assert total.item() > 0
        assert "direction" in breakdown or "total" in breakdown


# ═══════════════════════════════════════════════════════════════════════
# 5. Tree Models tests
# ═══════════════════════════════════════════════════════════════════════
class TestTreeModels:
    def test_flatten_sequences(self):
        from backend.v2.tree_models import flatten_sequences
        X = np.random.randn(50, 60, 10).astype(np.float32)
        feature_cols = [f"f{i}" for i in range(10)]
        flat = flatten_sequences(X, feature_cols)
        assert flat.shape[0] == 50
        assert flat.shape[1] == 30  # 10 features * 3 (last + mean + std)

    def test_lightgbm_train_predict(self):
        """Test LightGBM training and prediction (if installed)."""
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            pytest.skip("lightgbm not installed")

        from backend.v2.tree_models import train_lightgbm, predict_tree_models, flatten_sequences
        n = 200
        X = np.random.randn(n, 60, 10).astype(np.float32)
        y_dir = (np.random.rand(n) > 0.5).astype(np.float32)
        y_rets = np.random.randn(n, 4).astype(np.float32)
        feature_cols = [f"f{i}" for i in range(10)]

        models = train_lightgbm(X, y_dir, y_rets, feature_cols)
        assert models is not None
        assert "direction" in models

        preds = predict_tree_models({"lgbm": models}, X[:5], feature_cols)
        assert "lgbm" in preds
        assert "direction" in preds["lgbm"]


# ═══════════════════════════════════════════════════════════════════════
# 6. Meta-Router tests
# ═══════════════════════════════════════════════════════════════════════
class TestMetaRouter:
    def test_rule_based_weights(self):
        from backend.v2.meta_router import rule_based_weights
        w = rule_based_weights(0, uncertainty=0.1, vix_level=15)  # 0 = bull
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in w.values())

        w_bear = rule_based_weights(1, uncertainty=0.3, vix_level=35, has_lgbm=True)  # 1 = bear
        # Bear with tree models: tree models should get decent weight
        assert w_bear["lgbm"] > 0

    def test_combine_predictions(self):
        from backend.v2.meta_router import combine_predictions
        n = 10
        neural = {"direction": np.random.rand(n), "return_1d": np.random.randn(n)}
        lgbm = {"direction": np.random.rand(n), "return_1d": np.random.randn(n)}
        xgb = {"direction": np.random.rand(n), "return_1d": np.random.randn(n)}
        weights = {"neural": 0.5, "lgbm": 0.3, "xgb": 0.2}

        combined = combine_predictions(neural, lgbm, xgb, weights)
        assert "direction" in combined
        assert len(combined["direction"]) == n

    def test_compute_uncertainty(self):
        from backend.v2.meta_router import compute_uncertainty
        # Very confident → low uncertainty
        assert compute_uncertainty(0.99) < 0.1
        # 50/50 → high uncertainty
        assert compute_uncertainty(0.5) > 0.9


# ═══════════════════════════════════════════════════════════════════════
# 7. Calibration tests
# ═══════════════════════════════════════════════════════════════════════
class TestCalibration:
    def test_combined_calibrator(self):
        from backend.v2.calibration import CombinedCalibrator
        np.random.seed(42)
        n = 200
        raw_probs = np.random.rand(n)
        true_labels = (raw_probs + np.random.randn(n) * 0.2 > 0.5).astype(int)

        cal = CombinedCalibrator()
        cal.fit(raw_probs, true_labels)

        test_probs = np.array([0.1, 0.5, 0.9])
        calibrated = cal.predict(test_probs)
        assert len(calibrated) == 3
        assert all(0 <= p <= 1 for p in calibrated)

    def test_classification_metrics(self):
        from backend.v2.calibration import compute_classification_metrics
        n = 100
        probs = np.random.rand(n)
        labels = (np.random.rand(n) > 0.5).astype(int)
        result = compute_classification_metrics(labels, probs)
        # May be nested: {"raw": {...}, ...} or flat
        metrics = result.get("raw", result) if isinstance(result, dict) else result
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert "brier" in metrics or "brier_score" in metrics
        acc = metrics["accuracy"]
        assert 0 <= acc <= 1

    def test_concept_drift_detection(self):
        from backend.v2.calibration import detect_concept_drift
        # Perfect calibration → low Brier
        probs = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        labels = np.array([0, 1, 0, 1, 0])
        result = detect_concept_drift(probs, labels, baseline_brier=0.1)
        assert "drift_detected" in result


# ═══════════════════════════════════════════════════════════════════════
# 8. Backtest tests
# ═══════════════════════════════════════════════════════════════════════
class TestBacktest:
    def test_kelly_fraction(self):
        from backend.v2.backtest import kelly_fraction
        # Win rate 60%, avg win = avg loss → positive Kelly
        f = kelly_fraction(0.6, 0.02, 0.02, cap=0.1)
        assert 0 < f <= 0.1

        # Win rate 50%, equal payoffs → Kelly = 0 (clamped to min)
        f = kelly_fraction(0.5, 0.01, 0.01, cap=0.1)
        assert f >= 0.02  # min_fraction

    def test_run_backtest(self):
        from backend.v2.backtest import run_backtest
        np.random.seed(42)
        n = 250
        dates = pd.bdate_range("2023-01-01", periods=n)
        rets = np.random.randn(n) * 0.015
        probs = 0.5 + np.random.randn(n) * 0.15
        probs = np.clip(probs, 0.01, 0.99)

        result = run_backtest(dates, rets, probs)
        assert result.metrics is not None
        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert len(result.equity_curve) > 0
        assert len(result.trades) >= 0

    def test_compute_monthly_pnl(self):
        from backend.v2.backtest import compute_monthly_pnl
        n = 250
        dates = pd.bdate_range("2023-01-01", periods=n)
        equity = list(100 + np.cumsum(np.random.randn(n) * 0.5))
        monthly = compute_monthly_pnl(dates, equity)
        assert isinstance(monthly, dict)
        assert len(monthly) > 0


# ═══════════════════════════════════════════════════════════════════════
# 9. Integration smoke test
# ═══════════════════════════════════════════════════════════════════════
class TestIntegration:
    def test_full_pipeline_smoke(self):
        """
        End-to-end smoke test: synthetic data → features → model → predict.
        Uses tiny model to keep test fast.
        """
        import torch
        from backend.v2.model import HybridForecaster
        from backend.v2.features import add_technical_indicators, generate_targets

        np.random.seed(42)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 1.5)
        close = np.maximum(close, 10)
        df = pd.DataFrame({
            "Open": close * (1 + np.random.randn(n) * 0.005),
            "High": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "Low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        }, index=dates)

        # Feature engineering
        df = add_technical_indicators(df)
        df = generate_targets(df)
        df = df.dropna()

        assert len(df) > 50

        # Build tiny model
        n_features = 5  # Just OHLCV for smoke test
        model = HybridForecaster(
            n_features=n_features, d_model=32, n_heads=2,
            n_layers=1, dropout=0.1, n_stocks=1,
        )
        model.eval()

        # Forward pass
        seq_len = 30
        # Use any 5 numeric columns (feature names may have been altered by add_technical_indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5].tolist()
        assert len(numeric_cols) >= 5, f"Not enough numeric columns: {df.columns.tolist()}"
        data = df[numeric_cols].values[-seq_len:].astype(np.float32)
        x = torch.FloatTensor(data).unsqueeze(0)  # (1, seq_len, 5)
        stock_ids = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            out = model(x, stock_ids)

        assert "direction" in out
        assert "return_1d" in out
        assert out["direction"].shape == (1, 1)
        p_up = out["direction"].item()
        assert 0 <= p_up <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
