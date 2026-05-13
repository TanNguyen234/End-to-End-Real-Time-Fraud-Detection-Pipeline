"""Unit tests for ModelService – preprocessing and prediction logic."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.models.schema import FraudPredictionRequest
from app.services.model_service import ModelService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(*, predict_label=0, predict_proba=0.05) -> ModelService:
    """Return a ModelService whose heavy artifacts are fully mocked."""
    svc = ModelService.__new__(ModelService)
    svc.model_version = "xgboost-v1"
    svc.model_path = "models/fraud_model.json"
    svc.scaler_path = "models/scaler.joblib"
    svc.features_path = "models/features.json"

    # Mock scaler – always returns 0.5
    svc.scaler = MagicMock()
    svc.scaler.transform.return_value = np.array([[0.5]])

    # Feature list matching training order
    svc.features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Mock XGBoost model
    svc.model = MagicMock()
    svc.model.predict.return_value = np.array([predict_label])
    svc.model.predict_proba.return_value = np.array([[1 - predict_proba, predict_proba]])

    return svc


def _zero_request(**overrides) -> FraudPredictionRequest:
    base = {f: 0.0 for f in ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]}
    base.update(overrides)
    return FraudPredictionRequest(**base)


# ── is_ready ──────────────────────────────────────────────────────────────────

class TestIsReady:
    def test_ready_when_model_loaded(self):
        svc = _make_service()
        assert svc.is_ready is True

    def test_not_ready_when_model_none(self):
        svc = _make_service()
        svc.model = None
        assert svc.is_ready is False


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_time_is_converted_to_hour_of_day(self):
        svc = _make_service()
        req = _zero_request(Time=3600.0)  # 1 hour → 1.0
        df = svc.preprocess(req)
        assert df["Time"].iloc[0] == pytest.approx(1.0)

    def test_time_wraps_at_24h(self):
        svc = _make_service()
        req = _zero_request(Time=86400.0)  # exactly 24h → 0.0
        df = svc.preprocess(req)
        assert df["Time"].iloc[0] == pytest.approx(0.0)

    def test_time_midnight_stays_zero(self):
        svc = _make_service()
        req = _zero_request(Time=0.0)
        df = svc.preprocess(req)
        assert df["Time"].iloc[0] == pytest.approx(0.0)

    def test_amount_is_scaled(self):
        svc = _make_service()
        req = _zero_request(Amount=100.0)
        df = svc.preprocess(req)
        assert df["Amount"].iloc[0] == pytest.approx(0.5)

    def test_scaler_called_with_amount(self):
        svc = _make_service()
        req = _zero_request(Amount=200.0)
        svc.preprocess(req)
        svc.scaler.transform.assert_called_once()

    def test_feature_columns_match_training_order(self, sample_request):
        svc = _make_service()
        df = svc.preprocess(sample_request)
        assert list(df.columns) == svc.features

    def test_output_is_single_row(self, sample_request):
        svc = _make_service()
        df = svc.preprocess(sample_request)
        assert len(df) == 1

    def test_preprocess_without_scaler_skips_scaling(self):
        svc = _make_service()
        svc.scaler = None
        req = _zero_request(Amount=100.0)
        # Should not raise
        df = svc.preprocess(req)
        assert "Amount" in df.columns


# ── preprocess_batch ──────────────────────────────────────────────────────────

class TestPreprocessBatch:
    def test_batch_returns_correct_row_count(self, sample_request):
        svc = _make_service()
        svc.scaler.transform.return_value = np.array([[0.5], [0.5]])
        df = svc.preprocess_batch([sample_request, sample_request])
        assert len(df) == 2

    def test_batch_time_conversion(self):
        svc = _make_service()
        r1 = _zero_request(Time=3600.0)
        r2 = _zero_request(Time=7200.0)
        svc.scaler.transform.return_value = np.array([[0.5], [0.5]])
        df = svc.preprocess_batch([r1, r2])
        assert df["Time"].iloc[0] == pytest.approx(1.0)
        assert df["Time"].iloc[1] == pytest.approx(2.0)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_not_fraud(self, sample_request):
        svc = _make_service(predict_label=0, predict_proba=0.05)
        is_fraud, prob = svc.predict(sample_request)
        assert is_fraud is False
        assert prob == pytest.approx(0.05)

    def test_predict_returns_fraud(self, sample_request):
        svc = _make_service(predict_label=1, predict_proba=0.95)
        is_fraud, prob = svc.predict(sample_request)
        assert is_fraud is True
        assert prob == pytest.approx(0.95)

    def test_predict_raises_when_model_not_loaded(self, sample_request):
        svc = _make_service()
        svc.model = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            svc.predict(sample_request)

    def test_probability_is_float(self, sample_request):
        svc = _make_service()
        _, prob = svc.predict(sample_request)
        assert isinstance(prob, float)

    def test_is_fraud_is_bool(self, sample_request):
        svc = _make_service()
        is_fraud, _ = svc.predict(sample_request)
        assert isinstance(is_fraud, bool)


# ── predict_batch ─────────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_length_matches_input(self, sample_request):
        svc = _make_service()
        svc.model.predict.return_value = np.array([0, 1])
        svc.model.predict_proba.return_value = np.array([[0.95, 0.05], [0.10, 0.90]])
        svc.scaler.transform.return_value = np.array([[0.5], [0.5]])
        results = svc.predict_batch([sample_request, sample_request])
        assert len(results) == 2

    def test_batch_raises_when_model_not_loaded(self, sample_request):
        svc = _make_service()
        svc.model = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            svc.predict_batch([sample_request])
