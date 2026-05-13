"""Tests for Pydantic schema validation (schema.py)."""
import pytest
from pydantic import ValidationError

from app.models.schema import (
    FraudPredictionRequest,
    BatchFraudPredictionRequest,
    FraudPredictionResult,
    FraudPredictionResponse,
    BatchFraudPredictionResponse,
    HealthResponse,
)

VALID_FIELDS = {
    "Time": 406.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "Amount": 149.62,
}


class TestFraudPredictionRequest:
    def test_valid_payload_accepted(self):
        req = FraudPredictionRequest(**VALID_FIELDS)
        assert req.Time == 406.0
        assert req.Amount == 149.62

    def test_missing_time_raises(self):
        bad = {k: v for k, v in VALID_FIELDS.items() if k != "Time"}
        with pytest.raises(ValidationError):
            FraudPredictionRequest(**bad)

    def test_missing_amount_raises(self):
        bad = {k: v for k, v in VALID_FIELDS.items() if k != "Amount"}
        with pytest.raises(ValidationError):
            FraudPredictionRequest(**bad)

    def test_negative_time_raises(self):
        with pytest.raises(ValidationError):
            FraudPredictionRequest(**{**VALID_FIELDS, "Time": -1.0})

    def test_zero_time_is_valid(self):
        req = FraudPredictionRequest(**{**VALID_FIELDS, "Time": 0.0})
        assert req.Time == 0.0

    def test_negative_amount_raises(self):
        with pytest.raises(ValidationError):
            FraudPredictionRequest(**{**VALID_FIELDS, "Amount": -0.01})

    def test_zero_amount_is_valid(self):
        req = FraudPredictionRequest(**{**VALID_FIELDS, "Amount": 0.0})
        assert req.Amount == 0.0

    def test_string_in_numeric_field_raises(self):
        with pytest.raises(ValidationError):
            FraudPredictionRequest(**{**VALID_FIELDS, "V1": "bad"})

    def test_all_30_fields_present(self):
        req = FraudPredictionRequest(**VALID_FIELDS)
        d = req.model_dump()
        assert len(d) == 30  # Time + V1–V28 + Amount


class TestBatchFraudPredictionRequest:
    def _make_req(self):
        return FraudPredictionRequest(**VALID_FIELDS)

    def test_single_transaction_accepted(self):
        req = BatchFraudPredictionRequest(transactions=[self._make_req()])
        assert len(req.transactions) == 1

    def test_100_transactions_accepted(self):
        reqs = [self._make_req()] * 100
        batch = BatchFraudPredictionRequest(transactions=reqs)
        assert len(batch.transactions) == 100

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError):
            BatchFraudPredictionRequest(transactions=[])

    def test_101_transactions_raises(self):
        reqs = [self._make_req()] * 101
        with pytest.raises(ValidationError):
            BatchFraudPredictionRequest(transactions=reqs)


class TestFraudPredictionResult:
    def test_valid_result_accepted(self):
        r = FraudPredictionResult(is_fraud=False, probability=0.05, model_version="xgboost-v1")
        assert r.is_fraud is False

    def test_probability_below_zero_raises(self):
        with pytest.raises(ValidationError):
            FraudPredictionResult(is_fraud=False, probability=-0.01, model_version="v1")

    def test_probability_above_one_raises(self):
        with pytest.raises(ValidationError):
            FraudPredictionResult(is_fraud=True, probability=1.01, model_version="v1")

    def test_probability_boundary_zero(self):
        r = FraudPredictionResult(is_fraud=False, probability=0.0, model_version="v1")
        assert r.probability == 0.0

    def test_probability_boundary_one(self):
        r = FraudPredictionResult(is_fraud=True, probability=1.0, model_version="v1")
        assert r.probability == 1.0


class TestHealthResponse:
    def test_ok_status(self):
        h = HealthResponse(status="ok", model_loaded=True, version="1.0.0")
        assert h.status == "ok"

    def test_degraded_status(self):
        h = HealthResponse(status="degraded", model_loaded=False, version="1.0.0")
        assert h.model_loaded is False
