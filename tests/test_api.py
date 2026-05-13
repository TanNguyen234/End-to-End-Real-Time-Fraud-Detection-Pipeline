"""Integration tests for all FastAPI endpoints (v1)."""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import app.main as app_module
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_ready_service(predict_return=(False, 0.05), batch_return=None):
    """Return a mock ModelService that is ready and returns controlled values."""
    svc = MagicMock()
    svc.is_ready = True
    svc.predict.return_value = predict_return
    if batch_return is None:
        batch_return = [predict_return]
    svc.predict_batch.return_value = batch_return
    return svc


def _unavailable_service():
    svc = MagicMock()
    svc.is_ready = False
    return svc


VALID_PAYLOAD = {
    "Time": 406.0,
    "V1": -2.31222654232745, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.99790558754539,
    "V5": -0.522187864667764, "V6": -1.42654531920595,
    "V7": -2.53738733740133, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915,
    "V11": 3.20203320709635, "V12": -2.89990738849473,
    "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657,
    "V17": -2.83005567450437, "V18": -0.0168224681808257,
    "V19": 0.416955705175855, "V20": 0.126910559061474,
    "V21": 0.517232370861764, "V22": -0.0350493686052974,
    "V23": -0.465211076182388, "V24": 0.320198199234528,
    "V25": 0.0445191674731724, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919,
    "Amount": 149.62,
}


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/v1/health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_ok_when_model_ready(self):
        app_module.model_service = _mock_ready_service()
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "version" in body

    def test_health_degraded_when_model_not_ready(self):
        app_module.model_service = _unavailable_service()
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_health_degraded_when_service_is_none(self):
        app_module.model_service = None
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/v1/predictions  (single)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictEndpoint:
    # ── happy path ──────────────────────────────────────────────────────────

    def test_returns_200_for_valid_payload(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_envelope_has_data_key(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions", json=VALID_PAYLOAD)
        assert "data" in r.json()

    def test_not_fraud_prediction(self):
        app_module.model_service = _mock_ready_service(predict_return=(False, 0.05))
        body = client.post("/api/v1/predictions", json=VALID_PAYLOAD).json()
        assert body["data"]["is_fraud"] is False
        assert body["data"]["probability"] == pytest.approx(0.05)

    def test_fraud_prediction(self):
        app_module.model_service = _mock_ready_service(predict_return=(True, 0.94))
        body = client.post("/api/v1/predictions", json=VALID_PAYLOAD).json()
        assert body["data"]["is_fraud"] is True
        assert body["data"]["probability"] == pytest.approx(0.94)

    def test_model_version_present(self):
        app_module.model_service = _mock_ready_service()
        body = client.post("/api/v1/predictions", json=VALID_PAYLOAD).json()
        assert body["data"]["model_version"] == "xgboost-v1"

    def test_probability_in_range(self):
        app_module.model_service = _mock_ready_service(predict_return=(False, 0.12))
        body = client.post("/api/v1/predictions", json=VALID_PAYLOAD).json()
        prob = body["data"]["probability"]
        assert 0.0 <= prob <= 1.0

    # ── model unavailable ───────────────────────────────────────────────────

    def test_503_when_service_none(self):
        app_module.model_service = None
        r = client.post("/api/v1/predictions", json=VALID_PAYLOAD)
        assert r.status_code == 503

    def test_503_when_model_not_ready(self):
        app_module.model_service = _unavailable_service()
        r = client.post("/api/v1/predictions", json=VALID_PAYLOAD)
        assert r.status_code == 503

    def test_503_error_envelope(self):
        app_module.model_service = None
        r = client.post("/api/v1/predictions", json=VALID_PAYLOAD)
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "model_unavailable"

    # ── validation errors ───────────────────────────────────────────────────

    def test_422_for_missing_required_field(self):
        app_module.model_service = _mock_ready_service()
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Amount"}
        r = client.post("/api/v1/predictions", json=payload)
        assert r.status_code == 422

    def test_422_for_negative_amount(self):
        app_module.model_service = _mock_ready_service()
        bad = {**VALID_PAYLOAD, "Amount": -1.0}
        r = client.post("/api/v1/predictions", json=bad)
        assert r.status_code == 422

    def test_422_for_negative_time(self):
        app_module.model_service = _mock_ready_service()
        bad = {**VALID_PAYLOAD, "Time": -5.0}
        r = client.post("/api/v1/predictions", json=bad)
        assert r.status_code == 422

    def test_422_error_envelope_has_error_key(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions", json={})
        assert r.status_code == 422
        assert "error" in r.json()

    def test_422_details_identify_bad_fields(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions", json={})
        details = r.json()["error"]["details"]
        field_names = [d["field"] for d in details]
        assert any("Time" in f or "V1" in f or "Amount" in f for f in field_names)

    def test_422_for_non_numeric_field(self):
        app_module.model_service = _mock_ready_service()
        bad = {**VALID_PAYLOAD, "V1": "not-a-number"}
        r = client.post("/api/v1/predictions", json=bad)
        assert r.status_code == 422

    def test_422_for_empty_body(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions", json={})
        assert r.status_code == 422

    # ── wrong methods ────────────────────────────────────────────────────────

    def test_get_method_not_allowed(self):
        r = client.get("/api/v1/predictions")
        assert r.status_code == 405


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/v1/predictions/batch
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictBatchEndpoint:
    BATCH_PAYLOAD = {"transactions": [VALID_PAYLOAD, VALID_PAYLOAD]}

    # ── happy path ──────────────────────────────────────────────────────────

    def test_returns_200_for_valid_batch(self):
        app_module.model_service = _mock_ready_service(
            batch_return=[(False, 0.05), (False, 0.07)]
        )
        r = client.post("/api/v1/predictions/batch", json=self.BATCH_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_data_list(self):
        app_module.model_service = _mock_ready_service(
            batch_return=[(False, 0.05), (True, 0.92)]
        )
        body = client.post("/api/v1/predictions/batch", json=self.BATCH_PAYLOAD).json()
        assert isinstance(body["data"], list)
        assert len(body["data"]) == 2

    def test_response_has_meta(self):
        app_module.model_service = _mock_ready_service(
            batch_return=[(False, 0.05), (True, 0.92)]
        )
        body = client.post("/api/v1/predictions/batch", json=self.BATCH_PAYLOAD).json()
        meta = body["meta"]
        assert meta["total"] == 2
        assert meta["fraud_count"] == 1
        assert meta["legitimate_count"] == 1

    def test_meta_all_legitimate(self):
        app_module.model_service = _mock_ready_service(
            batch_return=[(False, 0.03), (False, 0.04)]
        )
        body = client.post("/api/v1/predictions/batch", json=self.BATCH_PAYLOAD).json()
        assert body["meta"]["fraud_count"] == 0
        assert body["meta"]["legitimate_count"] == 2

    def test_meta_all_fraud(self):
        app_module.model_service = _mock_ready_service(
            batch_return=[(True, 0.91), (True, 0.87)]
        )
        body = client.post("/api/v1/predictions/batch", json=self.BATCH_PAYLOAD).json()
        assert body["meta"]["fraud_count"] == 2
        assert body["meta"]["legitimate_count"] == 0

    def test_single_transaction_batch(self):
        app_module.model_service = _mock_ready_service(batch_return=[(False, 0.05)])
        payload = {"transactions": [VALID_PAYLOAD]}
        body = client.post("/api/v1/predictions/batch", json=payload).json()
        assert body["meta"]["total"] == 1

    # ── model unavailable ───────────────────────────────────────────────────

    def test_503_when_service_none(self):
        app_module.model_service = None
        r = client.post("/api/v1/predictions/batch", json=self.BATCH_PAYLOAD)
        assert r.status_code == 503

    # ── validation errors ───────────────────────────────────────────────────

    def test_422_for_empty_transactions_list(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions/batch", json={"transactions": []})
        assert r.status_code == 422

    def test_422_for_batch_exceeding_100(self):
        app_module.model_service = _mock_ready_service()
        oversized = {"transactions": [VALID_PAYLOAD] * 101}
        r = client.post("/api/v1/predictions/batch", json=oversized)
        assert r.status_code == 422

    def test_422_for_missing_transactions_key(self):
        app_module.model_service = _mock_ready_service()
        r = client.post("/api/v1/predictions/batch", json={})
        assert r.status_code == 422

    def test_422_for_invalid_transaction_in_batch(self):
        app_module.model_service = _mock_ready_service()
        bad = {"transactions": [{**VALID_PAYLOAD, "Amount": -1}]}
        r = client.post("/api/v1/predictions/batch", json=bad)
        assert r.status_code == 422

    # ── wrong methods ────────────────────────────────────────────────────────

    def test_get_method_not_allowed(self):
        r = client.get("/api/v1/predictions/batch")
        assert r.status_code == 405
