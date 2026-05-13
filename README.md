# Fraud Detection API

A production-ready **FastAPI** service for real-time and batch credit-card fraud detection, powered by an **XGBoost** classifier trained on the ULB Credit Card Fraud dataset.

---

## API Overview

All endpoints are versioned under `/api/v1`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Model health / readiness check |
| `POST` | `/api/v1/predictions` | Classify a single transaction |
| `POST` | `/api/v1/predictions/batch` | Classify up to 100 transactions |

Interactive docs (Swagger UI) → `http://localhost:8000/docs`

---

## Project Structure

```
Fraud-Classification-System/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── health.py        # GET /api/v1/health
│   │       ├── predictions.py   # POST /api/v1/predictions[/batch]
│   │       └── router.py        # v1 router aggregator
│   ├── models/
│   │   └── schema.py            # Pydantic request / response models
│   ├── services/
│   │   └── model_service.py     # Preprocessing + inference
│   └── main.py                  # FastAPI app factory + global error handlers
├── scripts/
│   └── train.py                 # Training pipeline
├── models/                      # Saved artifacts (gitignored)
├── data/                        # Raw CSV data (gitignored)
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_api.py              # Integration tests (40+ cases)
│   ├── test_service.py          # Unit tests – ModelService
│   └── test_schemas.py          # Unit tests – Pydantic schemas
├── requirements.txt
└── .env
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Place the `creditcard.csv` file in the `data/` directory.

### 3. Train the model
```bash
python scripts/train.py
```
Artifacts are saved to `models/`: `fraud_model.json`, `scaler.joblib`, `features.json`.

### 4. Start the server
```bash
uvicorn app.main:app --reload
```

---

## API Reference

### `GET /api/v1/health`

Returns model readiness.

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

| Field | Values |
|-------|--------|
| `status` | `"ok"` or `"degraded"` |
| `model_loaded` | `true` / `false` |

---

### `POST /api/v1/predictions`

Classify a single transaction.

**Request body:** 30 numeric fields — `Time`, `V1`–`V28`, `Amount`.

```json
{ "Time": 406.0, "V1": -2.31, ..., "Amount": 149.62 }
```

**Response `200 OK`:**
```json
{
  "data": {
    "is_fraud": false,
    "probability": 0.02,
    "model_version": "xgboost-v1"
  }
}
```

**Errors:**

| Status | Code | When |
|--------|------|------|
| `422` | `validation_error` | Missing / invalid field |
| `503` | `model_unavailable` | Model not trained yet |

---

### `POST /api/v1/predictions/batch`

Classify 1–100 transactions in one call.

**Request body:**
```json
{
  "transactions": [
    { "Time": 406.0, "V1": ..., "Amount": 149.62 },
    { "Time": 812.0, "V1": ..., "Amount": 0.0 }
  ]
}
```

**Response `200 OK`:**
```json
{
  "data": [
    { "is_fraud": false, "probability": 0.02, "model_version": "xgboost-v1" },
    { "is_fraud": true,  "probability": 0.94, "model_version": "xgboost-v1" }
  ],
  "meta": {
    "total": 2,
    "fraud_count": 1,
    "legitimate_count": 1
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

**71 tests** across three modules:

| Module | Focus |
|--------|-------|
| `test_schemas.py` | Pydantic validation rules |
| `test_service.py` | Preprocessing + inference logic |
| `test_api.py` | HTTP endpoints (happy path, 422, 503) |

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/fraud_model.json` | XGBoost model |
| `SCALER_PATH` | `models/scaler.joblib` | RobustScaler artifact |
| `FEATURES_PATH` | `models/features.json` | Feature column order |
| `DATA_PATH` | `data/creditcard.csv` | Training data |
