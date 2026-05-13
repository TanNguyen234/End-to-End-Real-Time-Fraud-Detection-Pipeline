import logging
from typing import Any

import app.main as app_module
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.models.schema import (
    BatchFraudPredictionRequest,
    BatchFraudPredictionResponse,
    FraudPredictionRequest,
    FraudPredictionResponse,
    FraudPredictionResult,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MODEL_VERSION = "xgboost-v1"


def _service_unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": {
                "code": "model_unavailable",
                "message": "The ML model is not loaded. Run scripts/train.py first.",
            }
        },
    )


# ── POST /api/v1/predictions ──────────────────────────────────────────────────


@router.post(
    "",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict single transaction",
    description=(
        "Accepts one credit-card transaction and returns a fraud classification "
        "with a probability score."
    ),
    responses={
        422: {"description": "Validation error – check field constraints."},
        503: {"description": "Model not loaded."},
    },
)
async def predict(request: FraudPredictionRequest) -> Any:
    svc = app_module.model_service
    if svc is None or not svc.is_ready:
        return _service_unavailable()

    is_fraud, probability = svc.predict(request)
    return FraudPredictionResponse(
        data=FraudPredictionResult(
            is_fraud=is_fraud,
            probability=probability,
            model_version=MODEL_VERSION,
        )
    )


# ── POST /api/v1/predictions/batch ────────────────────────────────────────────


@router.post(
    "/batch",
    response_model=BatchFraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict batch of transactions",
    description=(
        "Accepts a list of up to 100 transactions and returns a prediction for each, "
        "plus aggregate statistics."
    ),
    responses={
        422: {"description": "Validation error – check field constraints or batch size."},
        503: {"description": "Model not loaded."},
    },
)
async def predict_batch(request: BatchFraudPredictionRequest) -> Any:
    svc = app_module.model_service
    if svc is None or not svc.is_ready:
        return _service_unavailable()

    results_raw = svc.predict_batch(request.transactions)
    results = [
        FraudPredictionResult(
            is_fraud=is_fraud,
            probability=prob,
            model_version=MODEL_VERSION,
        )
        for is_fraud, prob in results_raw
    ]
    fraud_count = sum(r.is_fraud for r in results)
    return BatchFraudPredictionResponse(
        data=results,
        meta={
            "total": len(results),
            "fraud_count": fraud_count,
            "legitimate_count": len(results) - fraud_count,
        },
    )
