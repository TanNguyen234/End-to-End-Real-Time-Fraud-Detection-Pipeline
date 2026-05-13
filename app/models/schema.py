from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

# ─── Request Schemas ───────────────────────────────────────────────────────────

_V_FIELD = {"default": ..., "description": "PCA-transformed feature."}

class FraudPredictionRequest(BaseModel):
    """Single-transaction fraud prediction request."""

    Time: float = Field(..., ge=0, description="Seconds elapsed since the first transaction in the dataset.")
    V1: float = Field(**_V_FIELD)
    V2: float = Field(**_V_FIELD)
    V3: float = Field(**_V_FIELD)
    V4: float = Field(**_V_FIELD)
    V5: float = Field(**_V_FIELD)
    V6: float = Field(**_V_FIELD)
    V7: float = Field(**_V_FIELD)
    V8: float = Field(**_V_FIELD)
    V9: float = Field(**_V_FIELD)
    V10: float = Field(**_V_FIELD)
    V11: float = Field(**_V_FIELD)
    V12: float = Field(**_V_FIELD)
    V13: float = Field(**_V_FIELD)
    V14: float = Field(**_V_FIELD)
    V15: float = Field(**_V_FIELD)
    V16: float = Field(**_V_FIELD)
    V17: float = Field(**_V_FIELD)
    V18: float = Field(**_V_FIELD)
    V19: float = Field(**_V_FIELD)
    V20: float = Field(**_V_FIELD)
    V21: float = Field(**_V_FIELD)
    V22: float = Field(**_V_FIELD)
    V23: float = Field(**_V_FIELD)
    V24: float = Field(**_V_FIELD)
    V25: float = Field(**_V_FIELD)
    V26: float = Field(**_V_FIELD)
    V27: float = Field(**_V_FIELD)
    V28: float = Field(**_V_FIELD)
    Amount: float = Field(..., ge=0, description="Transaction amount in currency units.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class BatchFraudPredictionRequest(BaseModel):
    """Batch of transactions for bulk fraud prediction (max 100)."""

    transactions: List[FraudPredictionRequest] = Field(
        ..., min_length=1, max_length=100,
        description="List of transactions to evaluate (1–100)."
    )


# ─── Response Schemas ──────────────────────────────────────────────────────────

class FraudPredictionResult(BaseModel):
    """Prediction result for a single transaction."""

    is_fraud: bool = Field(..., description="True when the transaction is classified as fraudulent.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score in [0, 1].")
    model_version: str = Field(..., description="Identifier of the model that produced this prediction.")


class FraudPredictionResponse(BaseModel):
    """Envelope for a single-transaction prediction."""

    data: FraudPredictionResult


class BatchFraudPredictionResponse(BaseModel):
    """Envelope for batch predictions."""

    data: List[FraudPredictionResult]
    meta: dict = Field(
        ...,
        description="Batch statistics: total, fraud_count, legitimate_count.",
    )


# ─── Health / Info Schemas ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health check payload."""

    status: str = Field(..., description="'ok' or 'degraded'.")
    model_loaded: bool
    version: str


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    code: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: dict  # {"code": str, "message": str, "details": list[ErrorDetail]}
