import os
import json
import logging
from typing import List, Tuple

import joblib
import pandas as pd
import xgboost as xgb

from app.models.schema import FraudPredictionRequest

logger = logging.getLogger(__name__)

MODEL_VERSION = "xgboost-v1"


class ModelService:
    """Encapsulates model loading, preprocessing and inference."""

    def __init__(self):
        self.model: xgb.XGBClassifier | None = None
        self.scaler = None
        self.features: List[str] | None = None
        self.model_version: str = MODEL_VERSION

        self.model_path = os.getenv("MODEL_PATH", "models/fraud_model.json")
        self.scaler_path = os.getenv("SCALER_PATH", "models/scaler.joblib")
        self.features_path = os.getenv("FEATURES_PATH", "models/features.json")

        self.load_artifacts()

    # ── Artifact management ───────────────────────────────────────────────────

    def load_artifacts(self) -> None:
        """Load model, scaler, and feature list from disk."""
        try:
            if os.path.exists(self.model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info("Model loaded from %s", self.model_path)
            else:
                logger.warning("Model file not found at %s", self.model_path)

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Scaler loaded from %s", self.scaler_path)
            else:
                logger.warning("Scaler file not found at %s", self.scaler_path)

            if os.path.exists(self.features_path):
                with open(self.features_path, "r") as fh:
                    self.features = json.load(fh)
                logger.info("Feature list loaded from %s", self.features_path)
        except Exception as exc:
            logger.error("Error loading model artifacts: %s", exc)

    @property
    def is_ready(self) -> bool:
        """True when the model is loaded and ready for inference."""
        return self.model is not None

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, request: FraudPredictionRequest) -> pd.DataFrame:
        """Apply the same preprocessing steps used during training.

        1. Convert raw *Time* (seconds since epoch) → hour-of-day.
        2. Apply ``RobustScaler`` to *Amount*.
        3. Reorder columns to match the training feature order.
        """
        data = request.model_dump()
        df = pd.DataFrame([data])

        # 1. Time → hour-of-day (cyclic, 0–24)
        df["Time"] = (df["Time"] / 3600) % 24

        # 2. Amount scaling
        if self.scaler is not None:
            df["Amount"] = self.scaler.transform(df[["Amount"]])
        else:
            logger.warning("Scaler not loaded – Amount not scaled.")

        # 3. Enforce column order
        if self.features:
            df = df[self.features]

        return df

    def preprocess_batch(self, requests: List[FraudPredictionRequest]) -> pd.DataFrame:
        """Vectorised preprocessing for a list of transactions."""
        rows = [r.model_dump() for r in requests]
        df = pd.DataFrame(rows)

        df["Time"] = (df["Time"] / 3600) % 24

        if self.scaler is not None:
            df["Amount"] = self.scaler.transform(df[["Amount"]])
        else:
            logger.warning("Scaler not loaded – Amount not scaled.")

        if self.features:
            df = df[self.features]

        return df

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, request: FraudPredictionRequest) -> Tuple[bool, float]:
        """Return (is_fraud, probability) for a single transaction."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded.")

        df = self.preprocess(request)
        proba = float(self.model.predict_proba(df)[0][1])
        label = bool(int(self.model.predict(df)[0]))
        return label, proba

    def predict_batch(
        self, requests: List[FraudPredictionRequest]
    ) -> List[Tuple[bool, float]]:
        """Return a list of (is_fraud, probability) tuples for a batch."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded.")

        df = self.preprocess_batch(requests)
        probas = self.model.predict_proba(df)[:, 1]
        labels = self.model.predict(df).astype(bool)
        return [(bool(lbl), float(prob)) for lbl, prob in zip(labels, probas)]
