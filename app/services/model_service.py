import os
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from app.models.schema import FraudPredictionRequest
import logging

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.model_path = os.getenv("MODEL_PATH", "models/fraud_model.json")
        self.scaler_path = os.getenv("SCALER_PATH", "models/scaler.joblib")
        self.features_path = os.getenv("FEATURES_PATH", "models/features.json")
        self.load_model()

    def load_model(self):
        """Load model, scaler and feature list from disk."""
        try:
            if os.path.exists(self.model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.warning(f"Scaler file not found at {self.scaler_path}")

            if os.path.exists(self.features_path):
                import json
                with open(self.features_path, 'r') as f:
                    self.features = json.load(f)
                logger.info(f"Features list loaded from {self.features_path}")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")

    def preprocess(self, request: FraudPredictionRequest) -> pd.DataFrame:
        """
        Apply the same preprocessing steps as used during training.
        """
        data = request.model_dump()
        df = pd.DataFrame([data])
        
        # 1. Time transformation: convert seconds to hour of day
        df['Time'] = (df['Time'] / 3600) % 24
        
        # 2. Amount scaling
        if self.scaler:
            df['Amount'] = self.scaler.transform(df['Amount'].values.reshape(-1, 1))
        else:
            logger.warning("Scaler not loaded, skipping Amount scaling.")
            
        # Ensure feature order matches training
        if self.features:
            df = df[self.features]
            
        return df

    def predict(self, request: FraudPredictionRequest):
        """
        Make a prediction.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
            
        input_df = self.preprocess(request)
        
        # XGBoost prediction
        # predict() returns class (0 or 1)
        # predict_proba() returns probabilities for each class
        prob = self.model.predict_proba(input_df)[0][1]
        prediction = int(self.model.predict(input_df)[0])
        
        return bool(prediction), float(prob)
