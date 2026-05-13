import pytest
import pandas as pd
import numpy as np
from app.services.model_service import ModelService
from app.models.schema import FraudPredictionRequest
from unittest.mock import MagicMock

@pytest.fixture
def model_service():
    service = ModelService()
    # Mock the scaler
    service.scaler = MagicMock()
    service.scaler.transform.return_value = np.array([[0.5]])
    # Mock the features
    service.features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    # Mock the model
    service.model = MagicMock()
    service.model.predict.return_value = np.array([0])
    service.model.predict_proba.return_value = np.array([[0.9, 0.1]])
    return service

def test_preprocess(model_service):
    request = FraudPredictionRequest(
        Time=3600.0, # 1 hour
        V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0, V8=0, V9=0, V10=0,
        V11=0, V12=0, V13=0, V14=0, V15=0, V16=0, V17=0, V18=0, V19=0, V20=0,
        V21=0, V22=0, V23=0, V24=0, V25=0, V26=0, V27=0, V28=0,
        Amount=100.0
    )
    
    df = model_service.preprocess(request)
    
    assert df['Time'].iloc[0] == 1.0 # 3600 / 3600 % 24
    assert df['Amount'].iloc[0] == 0.5 # Mocked value
    assert len(df.columns) == 30

def test_predict(model_service):
    request = FraudPredictionRequest(
        Time=0, V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0, V8=0, V9=0, V10=0,
        V11=0, V12=0, V13=0, V14=0, V15=0, V16=0, V17=0, V18=0, V19=0, V20=0,
        V21=0, V22=0, V23=0, V24=0, V25=0, V26=0, V27=0, V28=0,
        Amount=0
    )
    
    is_fraud, probability = model_service.predict(request)
    
    assert is_fraud is False
    assert probability == 0.1
