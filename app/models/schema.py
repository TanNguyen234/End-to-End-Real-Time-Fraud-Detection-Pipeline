from pydantic import BaseModel, Field
from typing import List

class FraudPredictionRequest(BaseModel):
    Time: float = Field(..., description="Number of seconds elapsed between this transaction and the first transaction in the dataset.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount.")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0,
                "V1": -2.31222654232745,
                "V2": 1.95199201064158,
                "V3": -1.60985073229769,
                "V4": 3.99790558754539,
                "V5": -0.522187864667764,
                "V6": -1.42654531920595,
                "V7": -2.53738733740133,
                "V8": 1.39165724829804,
                "V9": -2.77008927719433,
                "V10": -2.77227214465915,
                "V11": 3.20203320709635,
                "V12": -2.89990738849473,
                "V13": -0.595221881324605,
                "V14": -4.28925378244217,
                "V15": 0.389724120274487,
                "V16": -1.14074717980657,
                "V17": -2.83005567450437,
                "V18": -0.0168224681808257,
                "V19": 0.416955705175855,
                "V20": 0.126910559061474,
                "V21": 0.517232370861764,
                "V22": -0.0350493686052974,
                "V23": -0.465211076182388,
                "V24": 0.320198199234528,
                "V25": 0.0445191674731724,
                "V26": 0.177839798284401,
                "V27": 0.261145002567677,
                "V28": -0.143275874698919,
                "Amount": 0.0
            }
        }

class FraudPredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    model_version: str
