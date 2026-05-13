import os
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from app.models.schema import FraudPredictionRequest, FraudPredictionResponse
from app.services.model_service import ModelService
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model service instance
model_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model_service
    model_service = ModelService()
    if model_service.model is None:
        logger.error("Failed to load model on startup.")
    else:
        logger.info("Model service initialized.")
    yield
    # Clean up (if needed)
    logger.info("Shutting down...")

app = FastAPI(
    title="Fraud Classification API",
    description="API for detecting fraudulent credit card transactions.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Fraud Classification API is running", "status": "ok"}

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict(request: FraudPredictionRequest):
    """
    Predict if a transaction is fraudulent.
    """
    if model_service is None or model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable.")
    
    try:
        is_fraud, probability = model_service.predict(request)
        return FraudPredictionResponse(
            is_fraud=is_fraud,
            probability=probability,
            model_version="xgboost-v1"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
