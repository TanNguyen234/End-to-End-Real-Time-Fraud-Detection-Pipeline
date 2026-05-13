from fastapi import APIRouter

from app.api.v1 import health, predictions

router = APIRouter()
router.include_router(health.router, tags=["Health"])
router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
