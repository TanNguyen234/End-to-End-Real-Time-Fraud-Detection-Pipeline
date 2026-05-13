import app.main as app_module
from fastapi import APIRouter
from app.models.schema import HealthResponse

router = APIRouter()

APP_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the operational status of the API and whether the model is loaded.",
)
async def health_check() -> HealthResponse:
    svc = app_module.model_service
    model_loaded = svc is not None and svc.is_ready
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=APP_VERSION,
    )
