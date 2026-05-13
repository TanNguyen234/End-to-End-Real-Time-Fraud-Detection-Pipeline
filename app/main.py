import logging
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.services.model_service import ModelService

# ── Bootstrap ─────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level reference kept for testability (can be patched in unit tests)
model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    model_service = ModelService()
    if model_service.is_ready:
        logger.info("Model service ready.")
    else:
        logger.warning("Model NOT loaded – /predict will return 503.")
    yield
    logger.info("Shutdown complete.")


# ── Application factory ───────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time and batch credit-card fraud detection.\n\n"
        "All endpoints are versioned under `/api/v1`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Global exception handlers ─────────────────────────────────────────────────


def _error_body(code: str, message: str, details: list | None = None) -> dict:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(loc) for loc in err["loc"] if loc != "body"),
            "message": err["msg"],
            "code": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body(
            code="validation_error",
            message="Request validation failed.",
            details=details,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(
            code="internal_error",
            message="An unexpected error occurred.",
        ),
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(v1_router, prefix="/api/v1")


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
