import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent

from backend.config import settings
from backend.core.logging import configure_logging
from backend.core.lifecycle import (
    check_database,
    initialise_database,
    provision_admin,
    warm_up_models,
)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.auth import hash_password
from backend.auth import router as auth_router
from backend.database import SessionLocal
from backend.dependencies import get_analysis_service
from backend.middleware import (
    RequestBodyGuardMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from backend.models import User
from backend.rate_limit import limiter
from backend.routers.admin import router as admin_router
from backend.routers.analysis import router as analysis_router
from backend.routers.health import create_health_router



configure_logging()
logger = logging.getLogger("fake_job_detection_api")

CORS_ORIGINS = settings.cors_origin_list

# Resolve the singleton once at import time for lifespan + health checks.
analysis_service = get_analysis_service()


def _init_database() -> None:
    initialise_database(settings.app_env, BACKEND_DIR / "alembic.ini")


def _provision_admin() -> None:
    provision_admin(
        app_env=settings.app_env,
        email=settings.admin_email,
        username=settings.admin_username,
        password=settings.admin_password,
        session_factory=SessionLocal,
        user_model=User,
        hash_password=hash_password,
        logger=logger,
    )


def _check_database() -> bool:
    return check_database(SessionLocal)


def _warm_up_models_background() -> None:
    warm_up_models(analysis_service, logger)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _init_database()
        _provision_admin()
    except Exception:
        logger.exception("Database initialisation failed.")
        if settings.app_env == "production":
            raise

    # Start model warm-up in background
    if settings.app_env != "test":
        threading.Thread(target=_warm_up_models_background, daemon=True).start()
    yield



# App factory

app = FastAPI(
    title="Fake Job Detection API",
    description=(
        "Provides a backend ensemble risk result, model-derived XAI evidence, "
        "and gentle educational guidance for English job advertisements."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# Rate limiter attach state + exception handler
app.state.limiter = limiter
# SlowAPI's handler is runtime-compatible with Starlette, but its published
# callable annotation narrows the exception parameter to RateLimitExceeded.
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type]
)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    public_errors = [
        {
            "type": error.get("type", "validation_error"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "Invalid value."),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": public_errors,
            "error": {
                "code": "REQUEST_VALIDATION_FAILED",
                "message": "The request payload is invalid.",
                "request_id": request_id,
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled request failure",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "The service could not complete the request.",
                "request_id": request_id,
            },
        },
    )

# Middleware stack
app.add_middleware(RequestBodyGuardMiddleware)  # ASGI-level: body size + content-type
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-History-Persisted"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Routers
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(admin_router)
app.include_router(create_health_router(analysis_service, _check_database))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
