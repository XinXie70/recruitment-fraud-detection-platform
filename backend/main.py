import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# sys.path setup must happen BEFORE any local imports
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"

for import_path in (BACKEND_DIR, PROJECT_ROOT, MODEL_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from config import settings

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from auth import hash_password
from auth import router as auth_router
from database import Base, SessionLocal, engine
from dependencies import get_analysis_service
from middleware import (
    RequestBodyGuardMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from models import User
from rate_limit import limiter
from routers.admin import router as admin_router
from routers.analysis import router as analysis_router
from services.resilience import ServiceStatus, SystemHealth



# Structured logging

class _JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines for Cloud Run / structured log ingestion."""
    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr in (
            "request_id",
            "path",
            "method",
            "body_len",
            "status_code",
            "duration_ms",
        ):
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    root.handlers.clear()
    root.addHandler(handler)


_setup_logging()
logger = logging.getLogger("fake_job_detection_api")

CORS_ORIGINS = settings.cors_origin_list

# Resolve the singleton once at import time for lifespan + health checks.
analysis_service = get_analysis_service()


def _init_database() -> None:
    if settings.app_env == "production":

        return
    Base.metadata.create_all(bind=engine)


def _provision_admin() -> None:

    credentials = (
        settings.admin_email.strip().lower(),
        settings.admin_username.strip(),
        settings.admin_password,
    )
    if not any(credentials):
        logger.info("Dedicated administrator provisioning is not configured.")
        return
    if not all(credentials):
        raise RuntimeError(
            "ADMIN_EMAIL, ADMIN_USERNAME and ADMIN_PASSWORD must all be configured."
        )

    email, username, password = credentials
    password_is_valid = (
        len(password) >= 8
        and len(password.encode("utf-8")) <= 72
        and any(char.isupper() for char in password)
        and any(char.islower() for char in password)
        and any(char.isdigit() for char in password)
    )
    if not password_is_valid:
        raise RuntimeError(
            "ADMIN_PASSWORD must be 8-72 bytes and contain uppercase, lowercase and digits."
        )

    db = SessionLocal()
    try:
        by_email = db.query(User).filter(User.email == email).one_or_none()
        by_username = db.query(User).filter(User.username == username).one_or_none()
        if by_email and by_username and by_email.id != by_username.id:
            raise RuntimeError("Administrator email and username belong to different users.")

        admin = by_email or by_username
        if admin is None:
            admin = User(
                email=email,
                username=username,
                password_hash=hash_password(password),
                is_admin=True,
            )
            db.add(admin)
        else:
            admin.email = email
            admin.username = username
            admin.password_hash = hash_password(password)
            admin.is_admin = True
        db.commit()
        logger.info("Dedicated administrator account is ready: %s", username)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _check_database() -> bool:

    db = None
    try:
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        if db is not None:
            db.close()


def _warm_up_models_background() -> None:

    try:
        outcomes = analysis_service.warm_up()
        failed = {key: error for key, error in outcomes.items() if error}
        if analysis_service.ready:
            logger.info("Ensemble runtime is ready. Failed members: %s", failed or "none")
        else:
            logger.error("No ensemble model could be loaded: %s", failed)
    except Exception:
        logger.exception("Background model warm-up failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _init_database()
        _provision_admin()
    except Exception:
        logger.exception("Database initialisation failed.")

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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Routers
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(admin_router)


# Endpoints

class ReadyResponse(BaseModel):
    status: str
    model_ready: bool
    database_connected: bool


@app.get("/api/live", include_in_schema=False)
def liveness_check() -> dict[str, str]:
    """Process-level probe that does not depend on the database or model runtime."""
    return {"status": "alive"}


@app.get("/api/health")
@limiter.limit(settings.rate_limit_global)
def health_check(request: Request):
    """Rich health endpoint showing ensemble, DB, and external service status."""
    health = SystemHealth.from_analysis_service(analysis_service)
    health.database_connected = _check_database()
    health.ollama_available = analysis_service.gentle_ai.ollama_enabled

    status_code = 503 if health.status == ServiceStatus.UNAVAILABLE else 200

    return JSONResponse(
        status_code=status_code,
        content={
            "status": health.status.value,
            "service": health.service,
            "version": health.version,
            "ensemble": {
                "available": health.ensemble_members_available,
                "total": health.ensemble_members_total,
                "failed_members": sorted(health.failed_members),
            },
            "database_connected": health.database_connected,
            "ollama_available": health.ollama_available,
        },
    )


@app.get("/api/ready", response_model=ReadyResponse)
def readiness_check() -> ReadyResponse:
    model_ready = analysis_service.ready
    database_connected = _check_database()
    if not model_ready or not database_connected:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service dependencies are not ready.",
                "model_ready": model_ready,
                "database_connected": database_connected,
            },
        )
    return ReadyResponse(
        status="ready",
        model_ready=True,
        database_connected=True,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
