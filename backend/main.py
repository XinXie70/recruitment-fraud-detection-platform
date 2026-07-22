import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# ── sys.path setup must happen BEFORE any local imports ──────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"

for import_path in (BACKEND_DIR, PROJECT_ROOT, MODEL_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from config import settings

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from auth import router as auth_router
from database import Base, engine, SessionLocal
from dependencies import get_analysis_service
from middleware import RequestBodyGuardMiddleware, RequestIDMiddleware
from rate_limit import limiter
from routers.admin import router as admin_router
from routers.analysis import router as analysis_router
from services.resilience import ServiceStatus, SystemHealth


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines for Cloud Run / structured log ingestion."""
    def format(self, record: logging.LogRecord) -> str:
        import json, time as _time
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr in ("request_id", "path", "method", "body_len"):
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
    Base.metadata.create_all(bind=engine)


def _check_database() -> bool:
    """Ping the database to verify connectivity."""
    try:
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception:
        return False


def _warm_up_models_background() -> None:
    """Load ML models in a background thread so auth endpoints are available immediately."""
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
    except Exception:
        logger.exception("Database initialisation failed.")

    # Start model warm-up in background — do NOT block app startup.
    threading.Thread(target=_warm_up_models_background, daemon=True).start()
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fake Job Detection API",
    description=(
        "Provides a backend ensemble risk result, model-derived XAI evidence, "
        "and gentle educational guidance for English job advertisements."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# Rate limiter — attach state + exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware stack (order matters: outer → inner)
app.add_middleware(RequestBodyGuardMiddleware)  # ASGI-level: body size + content-type
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

# Routers
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(admin_router)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
class ReadyResponse(BaseModel):
    status: str
    model_ready: bool


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
                "failed_members": health.failed_members,
            },
            "database_connected": health.database_connected,
            "ollama_available": health.ollama_available,
        },
    )


@app.get("/api/ready", response_model=ReadyResponse)
def readiness_check() -> ReadyResponse:
    if not analysis_service.ready:
        raise HTTPException(status_code=503, detail="Prediction models are not ready.")
    return ReadyResponse(status="ready", model_ready=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
