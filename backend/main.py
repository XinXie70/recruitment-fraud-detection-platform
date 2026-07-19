import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"

for import_path in (BACKEND_DIR, PROJECT_ROOT, MODEL_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import router as auth_router
from database import Base, engine
from routers.analysis import analysis_service, router as analysis_router


logger = logging.getLogger("fake_job_detection_api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s request_id=%(request_id)s %(message)s",
)

DEFAULT_CORS_ORIGINS = "http://127.0.0.1:5190,http://localhost:5190"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]


def _validate_secret_key() -> None:
    secret = os.getenv("SECRET_KEY", "")
    if not secret or secret == "change-this-secret-key-for-local-development":
        logger.critical(
            "SECRET_KEY is using the default placeholder. "
            "Set a strong random SECRET_KEY environment variable in production."
        )
        if os.getenv("ENVIRONMENT", "").lower() == "production":
            sys.exit(1)


def _init_database() -> None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        logger.exception("Database initialisation failed – aborting startup.")
        sys.exit(1)


_request_id_context: dict[int, str] = {}


def _set_request_id() -> str:
    import threading
    rid = uuid.uuid4().hex[:12]
    _request_id_context[threading.get_ident()] = rid
    return rid


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        import threading
        record.request_id = _request_id_context.get(threading.get_ident(), "-")
        return True


for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_secret_key()
    _init_database()

    outcomes = analysis_service.warm_up()
    failed = {key: error for key, error in outcomes.items() if error}
    if analysis_service.ready:
        logger.info("Ensemble runtime is ready. Failed members: %s", failed or "none")
    else:
        logger.error("No ensemble model could be loaded: %s", failed)
    yield
    # --- graceful shutdown ---
    logger.info("Shutting down – cleaning up model executors.")
    try:
        analysis_service.shutdown()
    except Exception:
        logger.exception("Error during model executor shutdown.")


app = FastAPI(
    title="Fake Job Detection API",
    description=(
        "Provides a backend ensemble risk result, model-derived XAI evidence, "
        "and gentle educational guidance for English job advertisements."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = _set_request_id()
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = rid
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(auth_router)
app.include_router(analysis_router)


class HealthResponse(BaseModel):
    status: str
    service: str
    model_ready: bool


class ReadyResponse(BaseModel):
    status: str
    model_ready: bool


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="fake_job_detection_api",
        model_ready=analysis_service.ready,
    )


@app.get("/api/ready", response_model=ReadyResponse)
def readiness_check() -> ReadyResponse:
    if not analysis_service.ready:
        raise HTTPException(status_code=503, detail="Prediction models are not ready.")
    return ReadyResponse(status="ready", model_ready=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
