import sys
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

# Resolve path directory to import from model pipelines
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"

# Add directories to system path for import resolution
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from auth import get_current_user, router as auth_router
from database import Base, engine
from models import User
from url_analyzer import analyze_urls

try:
    from final_model_pipelines.lr_pipeline.predict import predict_job_posting as predict_lr
    from final_model_pipelines.svm_pipeline.predict import predict_job_posting as predict_svm
    from final_model_pipelines.xgboost_pipeline.predict import predict_job_posting as predict_xgboost
    from final_model_pipelines.dnn_pipeline.predict import predict_job_posting as predict_dnn
    from final_model_pipelines.rnn_pipeline.predict import predict_job_posting as predict_rnn
    from final_model_pipelines.bilstm_pipeline.predict import predict_job_posting as predict_bilstm
except ImportError as e:
    try:
        from model.final_model_pipelines.lr_pipeline.predict import predict_job_posting as predict_lr
        from model.final_model_pipelines.svm_pipeline.predict import predict_job_posting as predict_svm
        from model.final_model_pipelines.xgboost_pipeline.predict import predict_job_posting as predict_xgboost
        from model.final_model_pipelines.dnn_pipeline.predict import predict_job_posting as predict_dnn
        from model.final_model_pipelines.rnn_pipeline.predict import predict_job_posting as predict_rnn
        from model.final_model_pipelines.bilstm_pipeline.predict import predict_job_posting as predict_bilstm
    except ImportError:
        raise ImportError(
            "Could not import final_model_pipelines. Please ensure python paths "
            f"are configured properly. Original error: {e}"
        )

logger = logging.getLogger("fake_job_detection_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "50000"))
DEFAULT_CORS_ORIGINS = "http://127.0.0.1:5190,http://localhost:5190"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

MODEL_READY = False
MODEL_STARTUP_ERROR: str | None = None


def _init_database() -> None:
    Base.metadata.create_all(bind=engine)


def _warm_up_models() -> None:
    sample = "Software engineer role with clear requirements, company benefits, and standard interview process."
    predict_lr(sample)
    predict_svm(sample)
    predict_xgboost(sample)
    predict_dnn(sample)
    predict_rnn(sample)
    predict_bilstm(sample)


def predict_with_served_models(input_text: str) -> dict:
    return {
        "logistic_regression": predict_lr(input_text),
        "svm": predict_svm(input_text),
        "xgboost": predict_xgboost(input_text),
        "dnn": predict_dnn(input_text),
        "rnn": predict_rnn(input_text),
        "bilstm": predict_bilstm(input_text),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_READY, MODEL_STARTUP_ERROR
    _init_database()
    try:
        _warm_up_models()
        MODEL_READY = True
        MODEL_STARTUP_ERROR = None
        logger.info("Model artifacts loaded successfully.")
    except Exception as exc:
        MODEL_READY = False
        MODEL_STARTUP_ERROR = str(exc)
        logger.exception("Model warm-up failed.")

    yield

app = FastAPI(
    title="Fake Job Detection API",
    description="Provides real-time machine learning prediction of fake job postings using LR and DNN.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for the React/Vite local environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)


class PredictionRequest(BaseModel):
    text: str = Field(..., max_length=MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Job description text cannot be empty.")
        return cleaned


class ModelPrediction(BaseModel):
    risk_score: float | None = Field(None, ge=0, le=1)
    classification_label: str | None = None
    prediction: Literal["real", "fake"] | None = None
    recommended_action: str | None = None
    status: str | None = None
    message: str | None = None
    job_relevance_score: float | None = Field(None, ge=0, le=1)


class UrlResult(BaseModel):
    url: str
    domain: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    flags: list[str]
    flag_codes: list[str]


class UrlAnalysis(BaseModel):
    urls_found: int
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    high_risk_count: int
    medium_risk_count: int
    urls: list[UrlResult]
    reasons: list[str]


class PredictionResponse(BaseModel):
    logistic_regression: ModelPrediction
    svm: ModelPrediction
    xgboost: ModelPrediction
    dnn: ModelPrediction
    rnn: ModelPrediction
    bilstm: ModelPrediction
    url_analysis: UrlAnalysis


class HealthResponse(BaseModel):
    status: str
    service: str
    model_ready: bool


class ReadyResponse(BaseModel):
    status: str
    model_ready: bool


@app.post("/api/analyze-url", response_model=UrlAnalysis)
def analyze_url_payload(
    payload: PredictionRequest,
    current_user: User = Depends(get_current_user),
):
    """Accepts text or a standalone link and returns URL safety analysis."""
    try:
        return analyze_urls(payload.text)
    except Exception:
        logger.exception("URL analysis failed.")
        raise HTTPException(status_code=500, detail="URL analysis failed. Please try again later.")


@app.post("/api/predict", response_model=PredictionResponse)
def predict_job(
    payload: PredictionRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Accepts job posting text and returns classification results from both 
    Logistic Regression and Deep Neural Network models.
    """
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Prediction models are not ready.")

    try:
        model_results = predict_with_served_models(payload.text)
        return {
            **model_results,
            "url_analysis": analyze_urls(payload.text),
        }
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Liveness check for application monitoring."""
    return {
        "status": "healthy",
        "service": "fake_job_detection_api",
        "model_ready": MODEL_READY,
    }


@app.get("/api/ready", response_model=ReadyResponse)
def readiness_check():
    """Readiness check that fails when model artifacts cannot be loaded."""
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Prediction models are not ready.",
        )
    return {"status": "ready", "model_ready": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
