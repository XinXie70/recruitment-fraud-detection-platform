from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from auth import get_current_user
from config import settings
from database import get_db
from dependencies import get_analysis_service
from middleware import get_request_id
from models import AnalysisHistory, User
from rate_limit import limiter
from schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    EducationListResponse,
    URLAnalysis,
)
from services.analysis_service import AnalysisService, InputRejectedError
from services.ensemble_predictor import EnsembleUnavailableError
from url_analyzer import analyze_urls
from xai_gentle import EducationItem
from xai_gentle.contracts import EducationTopic


logger = logging.getLogger("fake_job_detection_api.analysis")
router = APIRouter(tags=["analysis"])


def _save_history(
    text: str,
    result: AnalysisResponse,
    user_id: int,
    db,
) -> None:
    """Persist analysis result to history table."""
    try:
        import hashlib

        history = AnalysisHistory(
            user_id=user_id,
            input_preview=text[:500],
            input_hash=hashlib.sha256(text.encode()).hexdigest(),
            risk_score=result.ensemble.risk_score,
            risk_level=result.ensemble.risk_level,
            status=result.status,
            ensemble_available=sum(
                1 for m in result.member_outputs if m.status == "success"
            ),
            ensemble_total=len(result.member_outputs),
            created_at=datetime.now(timezone.utc),
        )
        db.add(history)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist analysis history")


def _run_analysis(
    payload: AnalysisRequest,
    service: AnalysisService,
    request_id: str,
) -> AnalysisResponse:
    try:
        return service.analyze(payload.text)
    except InputRejectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": exc.status,
                "message": exc.reason,
                "job_relevance_score": exc.job_relevance_score,
            },
        ) from exc
    except EnsembleUnavailableError as exc:
        logger.error(
            "No ensemble member available",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Analysis failed",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed. Please try again later.",
        ) from exc


@router.post("/api/v1/analyze", response_model=AnalysisResponse)
@limiter.limit(settings.rate_limit_analyze)
def analyze_v1(
    request: Request,
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    request_id: str = Depends(get_request_id),
    db=Depends(get_db),
) -> AnalysisResponse:
    result = _run_analysis(payload, service, request_id)
    _save_history(payload.text, result, current_user.id, db)
    return result


@router.post("/api/predict", response_model=AnalysisResponse)
@limiter.limit(settings.rate_limit_analyze)
def predict_compatibility(
    request: Request,
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    request_id: str = Depends(get_request_id),
    db=Depends(get_db),
) -> AnalysisResponse:
    """Backward-compatible alias for /api/v1/analyze (used by legacy React frontend)."""
    result = _run_analysis(payload, service, request_id)
    _save_history(payload.text, result, current_user.id, db)
    return result


@router.post("/api/analyze-url", response_model=URLAnalysis)
def analyze_url_payload(
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
) -> URLAnalysis:
    try:
        return URLAnalysis.model_validate(analyze_urls(payload.text))
    except Exception as exc:
        logger.exception("URL analysis failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="URL analysis failed. Please try again later.",
        ) from exc


@router.get("/api/v1/education", response_model=EducationListResponse)
def list_education(
    topic: EducationTopic | None = Query(default=None),
    service: AnalysisService = Depends(get_analysis_service),
) -> EducationListResponse:
    return EducationListResponse(items=service.gentle_ai.list_items(topic))


@router.get("/api/v1/education/{item_id}", response_model=EducationItem)
def get_education_item(
    item_id: str,
    service: AnalysisService = Depends(get_analysis_service),
) -> EducationItem:
    item = service.gentle_ai.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Education item not found.")
    return item
