from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from auth import get_current_user
from models import User
from schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    EducationListResponse,
    URLAnalysis,
)
from services.analysis_service import AnalysisService, InputRejectedError
from services.ensemble_predictor import EnsembleUnavailableError
from url_analyzer import analyze_urls
from backend.xai_gentle import EducationItem


logger = logging.getLogger("fake_job_detection_api.analysis")
router = APIRouter(tags=["analysis"])
analysis_service = AnalysisService.from_environment()


def get_analysis_service() -> AnalysisService:
    return analysis_service


def _run_analysis(payload: AnalysisRequest, service: AnalysisService) -> AnalysisResponse:
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
        logger.exception("No ensemble member was available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Analysis failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed. Please try again later.",
        ) from exc


@router.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze_v1(
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    return _run_analysis(payload, service)


@router.post("/api/predict", response_model=AnalysisResponse)
def predict_compatibility(
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """Backward-compatible route used by the existing React application."""
    return _run_analysis(payload, service)


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


EducationTopic = Literal["fake_jobs", "misinformation", "phishing", "scam_patterns"]


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
