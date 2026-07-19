from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import AnalysisHistory, User
from schemas.analysis import (
    AnalysisHistoryItem,
    AnalysisHistoryListResponse,
    AnalysisRequest,
    AnalysisResponse,
    AdminStatsResponse,
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


def _save_analysis(
    db: Session,
    user_id: int,
    text: str,
    result: AnalysisResponse,
) -> AnalysisHistory:
    snippet = text[:500] if len(text) > 500 else text
    history = AnalysisHistory(
        user_id=user_id,
        input_text=snippet,
        risk_score=result.ensemble.risk_score,
        risk_level=result.ensemble.risk_level,
        classification_label=result.ensemble.classification_label,
        result_json=json.loads(result.model_dump_json()),
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history


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
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    result = _run_analysis(payload, service)
    _save_analysis(db, current_user.id, payload.text, result)
    return result


@router.post("/api/predict", response_model=AnalysisResponse)
def predict_compatibility(
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Backward-compatible route used by the existing React application."""
    result = _run_analysis(payload, service)
    _save_analysis(db, current_user.id, payload.text, result)
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


# ---------------------------------------------------------------------------
# User analysis history
# ---------------------------------------------------------------------------


@router.get("/api/v1/history", response_model=AnalysisHistoryListResponse)
def get_user_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisHistoryListResponse:
    total = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.user_id == current_user.id)
        .scalar()
    )
    rows = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.user_id == current_user.id)
        .order_by(AnalysisHistory.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return AnalysisHistoryListResponse(
        total=total or 0,
        items=[
            AnalysisHistoryItem(
                id=row.id,
                risk_score=row.risk_score,
                risk_level=row.risk_level,
                classification_label=row.classification_label,
                input_text_snippet=row.input_text[:200] if len(row.input_text) > 200 else row.input_text,
                created_at=row.created_at.isoformat(),
            )
            for row in rows
        ],
    )


@router.get("/api/v1/history/{analysis_id}", response_model=AnalysisResponse)
def get_analysis_detail(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    row = (
        db.query(AnalysisHistory)
        .filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if row.result_json is None:
        raise HTTPException(status_code=500, detail="Result data missing.")
    return AnalysisResponse.model_validate(row.result_json)


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------


def _require_admin(current_user: User) -> None:
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required.")


@router.get("/api/v1/admin/stats", response_model=AdminStatsResponse)
def admin_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AdminStatsResponse:
    _require_admin(current_user)
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_analyses = db.query(func.count(AnalysisHistory.id)).scalar() or 0
    analyses_today = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.created_at >= today_start)
        .scalar()
        or 0
    )
    risk_rows = (
        db.query(AnalysisHistory.risk_level, func.count(AnalysisHistory.id))
        .group_by(AnalysisHistory.risk_level)
        .all()
    )
    risk_distribution = {row[0]: row[1] for row in risk_rows}
    recent = (
        db.query(AnalysisHistory)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(10)
        .all()
    )
    return AdminStatsResponse(
        total_users=total_users,
        total_analyses=total_analyses,
        analyses_today=analyses_today,
        risk_distribution=risk_distribution,
        recent_analyses=[
            AnalysisHistoryItem(
                id=row.id,
                risk_score=row.risk_score,
                risk_level=row.risk_level,
                classification_label=row.classification_label,
                input_text_snippet=row.input_text[:200] if len(row.input_text) > 200 else row.input_text,
                created_at=row.created_at.isoformat(),
            )
            for row in recent
        ],
    )


@router.get("/api/v1/admin/analyses", response_model=AnalysisHistoryListResponse)
def admin_analyses(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    risk_level: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisHistoryListResponse:
    _require_admin(current_user)
    query = db.query(AnalysisHistory)
    if risk_level:
        query = query.filter(AnalysisHistory.risk_level == risk_level)
    total = query.count()
    rows = (
        query.order_by(AnalysisHistory.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return AnalysisHistoryListResponse(
        total=total,
        items=[
            AnalysisHistoryItem(
                id=row.id,
                risk_score=row.risk_score,
                risk_level=row.risk_level,
                classification_label=row.classification_label,
                input_text_snippet=row.input_text[:200] if len(row.input_text) > 200 else row.input_text,
                created_at=row.created_at.isoformat(),
            )
            for row in rows
        ],
    )
