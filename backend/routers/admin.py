"""
Admin routes,  analytics dashboard, user management, and audit log.
All endpoints require  s_admin=True  on the authenticated user.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import AnalysisHistory, User
from backend.schemas.admin import (
    AdminStats,
    AdminUserItem,
    AnalysisHistoryItem,
    ModelMetricsResponse,
    PaginatedResponse,
)
from backend.utils import Pagination, paginate

logger = logging.getLogger("fake_job_detection_api.admin")
router = APIRouter(prefix="/api/admin", tags=["admin"])
MODEL_METRICS_PATH = Path(__file__).resolve().parents[1] / "data" / "model_metrics.json"


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


@lru_cache(maxsize=1)
def load_model_metrics() -> ModelMetricsResponse:
    with MODEL_METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return ModelMetricsResponse.model_validate(json.load(metrics_file))


@router.get("/model-metrics", response_model=ModelMetricsResponse)
def admin_model_metrics(_admin: User = Depends(require_admin)) -> ModelMetricsResponse:
    """Return  offline evaluation metrics"""
    return load_model_metrics()



# Stats

@router.get("/stats", response_model=AdminStats)
def admin_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    total_users = db.query(func.count(User.id)).scalar() or 0
    (
        total_analyses,
        analyses_today,
        avg_risk,
        high_risk,
        medium_risk,
        low_risk,
    ) = db.query(
        func.count(AnalysisHistory.id),
        func.sum(case((AnalysisHistory.created_at >= today, 1), else_=0)),
        func.avg(AnalysisHistory.risk_score),
        func.sum(case((AnalysisHistory.risk_level == "high", 1), else_=0)),
        func.sum(case((AnalysisHistory.risk_level == "medium", 1), else_=0)),
        func.sum(case((AnalysisHistory.risk_level == "low", 1), else_=0)),
    ).one()

    total_analyses = total_analyses or 0
    analyses_today = analyses_today or 0
    avg_risk = avg_risk or 0.0
    high_risk = high_risk or 0
    medium_risk = medium_risk or 0
    low_risk = low_risk or 0

    return AdminStats(
        total_users=total_users,
        total_analyses=total_analyses,
        analyses_today=analyses_today,
        avg_risk_score=round(float(avg_risk), 4),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
    )



# Users

@router.get("/users", response_model=PaginatedResponse)
def admin_users(
    page: Pagination = Depends(paginate),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    total = db.query(func.count(User.id)).scalar() or 0
    users_with_counts = (
        db.query(User, func.count(AnalysisHistory.id).label("analysis_count"))
        .outerjoin(AnalysisHistory, AnalysisHistory.user_id == User.id)
        .group_by(User.id)
        .order_by(User.created_at.desc())
        .offset(page.offset)
        .limit(page.limit)
        .all()
    )

    items = [
        AdminUserItem(
            id=user.id,
            email=user.email,
            username=user.username,
            is_admin=user.is_admin,
            analysis_count=analysis_count,
            created_at=user.created_at,
        ).model_dump()
        for user, analysis_count in users_with_counts
    ]

    total_pages = max(1, (total + page.size - 1) // page.size)
    return PaginatedResponse(
        items=items,
        total=total,
        page=page.page,
        page_size=page.size,
        total_pages=total_pages,
    )



# Analysis History

@router.get("/analyses", response_model=PaginatedResponse)
def admin_analyses(
    page: Pagination = Depends(paginate),
    user_id: int | None = None,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    query = db.query(AnalysisHistory)
    if user_id is not None:
        query = query.filter(AnalysisHistory.user_id == user_id)

    total = query.count()
    analyses = (
        query.order_by(AnalysisHistory.created_at.desc())
        .offset(page.offset)
        .limit(page.limit)
        .all()
    )

    items = [AnalysisHistoryItem.model_validate(a).model_dump() for a in analyses]
    total_pages = max(1, (total + page.size - 1) // page.size)
    return PaginatedResponse(
        items=items,
        total=total,
        page=page.page,
        page_size=page.size,
        total_pages=total_pages,
    )
