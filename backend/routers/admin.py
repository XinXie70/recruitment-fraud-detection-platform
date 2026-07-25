"""
Admin routes — analytics dashboard, user management, and audit log.

All endpoints require ``is_admin=True`` on the authenticated user.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import AnalysisHistory, User
from schemas.admin import (
    AdminStats,
    AdminUserItem,
    AnalysisHistoryItem,
    PaginatedResponse,
)
from utils import Pagination, paginate

logger = logging.getLogger("fake_job_detection_api.admin")
router = APIRouter(prefix="/api/admin", tags=["admin"])


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@router.get("/stats", response_model=AdminStats)
def admin_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    total_users = db.query(func.count(User.id)).scalar() or 0
    total_analyses = db.query(func.count(AnalysisHistory.id)).scalar() or 0
    analyses_today = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.created_at >= today)
        .scalar()
        or 0
    )
    avg_risk = (
        db.query(func.avg(AnalysisHistory.risk_score)).scalar() or 0.0
    )
    high_risk = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.risk_level == "high")
        .scalar()
        or 0
    )
    medium_risk = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.risk_level == "medium")
        .scalar()
        or 0
    )
    low_risk = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.risk_level == "low")
        .scalar()
        or 0
    )

    return AdminStats(
        total_users=total_users,
        total_analyses=total_analyses,
        analyses_today=analyses_today,
        avg_risk_score=round(float(avg_risk), 4),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Analysis History
# ---------------------------------------------------------------------------
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
