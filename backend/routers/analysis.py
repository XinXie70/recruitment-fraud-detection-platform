from __future__ import annotations

import hashlib
import hmac
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from backend.auth import get_current_user
from backend.config import settings
from backend.database import get_db
from backend.dependencies import get_analysis_service
from backend.middleware import get_request_id
from backend.models import AnalysisHistory, User
from backend.rate_limit import limiter
from backend.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    EducationListResponse,
    URLAnalysis,
    UserAnalysisHistoryDetail,
    UserAnalysisHistoryItem,
    UserAnalysisHistoryPage,
)
from backend.services.analysis_service import AnalysisService, InputRejectedError
from backend.services.fp_gate_predictor import EnsembleUnavailableError
from backend.url_analyzer import analyze_urls
from backend.utils import Pagination, paginate
from backend.xai_gentle import EducationItem
from backend.xai_gentle.contracts import EducationTopic


logger = logging.getLogger("fake_job_detection_api.analysis")
router = APIRouter(tags=["analysis"])

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_SECRET_PATTERN = re.compile(
    r"\b(password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _redact_history_preview(text: str) -> str:

    preview = text[:500]
    preview = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", preview)
    preview = _PHONE_PATTERN.sub("[REDACTED_PHONE]", preview)
    preview = _URL_PATTERN.sub("[REDACTED_URL]", preview)
    return _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", preview)


def _hash_history_input(text: str) -> str:

    return hmac.new(
        settings.secret_key.encode("utf-8"),
        text.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _save_history(
    text: str,
    result: AnalysisResponse,
    user_id: int,
    db,
) -> bool:


    try:
        history = AnalysisHistory(
            user_id=user_id,
            input_preview=_redact_history_preview(text),
            input_hash=_hash_history_input(text),
            risk_score=result.ensemble.risk_score,
            risk_level=result.ensemble.risk_level,
            status=result.status,
            ensemble_available=sum(
                1 for m in result.member_outputs if m.status == "success"
            ),
            ensemble_total=len(result.member_outputs),
            analysis_result={
                **result.model_dump(mode="json"),
                # Never persist the unredacted advert: it can contain contact
                # details, credentials, or other personal information.
                "inputText": _redact_history_preview(text),
            },
            created_at=datetime.now(timezone.utc),
        )
        db.add(history)
        db.commit()
        return True
    except Exception:
        db.rollback()
        logger.exception("Failed to persist analysis history")
        return False


def _run_analysis(
    payload: AnalysisRequest,
    service: AnalysisService,
    request_id: str,
) -> AnalysisResponse:
    try:
        return service.analyze(payload.text)
    except InputRejectedError as exc:
        raise HTTPException(
            status_code=422,
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


def _run_score(
    payload: AnalysisRequest,
    service: AnalysisService,
    request_id: str,
) -> AnalysisResponse:
    try:
        return service.score(payload.text)
    except InputRejectedError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "status": exc.status,
                "message": exc.reason,
                "job_relevance_score": exc.job_relevance_score,
            },
        ) from exc
    except EnsembleUnavailableError as exc:
        logger.error("No ensemble member available", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Score phase failed", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed. Please try again later.",
        ) from exc


@router.post("/api/v1/analyze/score", response_model=AnalysisResponse)
@limiter.limit(settings.rate_limit_analyze)
def analyze_score_v1(
    request: Request,
    payload: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    request_id: str = Depends(get_request_id),
) -> AnalysisResponse:

    return _run_score(payload, service, request_id)

@router.post("/api/v1/analyze", response_model=AnalysisResponse)
@limiter.limit(settings.rate_limit_analyze)
def analyze_v1(
    request: Request,
    payload: AnalysisRequest,
    response: Response,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    request_id: str = Depends(get_request_id),
    db=Depends(get_db),
) -> AnalysisResponse:
    result = _run_analysis(payload, service, request_id)
    persisted = _save_history(payload.text, result, current_user.id, db)
    response.headers["X-History-Persisted"] = str(persisted).lower()
    return result


@router.post("/api/analyze-url", response_model=URLAnalysis)
@limiter.limit(settings.rate_limit_analyze)
def analyze_url_payload(
    request: Request,
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


@router.get("/api/v1/history", response_model=UserAnalysisHistoryPage)
def list_own_analysis_history(
    page: Pagination = Depends(paginate),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
) -> UserAnalysisHistoryPage:
    query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == current_user.id)
    total = query.count()
    rows = (
        query.order_by(AnalysisHistory.created_at.desc())
        .offset(page.offset)
        .limit(page.limit)
        .all()
    )
    return UserAnalysisHistoryPage(
        items=[
            UserAnalysisHistoryItem(
                **UserAnalysisHistoryItem.model_validate(row).model_dump(
                    exclude={"has_result"}
                ),
                has_result=row.analysis_result is not None,
            )
            for row in rows
        ],
        total=total,
        page=page.page,
        page_size=page.size,
        total_pages=(total + page.size - 1) // page.size,
    )


@router.get(
    "/api/v1/history/{history_id}",
    response_model=UserAnalysisHistoryDetail,
)
def get_own_analysis_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
) -> UserAnalysisHistoryDetail:
    history = (
        db.query(AnalysisHistory)
        .filter(
            AnalysisHistory.id == history_id,
            AnalysisHistory.user_id == current_user.id,
        )
        .first()
    )
    if history is None or history.analysis_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Full analysis result is not available for this history item.",
        )
    return UserAnalysisHistoryDetail(
        id=history.id,
        analysis_result=history.analysis_result,
    )


@router.delete("/api/v1/history/{history_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_own_analysis_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
) -> Response:
    history = (
        db.query(AnalysisHistory)
        .filter(
            AnalysisHistory.id == history_id,
            AnalysisHistory.user_id == current_user.id,
        )
        .first()
    )
    if history is None:


        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis history item not found.",
        )
    try:
        db.delete(history)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to delete analysis history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis history could not be deleted.",
        ) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Education item not found."
        )
    return item
