"""Unit tests for analysis route error mapping and supporting endpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routers import analysis as analysis_router
from schemas.analysis import AnalysisRequest
from services.analysis_service import InputRejectedError
from services.ensemble_predictor import EnsembleUnavailableError
from xai_gentle import GentleAIService


class RaisingService:
    def __init__(self, error):
        self.error = error

    def analyze(self, text):
        raise self.error


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (InputRejectedError("invalid", "Not a job", 0.1), 422),
        (EnsembleUnavailableError("private model failure"), 503),
        (RuntimeError("private server detail"), 500),
    ],
)
def test_analysis_errors_map_to_safe_http_responses(error, status_code) -> None:
    with pytest.raises(HTTPException) as raised:
        analysis_router._run_analysis(
            AnalysisRequest(text="meaningful listing"),
            RaisingService(error),
            "request-id",
        )
    assert raised.value.status_code == status_code
    if status_code >= 500:
        assert "private" not in str(raised.value.detail)


def test_history_failure_rolls_back_without_crashing() -> None:
    class FailingDB:
        rolled_back = False

        def add(self, value):
            self.value = value

        def commit(self):
            raise RuntimeError("database offline")

        def rollback(self):
            self.rolled_back = True

    result = SimpleNamespace(
        ensemble=SimpleNamespace(risk_score=0.8, risk_level="high"),
        status="success",
        member_outputs=[SimpleNamespace(status="success")],
    )
    db = FailingDB()
    analysis_router._save_history("listing", result, 1, db)
    assert db.rolled_back is True


def test_url_analysis_error_is_sanitized(monkeypatch) -> None:
    monkeypatch.setattr(
        analysis_router,
        "analyze_urls",
        lambda text: (_ for _ in ()).throw(RuntimeError("private URL detail")),
    )
    with pytest.raises(HTTPException) as raised:
        analysis_router.analyze_url_payload(
            AnalysisRequest(text="https://example.com"),
            current_user=SimpleNamespace(id=1),
        )
    assert raised.value.status_code == 500
    assert "private" not in raised.value.detail


def test_education_endpoints_return_items_and_404() -> None:
    service = SimpleNamespace(gentle_ai=GentleAIService(ollama_enabled=False))
    listing = analysis_router.list_education(topic=None, service=service)
    assert listing.items

    item = analysis_router.get_education_item(listing.items[0].id, service=service)
    assert item.id == listing.items[0].id

    with pytest.raises(HTTPException) as raised:
        analysis_router.get_education_item("missing", service=service)
    assert raised.value.status_code == 404
