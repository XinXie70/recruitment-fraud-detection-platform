"""Integration-style tests for the analysis orchestration service."""

from __future__ import annotations

import pytest

from services.analysis_service import AnalysisService, InputRejectedError
from services.cache import TTLCache
from services.ensemble_predictor import (
    CalibrationConfig,
    EnsembleConfig,
    EnsembleMemberConfig,
    EnsemblePredictor,
)
from services.model_adapter import RawModelResult
from xai_gentle import GentleAIService, XAIService


class CountingRegistry:
    def __init__(self):
        self.calls = 0

    def predict_all(self, text, model_keys, timeout_seconds):
        self.calls += 1
        return [RawModelResult("model", "Model", "success", score=0.8)]

    def predict_raw_batches(self, texts, model_keys, timeout_seconds):
        return {"model": [0.8 for _ in texts]}

    def warm_up(self, sample, model_keys):
        return {"model": None}


def _service(*, validator=None, url_analyzer=None):
    registry = CountingRegistry()
    config = EnsembleConfig(
        version="test-v1",
        fitted=True,
        weight_source="test",
        low_threshold=0.3,
        high_threshold=0.7,
        models={
            "model": EnsembleMemberConfig(1.0, CalibrationConfig("identity"))
        },
    )
    service = AnalysisService(
        ensemble=EnsemblePredictor(registry, config),
        xai=XAIService(prefer_shap=False),
        gentle_ai=GentleAIService(ollama_enabled=False),
        validator=validator or (lambda text: {"is_valid": True, "job_relevance_score": 0.9}),
        url_analyzer=url_analyzer or (lambda text: {
            "urls_found": 0,
            "risk_score": 0,
            "risk_level": "low",
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "urls": [],
            "reasons": ["none"],
        }),
        cache=TTLCache(ttl_seconds=60, max_entries=10),
    )
    return service, registry


def test_analysis_result_is_cached() -> None:
    service, registry = _service()
    first = service.analyze("A legitimate software engineering role")
    second = service.analyze("A legitimate software engineering role")
    assert first == second
    assert registry.calls == 1


def test_invalid_input_is_rejected_before_model_execution() -> None:
    service, registry = _service(
        validator=lambda text: {
            "is_valid": False,
            "status": "not_a_job",
            "reason": "Not a job advertisement.",
            "job_relevance_score": 0.1,
        }
    )
    with pytest.raises(InputRejectedError, match="Not a job"):
        service.analyze("hello")
    assert registry.calls == 0


def test_url_failure_degrades_without_leaking_exception() -> None:
    def broken_url_analyzer(text):
        raise RuntimeError("private-url-service-secret")

    service, _ = _service(url_analyzer=broken_url_analyzer)
    result = service.analyze("A software engineering role with salary details")
    assert result.status == "degraded"
    assert result.url_analysis.reasons == ["URL analysis is temporarily unavailable."]


def test_warm_up_sets_readiness() -> None:
    service, _ = _service()
    assert service.warm_up() == {"model": None}
    assert service.ready is True
