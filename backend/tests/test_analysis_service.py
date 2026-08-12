"""Integration-style tests for the analysis orchestration service."""

from __future__ import annotations

import pytest

from backend.schemas.analysis import EnsembleResult, ModelMemberOutput
from backend.services.analysis_service import AnalysisService, InputRejectedError
from backend.services.cache import TTLCache
from backend.services.fp_gate_predictor import EnsembleComputation
from backend.xai_gentle import GentleAIService, XAIResult


class CountingPredictor:
    def __init__(self):
        self.calls = 0
        self.batch_calls = 0

    def predict(self, text):
        self.calls += 1
        ensemble = EnsembleResult(
            status="success",
            risk_score=0.8,
            classification_label="Likely Deceptive",
            risk_level="high",
            prediction="fake",
            recommended_action="High Risk Warning",
            low_threshold=0.15,
            high_threshold=0.32,
            active_model_count=2,
            failed_model_count=0,
            version="test-fp-gate",
            fitted=True,
            decision_strategy="bert_primary_lr_fp_gate",
            risk_score_source="bert",
            gate_triggered=False,
        )
        members = [
            ModelMemberOutput(
                key="bert",
                display_name="BERT",
                status="success",
                raw_score=0.8,
                role="primary_score",
                decision_active=True,
            )
        ]

        def score_batch(texts):
            self.batch_calls += 1
            return [0.8 for _ in texts]

        return EnsembleComputation(ensemble, members, score_batch)

    # Call the model in advance during startup
    def warm_up(self, sample):
        return {"final_ensemble": None}

    def is_available(self):
        return True


class StaticXAIService:
    def explain(self, text, score_batch, expected_output):
        return XAIResult(
            status="success",
            method="shap_partition",
            output_value=expected_output,
            items=[],
        )


def _service(*, validator=None, url_analyzer=None):
    predictor = CountingPredictor()
    service = AnalysisService(
        ensemble=predictor,
        xai=StaticXAIService(),
        gentle_ai=GentleAIService(ollama_enabled=False),
        validator=validator
        or (lambda text: {"is_valid": True, "job_relevance_score": 0.9}),
        url_analyzer=url_analyzer
        or (
            lambda text: {
                "urls_found": 0,
                "risk_score": 0,
                "risk_level": "low",
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "urls": [],
                "reasons": ["none"],
            }
        ),
        cache=TTLCache(ttl_seconds=60, max_entries=10),
    )
    return service, predictor


def test_analysis_result_is_cached() -> None:
    service, predictor = _service()
    first = service.analyze("A legitimate software engineering role")
    second = service.analyze("A legitimate software engineering role")
    assert first == second
    assert predictor.calls == 1


def test_score_phase_skips_xai_batch_scoring() -> None:
    service, predictor = _service()
    result = service.score("A legitimate software engineering role")
    assert result.phase == "score"
    assert result.ensemble.risk_score == pytest.approx(0.8)
    assert result.xai.status == "unavailable"
    assert "being prepared" in result.xai.message
    assert predictor.calls == 1
    assert predictor.batch_calls == 0


def test_score_phase_never_calls_ollama(monkeypatch) -> None:
    service, _ = _service()
    service.gentle_ai.ollama_enabled = True
    service.gentle_ai.ollama_model = "configured-model"
    rewrite_calls = 0

    def unexpected_rewrite(template):
        nonlocal rewrite_calls
        rewrite_calls += 1
        return template

    monkeypatch.setattr(service.gentle_ai, "_rewrite_with_ollama", unexpected_rewrite)

    result = service.score("A legitimate software engineering role")

    assert rewrite_calls == 0
    assert result.gentle_ai.provider == "template"
    assert "fast score phase" in result.gentle_ai.message

# Real-time health status check
def test_readiness_is_refreshed_from_live_model_probe() -> None:
    service, predictor = _service()
    predictor.is_available = lambda: False
    assert service.refresh_readiness() is False
    assert service.ready is False

    predictor.is_available = lambda: True
    assert service.refresh_readiness() is True
    assert service.ready is True


def test_complete_cache_does_not_change_score_phase_contract() -> None:
    service, predictor = _service()
    text = "A legitimate software engineering role"
    complete = service.analyze(text)
    score = service.score(text)
    cached_score = service.score(text)
    assert complete.phase == "complete"
    assert score.phase == "score"
    assert score.xai.status == "unavailable"
    assert cached_score == score
    assert predictor.calls == 1


def test_invalid_input_is_rejected_before_model_execution() -> None:
    service, predictor = _service(
        validator=lambda text: {
            "is_valid": False,
            "status": "not_a_job",
            "reason": "Not a job advertisement.",
            "job_relevance_score": 0.1,
        }
    )
    with pytest.raises(InputRejectedError, match="Not a job"):
        service.analyze("hello")
    assert predictor.calls == 0


def test_url_failure_degrades_without_leaking_exception() -> None:
    def broken_url_analyzer(text):
        raise RuntimeError("private-url-service-secret")

    service, _ = _service(url_analyzer=broken_url_analyzer)
    result = service.analyze("A software engineering role with salary details")
    assert result.status == "degraded"
    assert result.url_analysis.reasons == ["URL analysis is temporarily unavailable."]


def test_warm_up_sets_readiness() -> None:
    service, _ = _service()
    assert service.warm_up() == {"final_ensemble": None}
    assert service.ready is True
