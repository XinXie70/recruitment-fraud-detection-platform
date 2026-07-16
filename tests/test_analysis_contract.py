import pytest

from services.analysis_service import AnalysisService
from services.ensemble_predictor import (
    CalibrationConfig,
    EnsembleConfig,
    EnsembleMemberConfig,
    EnsemblePredictor,
)
from services.model_adapter import ModelRegistry
from backend.xai_gentle import GentleAIService, XAIService

from helpers import TextAwareAdapter


def build_service() -> AnalysisService:
    config = EnsembleConfig(
        version="contract-test",
        fitted=True,
        weight_source="contract_test",
        low_threshold=0.3,
        high_threshold=0.7,
        models={
            "mock": EnsembleMemberConfig(1.0, CalibrationConfig(kind="identity"))
        },
    )
    ensemble = EnsemblePredictor(
        ModelRegistry([TextAwareAdapter("mock", "Mock model", 0.2)]),
        config,
        timeout_seconds=2,
    )
    return AnalysisService(
        ensemble=ensemble,
        xai=XAIService(prefer_shap=False),
        gentle_ai=GentleAIService(ollama_enabled=False),
        validator=lambda text: {
            "is_valid": True,
            "status": "valid",
            "reason": "",
            "job_relevance_score": 0.9,
        },
        url_analyzer=lambda text: {
            "urls_found": 0,
            "risk_score": 0,
            "risk_level": "low",
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "urls": [],
            "reasons": ["No URLs were found."],
        },
    )


def test_complete_contract_contains_ensemble_xai_gentle_and_url_sections():
    response = build_service().analyze(
        "Urgent job opportunity. Pay a registration fee before the interview."
    )
    payload = response.model_dump(mode="json")

    assert payload["api_version"] == "1.0"
    assert payload["ensemble"]["risk_score"] == pytest.approx(0.7)
    assert len(payload["member_outputs"]) == 1
    assert payload["xai"]["target"] == "ensemble_fake_probability"
    assert payload["gentle_ai"]["provider"] == "template"
    assert payload["url_analysis"]["urls_found"] == 0
