"""Tests for deterministic Gentle AI fallback guidance."""

from __future__ import annotations

import pytest

from xai_gentle import EvidenceSpan, GentleAIService, RiskContext, XAIResult
from xai_gentle.gentle_fallback import build_template_guidance


def _risk(label="Suspicious", level="medium", score=0.5, action="Review Required"):
    return RiskContext(
        risk_score=score,
        risk_level=level,
        classification_label=label,
        recommended_action=action,
    )


def _xai() -> XAIResult:
    return XAIResult(
        status="success",
        method="occlusion_fallback",
        output_value=0.5,
        items=[
            EvidenceSpan(
                text="gift card payment",
                start=0,
                end=17,
                contribution=0.2,
                direction="raises_risk",
            )
        ],
    )


def test_disabled_ollama_returns_template_guidance() -> None:
    service = GentleAIService(ollama_enabled=False)
    result = service.generate(_risk(), _xai())

    assert result.status == "fallback"
    assert result.provider == "template"
    assert result.evidence_explanations
    assert "disabled" in result.message
    assert service.list_items("fake_jobs")
    assert service.get_item("missing") is None


def test_missing_ollama_model_uses_fallback() -> None:
    service = GentleAIService(ollama_enabled=True, ollama_model="")
    assert "not configured" in service.generate(_risk(), _xai()).message


def test_ollama_failure_is_contained(monkeypatch) -> None:
    service = GentleAIService(ollama_enabled=True, ollama_model="model")
    monkeypatch.setattr(
        service,
        "_rewrite_with_ollama",
        lambda template: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    result = service.generate(_risk(), _xai())
    assert result.status == "fallback"
    assert "RuntimeError" in result.message


@pytest.mark.parametrize(
    ("risk", "phrase"),
    [
        (_risk("Likely Deceptive", "high", 0.9, "High Risk Warning"), "extra care"),
        (_risk("Suspicious", "medium", 0.5, "Review Required"), "uncertain"),
        (_risk("Likely Legitimate", "low", 0.1, "Safe"), "Fewer concerning"),
    ],
)
def test_template_summary_matches_risk_level(risk, phrase) -> None:
    result = build_template_guidance(risk, _xai(), [])
    assert phrase in result.summary
