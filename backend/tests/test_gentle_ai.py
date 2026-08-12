"""Tests for deterministic Gentle AI fallback guidance."""

from __future__ import annotations
import pytest
from backend.xai_gentle import EvidenceSpan, GentleAIService, RiskContext, XAIResult
from backend.xai_gentle.gentle_fallback import build_template_guidance


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
        method="shap_partition",
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

# display the risk score
def test_generate_local_never_uses_ollama(monkeypatch) -> None:
    service = GentleAIService(ollama_enabled=True, ollama_model="configured-model")
    rewrite_calls = 0

    def unexpected_rewrite(template):
        nonlocal rewrite_calls
        rewrite_calls += 1
        return template

    monkeypatch.setattr(service, "_rewrite_with_ollama", unexpected_rewrite)

    result = service.generate_local(_risk(), _xai())

    assert rewrite_calls == 0
    assert result.provider == "template"
    assert "fast score phase" in result.message


def test_template_explanation_uses_matching_local_education_topic() -> None:
    service = GentleAIService(ollama_enabled=False)
    result = service.generate(_risk(), _xai())

    explanation = result.evidence_explanations[0].explanation
    assert "money, a fee" in explanation
    assert "up-front payment" in explanation
    assert "not independent proof" in explanation


def test_template_explanations_are_specific_to_phrase_meaning() -> None:
    service = GentleAIService(ollama_enabled=False)
    xai = XAIResult(
        status="success",
        method="shap_partition",
        output_value=0.2,
        items=[
            EvidenceSpan(
                text="urgent hiring",
                start=0,
                end=13,
                contribution=0.1,
                direction="raises_risk",
            ),
            EvidenceSpan(
                text="salary packaging",
                start=20,
                end=36,
                contribution=-0.05,
                direction="lowers_risk",
            ),
        ],
    )

    explanations = service.generate(_risk(), xai).evidence_explanations

    assert "urgency" in explanations[0].explanation
    assert "employment condition or employee benefit" in explanations[1].explanation
    assert explanations[0].explanation != explanations[1].explanation


def test_template_preserves_shap_direction_when_guidance_conflicts() -> None:
    service = GentleAIService(ollama_enabled=False)
    xai = XAIResult(
        status="success",
        method="shap_partition",
        output_value=0.2,
        items=[
            EvidenceSpan(
                text="$50 registration fee",
                start=0,
                end=20,
                contribution=-0.05,
                direction="lowers_risk",
            )
        ],
    )

    explanation = service.generate(_risk(), xai).evidence_explanations[0]

    assert explanation.direction == "lowers_risk"
    assert "SHAP nevertheless moved the model toward lower risk" in explanation.explanation
    assert "should not be treated as reassurance" in explanation.explanation


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
    assert result.message == "Local template guidance used because Ollama was unavailable."
    assert "offline" not in result.message


def test_ollama_request_disables_thinking_and_bounds_output(monkeypatch) -> None:
    import httpx

    request_payload = {}

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return FakeResponse({"models": [{"name": "qwen3:4b"}]})

        def post(self, _url, json):
            request_payload.update(json)
            return FakeResponse(
                {
                    "message": {
                        "content": (
                            '{"summary":"Calm summary.",'
                            '"evidence_explanations":["Calm evidence."]}'
                        )
                    }
                }
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = GentleAIService(
        ollama_enabled=True,
        ollama_model="qwen3:4b",
    ).generate(_risk(), _xai())

    assert result.provider == "ollama"
    assert request_payload["think"] is False
    assert request_payload["keep_alive"] == "10m"
    assert request_payload["options"] == {"temperature": 0, "num_predict": 512}


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
