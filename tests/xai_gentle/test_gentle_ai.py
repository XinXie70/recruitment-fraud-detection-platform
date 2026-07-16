from backend.xai_gentle import (
    EvidenceSpan,
    GentleAIService,
    RiskContext,
    XAIResult,
)


def test_template_fallback_uses_only_structured_evidence():
    risk = RiskContext(
        risk_score=0.82,
        classification_label="Likely Deceptive",
        risk_level="high",
        recommended_action="High Risk Warning",
    )
    xai = XAIResult(
        status="success",
        method="occlusion_fallback",
        output_value=0.82,
        items=[
            EvidenceSpan(
                text="registration fee",
                start=5,
                end=21,
                contribution=0.2,
                direction="raises_risk",
            )
        ],
    )

    result = GentleAIService(ollama_enabled=False).generate(risk, xai)

    assert result.status == "fallback"
    assert result.provider == "template"
    assert result.evidence_explanations[0].text == "registration fee"
    assert "20.0 percentage points" in result.evidence_explanations[0].explanation
    assert "proof" in result.disclaimer.lower()
    assert "fake-job-upfront-payment" in result.learning_item_ids


def test_learning_resources_use_evidence_topic_without_changing_xai_direction():
    risk = RiskContext(
        risk_score=0.2,
        classification_label="Likely Legitimate",
        risk_level="low",
        recommended_action="Safe",
    )
    xai = XAIResult(
        status="success",
        method="shap_partition",
        output_value=0.2,
        items=[
            EvidenceSpan(
                text="registration fee",
                start=0,
                end=16,
                contribution=-0.02,
                direction="lowers_risk",
            )
        ],
    )

    result = GentleAIService(ollama_enabled=False).generate(risk, xai)

    assert result.evidence_explanations[0].direction == "lowers_risk"
    assert "fake-job-upfront-payment" in result.learning_item_ids
