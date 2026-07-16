import pytest
from pydantic import ValidationError

from backend.xai_gentle import EvidenceSpan, RiskContext


def test_evidence_span_requires_consistent_offsets_and_direction():
    evidence = EvidenceSpan(
        text="registration fee",
        start=10,
        end=26,
        contribution=0.2,
        direction="raises_risk",
    )
    assert evidence.end > evidence.start

    with pytest.raises(ValidationError):
        EvidenceSpan(
            text="fee",
            start=10,
            end=8,
            contribution=0.2,
            direction="raises_risk",
        )


def test_gentle_risk_context_contains_only_stable_ensemble_fields():
    risk = RiskContext(
        risk_score=0.82,
        risk_level="high",
        classification_label="Likely Deceptive",
        recommended_action="High Risk Warning",
    )

    assert set(risk.model_dump()) == {
        "risk_score",
        "risk_level",
        "classification_label",
        "recommended_action",
    }
