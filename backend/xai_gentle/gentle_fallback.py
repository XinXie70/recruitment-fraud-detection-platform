from __future__ import annotations

from .contracts import (
    EducationItem,
    GentleAIResult,
    GentleEvidenceExplanation,
    RiskContext,
    XAIResult,
)


DISCLAIMER = (
    "This is educational decision support, not proof that an advertisement or "
    "company is genuine or deceptive. Verify important details independently."
)


def _summary_for(risk: RiskContext) -> str:
    score = round(risk.risk_score * 100)
    if risk.classification_label == "Likely Deceptive":
        return (
            f"The ensemble produced a {score}% risk score. Several model signals "
            "support taking extra care before continuing with this job offer."
        )
    if risk.classification_label == "Suspicious":
        return (
            f"The ensemble produced a {score}% risk score. The result is uncertain "
            "enough that the employer and offer should be checked carefully."
        )
    return (
        f"The ensemble produced a {score}% risk score. Fewer concerning model "
        "signals were found, but normal employer verification is still important."
    )


def _evidence_explanations(xai: XAIResult) -> list[GentleEvidenceExplanation]:
    explanations: list[GentleEvidenceExplanation] = []
    for item in xai.items:
        percentage_points = abs(item.contribution) * 100
        magnitude = (
            "less than 0.1"
            if percentage_points < 0.1
            else f"{percentage_points:.1f}"
        )
        if item.direction == "raises_risk":
            explanation = (
                "The XAI attribution assigned this text approximately "
                f"{magnitude} percentage points in the higher-risk direction. "
                "It is a model contribution, not independent proof of deception."
            )
        else:
            explanation = (
                "The XAI attribution assigned this text approximately "
                f"{magnitude} percentage points in the lower-risk direction. "
                "It does not by itself verify the employer or offer."
            )
        explanations.append(
            GentleEvidenceExplanation(
                text=item.text,
                start=item.start,
                end=item.end,
                direction=item.direction,
                explanation=explanation,
            )
        )
    return explanations


def build_template_guidance(
    risk: RiskContext,
    xai: XAIResult,
    learning_items: list[EducationItem],
    message: str | None = None,
) -> GentleAIResult:
    next_steps: list[str] = []
    for item in learning_items:
        for action in item.best_practices:
            if action not in next_steps:
                next_steps.append(action)
            if len(next_steps) == 4:
                break
        if len(next_steps) == 4:
            break

    return GentleAIResult(
        status="fallback",
        provider="template",
        summary=_summary_for(risk),
        evidence_explanations=_evidence_explanations(xai),
        next_steps=next_steps,
        learning_item_ids=[item.id for item in learning_items],
        disclaimer=DISCLAIMER,
        message=message,
    )
