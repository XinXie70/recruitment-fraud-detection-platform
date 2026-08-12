from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

from .contracts import (
    EducationItem,
    EvidenceSpan,
    GentleAIResult,
    GentleEvidenceExplanation,
    RiskContext,
    XAIResult,
)

# Fixed Disclaimer
DISCLAIMER = (
    "This is educational decision support, not proof that an advertisement or "
    "company is genuine or deceptive. Verify important details independently."
)


@dataclass(frozen=True)
class PhraseMeaning:
    """A cautious, local interpretation used to explain one SHAP phrase."""

    expected_direction: Literal["raises_risk", "lowers_risk"]
    supporting_reason: str
    conflicting_reason: str


PHRASE_MEANINGS = (
    (
        re.compile(
            r"\b(?:fee|payment|deposit|top[ -]?up|gift card|cryptocurrency|bitcoin|usdt|"
            r"registration fee|training kit)\b|\bpay\s+(?:a|an|\$|\d)",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="raises_risk",
            supporting_reason=(
                "It describes money, a fee, or something the applicant may need to pay "
                "before starting. The local job-scam guidance treats up-front payment "
                "requests as a warning sign."
            ),
            conflicting_reason=(
                "It appears to describe an up-front payment, which the local job-scam "
                "guidance treats as a warning sign. SHAP nevertheless moved the model "
                "toward lower risk in this exact context, so this phrase should not be "
                "treated as reassurance."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:unlimited earnings?|guaranteed (?:income|earnings?)|high income|"
            r"easy money|financial freedom|earn \$?\d+|\$\d+\+? per (?:day|hour)|"
            r"little effort)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="raises_risk",
            supporting_reason=(
                "It presents unusually large or open-ended earnings. The local guidance "
                "identifies high-income promises with little effort as a job-scam warning sign."
            ),
            conflicting_reason=(
                "It contains an unusually strong earnings promise, which the local guidance "
                "treats as a warning sign. SHAP moved the model toward lower risk here, so the "
                "model direction conflicts with the educational guidance."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:urgent(?: hiring)?|immediate start|act now|apply now|start today|"
            r"limited time|immediately)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="raises_risk",
            supporting_reason=(
                "It uses urgency or asks the applicant to act quickly. The local guidance "
                "notes that time pressure can reduce a person's opportunity to verify an offer."
            ),
            conflicting_reason=(
                "It uses urgency, which the local guidance treats as a reason to pause and "
                "verify. SHAP moved the model toward lower risk in this context, so that "
                "lower-risk direction is not independent reassurance."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:no interview|no experience|no degree|no qualifications?|"
            r"without (?:an )?interview)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="raises_risk",
            supporting_reason=(
                "It reduces or removes a normal hiring check such as an interview, experience, "
                "or qualifications. The local guidance recommends verifying offers with little "
                "or no hiring process."
            ),
            conflicting_reason=(
                "It describes a reduced hiring check, which the local guidance says deserves "
                "verification. SHAP moved the model toward lower risk here, so the phrase "
                "should still be checked rather than treated as proof of legitimacy."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:whatsapp|telegram|signal|gmail|yahoo|personal email|message us)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="raises_risk",
            supporting_reason=(
                "It points to a messaging app or personal email channel. The local guidance "
                "recommends independently verifying recruiters who avoid official company channels."
            ),
            conflicting_reason=(
                "It points to a messaging app or personal email channel, which the local guidance "
                "says should be independently verified. SHAP lowered the model score in this exact "
                "context, but that does not verify the contact."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:passport|driver'?s? licence|bank(?:ing)? details|identity documents?|"
            r"tax file number|tfn|verification code)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="raises_risk",
            supporting_reason=(
                "It refers to sensitive identity, banking, or account information. The local "
                "guidance advises applicants not to share these details before verifying a recruiter."
            ),
            conflicting_reason=(
                "It refers to sensitive personal information. Although SHAP moved the model toward "
                "lower risk in this context, the local guidance still recommends verifying the "
                "request before sharing anything."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:abn|official website|company website|head office|business address|"
            r"security licence|master licence|applications? close|formal interview|references?)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="lowers_risk",
            supporting_reason=(
                "It provides a concrete organisation, contact, licence, or recruitment detail. "
                "In this advertisement SHAP associated that verifiable detail with lower model risk, "
                "although the detail still needs independent checking."
            ),
            conflicting_reason=(
                "It looks like a concrete organisation or recruitment detail, but SHAP moved the "
                "model toward higher risk in this advertisement. That model association does not "
                "make the detail suspicious by itself."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:salary packaging|take-home pay|employee benefits?|great discounts?|"
            r"well-?being program|part[ -]?time|full[ -]?time|contract position|"
            r"shift|roster(?:ing)?|working hours?|annual leave|superannuation)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="lowers_risk",
            supporting_reason=(
                "It describes a concrete employment condition or employee benefit. In this "
                "advertisement SHAP associated that ordinary job detail with lower model risk; "
                "it does not verify that the offer is genuine."
            ),
            conflicting_reason=(
                "It describes an employment condition or benefit, but SHAP moved the model toward "
                "higher risk in this particular advertisement. The direction is context-specific "
                "and does not make the term a warning sign on its own."
            ),
        ),
    ),
    (
        re.compile(
            r"\b(?:responsibilit(?:y|ies)|duties|qualifications?|skills?|experience required|"
            r"customer service|administration|supervis(?:e|ion)|reporting to|based in)\b",
            re.IGNORECASE,
        ),
        PhraseMeaning(
            expected_direction="lowers_risk",
            supporting_reason=(
                "It gives a specific duty, skill, or role requirement. SHAP associated this level "
                "of job detail with lower model risk in the surrounding advertisement, but the "
                "employer must still be verified."
            ),
            conflicting_reason=(
                "It describes a duty, skill, or role requirement, yet SHAP moved the model toward "
                "higher risk in this context. The phrase is a model signal here, not a standalone "
                "real-world warning sign."
            ),
        ),
    ),
)

# Fixed explanation template
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

# Match safety education
def _matching_education_indicator(
    evidence_text: str,
    learning_items: list[EducationItem],
) -> tuple[EducationItem, str] | None:
    normalised_text = " ".join(evidence_text.lower().split())
    matches = [
        (item, term)
        for item in learning_items
        for term in item.indicator_terms
        if term.lower() in normalised_text
    ]
    if not matches:
        return None
    return max(matches, key=lambda match: len(match[1]))

def _phrase_meaning(evidence_text: str) -> PhraseMeaning | None:
    return next(
        (
            meaning
            for pattern, meaning in PHRASE_MEANINGS
            if pattern.search(evidence_text)
        ),
        None,
    )

# Explanation sequence
def _specific_explanation(
    item: EvidenceSpan,
    learning_items: list[EducationItem],
) -> str:
    education_match = _matching_education_indicator(item.text, learning_items)
    meaning = _phrase_meaning(item.text)

    if meaning is not None:
        if meaning.expected_direction == item.direction:
            reason = meaning.supporting_reason
        else:
            reason = meaning.conflicting_reason
    elif education_match is not None:
        education_item, indicator = education_match
        reason = (
            f'The phrase contains "{indicator}", which matches the local educational topic '
            f'"{education_item.title}". The SHAP direction describes how the model used it in '
            "this advertisement, not a separate judgement about the offer."
        )
    else:
        direction = "higher" if item.direction == "raises_risk" else "lower"
        reason = (
            f'SHAP found that the exact phrase "{item.text}" moved the model toward {direction} '
            "risk in the surrounding advertisement. It does not match a specific local safety "
            "indicator, so the system does not assign it an independent real-world meaning."
        )

    if item.direction == "raises_risk":
        return reason + " It is a model contribution, not independent proof of deception."
    return reason + " A lower-risk model signal does not verify the employer or offer."


def _evidence_explanations(
    xai: XAIResult,
    learning_items: list[EducationItem],
) -> list[GentleEvidenceExplanation]:
    explanations: list[GentleEvidenceExplanation] = []
    for item in xai.items:
        explanations.append(
            GentleEvidenceExplanation(
                text=item.text,
                start=item.start,
                end=item.end,
                direction=item.direction,
                explanation=_specific_explanation(item, learning_items),
            )
        )
    return explanations

# Construct the interpretation result
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
        evidence_explanations=_evidence_explanations(xai, learning_items),
        next_steps=next_steps,
        learning_item_ids=[item.id for item in learning_items],
        disclaimer=DISCLAIMER,
        message=message,
    )
