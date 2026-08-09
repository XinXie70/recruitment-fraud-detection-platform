

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


# Configurable weights

CONSENSUS_WEIGHT: float = 0.60  # α  weight for average score
CAUTION_WEIGHT: float = 0.40  # 1 − α  weight for maximum score
BINARY_THRESHOLD: float = 0.40  # Prediction threshold

# Risk tier boundaries
LOW_RISK_THRESHOLD: float = 30.0
HIGH_RISK_THRESHOLD: float = 60.0


@dataclass
class CombinedResult:


    risk_score: float
    risk_level: Literal["low", "medium", "high"]
    prediction: Literal["fake", "legitimate"]
    combined_prob: float
    model_count: int
    fake_count: int
    real_count: int
    all_agree_fake: bool
    all_agree_real: bool
    models_disagree: bool
    average_score: float
    max_score: float
    formula_used: str


def compute_combined_score(model_results: list[dict[str, Any]]) -> CombinedResult:

    valid = [
        m for m in model_results
        if isinstance(m.get("risk_score"), (int, float)) and not (
            isinstance(m.get("risk_score"), float) and
            (m["risk_score"] != m["risk_score"])  # NaN check
        )
    ]

    if not valid:
        return CombinedResult(
            risk_score=0,
            risk_level="low",
            prediction="legitimate",
            combined_prob=0.0,
            model_count=0,
            fake_count=0,
            real_count=0,
            all_agree_fake=False,
            all_agree_real=False,
            models_disagree=False,
            average_score=0.0,
            max_score=0.0,
            formula_used="no valid models",
        )

    # Extract scores and predictions
    scores = [m["risk_score"] for m in valid]
    predictions = [m.get("prediction", "") for m in valid]
    fake_count = sum(1 for p in predictions if p == "fake")
    real_count = sum(1 for p in predictions if p == "real")
    n = len(valid)

    average = sum(scores) / n
    maximum = max(scores)

    all_fake = fake_count == n
    all_real = real_count == n
    disagree = not all_fake and not all_real

    # Compute combined probability
    if disagree:
        combined_prob = CONSENSUS_WEIGHT * average + CAUTION_WEIGHT * maximum
        formula = (
            f"{CONSENSUS_WEIGHT} × avg({average:.4f}) "
            f"+ {CAUTION_WEIGHT} × max({maximum:.4f}) "
            f"= {combined_prob:.4f} (models disagree)"
        )
    else:
        combined_prob = average
        formula = f"avg({average:.4f}) = {combined_prob:.4f} (models agree)"

    # Risk score (0–100)
    risk_score = round(min(100.0, max(0.0, combined_prob * 100)))

    # Risk tier
    if risk_score >= HIGH_RISK_THRESHOLD:
        risk_level: Literal["low", "medium", "high"] = "high"
    elif risk_score >= LOW_RISK_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Binary prediction
    if all_real and combined_prob < BINARY_THRESHOLD:
        prediction: Literal["fake", "legitimate"] = "legitimate"
    elif all_fake:
        prediction = "fake"
    else:
        prediction = "fake" if combined_prob >= BINARY_THRESHOLD else "legitimate"

    return CombinedResult(
        risk_score=risk_score,
        risk_level=risk_level,
        prediction=prediction,
        combined_prob=round(combined_prob, 4),
        model_count=n,
        fake_count=fake_count,
        real_count=real_count,
        all_agree_fake=all_fake,
        all_agree_real=all_real,
        models_disagree=disagree,
        average_score=round(average, 4),
        max_score=round(maximum, 4),
        formula_used=formula,
    )
