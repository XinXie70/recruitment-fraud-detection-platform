"""
Risk Mapping Layer — post-processing mapping from binary risk_score.

Models are trained and inferred as binary classifiers (real / fake), outputting
the predicted probability of a fake job as risk_score.
This module maps risk_score to three-tier frontend display labels only; it does
not retrain the model.
"""

from __future__ import annotations

from typing import Any

LABEL_LIKELY_LEGITIMATE = "Likely Legitimate"
LABEL_SUSPICIOUS = "Suspicious"
LABEL_LIKELY_DECEPTIVE = "Likely Deceptive"

ACTION_SAFE = "Safe"
ACTION_REVIEW = "Review Required"
ACTION_HIGH_RISK = "High Risk Warning"


def apply_risk_mapping(
    risk_score: float,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, str]:
    """
    Map binary risk_score to a three-tier risk level and companion fields.

    Rules (LOW / HIGH tuned on validation set using PR curve and related metrics):
      score < LOW          → Likely Legitimate,  prediction=real,  Safe
      LOW <= score < HIGH  → Suspicious,         prediction=fake,  Review Required
      score >= HIGH        → Likely Deceptive,   prediction=fake,  High Risk Warning
    """
    if risk_score < low_threshold:
        return {
            "classification_label": LABEL_LIKELY_LEGITIMATE,
            "prediction": "real",
            "recommended_action": ACTION_SAFE,
        }
    if risk_score < high_threshold:
        return {
            "classification_label": LABEL_SUSPICIOUS,
            "prediction": "fake",
            "recommended_action": ACTION_REVIEW,
        }
    return {
        "classification_label": LABEL_LIKELY_DECEPTIVE,
        "prediction": "fake",
        "recommended_action": ACTION_HIGH_RISK,
    }


def build_structured_output(
    model_name: str,
    risk_score: float,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    """Full API response: risk_score + risk mapping post-processing results."""
    mapped = apply_risk_mapping(risk_score, low_threshold, high_threshold)
    return {
        "model": model_name,
        "risk_score": round(float(risk_score), 4),
        "classification_label": mapped["classification_label"],
        "prediction": mapped["prediction"],
        "recommended_action": mapped["recommended_action"],
    }
