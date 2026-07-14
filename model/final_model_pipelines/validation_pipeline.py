"""
Pre-prediction validation pipeline.

Flow:
  1. input_validator.validate_input_text() checks basic text validity.
  2. job_description_filter.check_job_description_relevance() checks job-topic relevance.
  3. Model inference runs only when both stages pass.
"""

from __future__ import annotations

from typing import Any

from final_model_pipelines.input_validator import validate_input_text
from final_model_pipelines.job_description_filter import check_job_description_relevance


def validate_job_input(text: str) -> dict[str, Any]:
    """
    Run the full pre-prediction validation pipeline.

    Returns:
        {
            "is_valid": bool,
            "status": "success" / "fail" / "invalid_input",
            "reason": str,
        }
    """
    input_result = validate_input_text(text)
    if not input_result["is_valid"]:
        return {
            "is_valid": False,
            "status": input_result["status"],
            "reason": input_result["reason"],
            "relevance_explanation": None,
        }

    relevance_result = check_job_description_relevance(text)
    if not relevance_result["is_job_related"]:
        return {
            "is_valid": False,
            "status": relevance_result["status"],
            "reason": relevance_result["reason"],
            "relevance_explanation": relevance_result.get("relevance_explanation"),
        }

    return {
        "is_valid": True,
        "status": relevance_result["status"],
        "reason": relevance_result["reason"],
        "relevance_explanation": relevance_result.get("relevance_explanation"),
    }


def build_rejection_response(model_name: str, validation: dict[str, Any]) -> dict[str, Any]:
    """Structured response when validation fails and model inference is skipped."""
    status = validation.get("status", "invalid_input")
    recommended_action = (
        "Please enter a valid job posting or job description."
        if status == "fail"
        else "Please enter a valid job description."
    )
    return {
        "status": status,
        "message": validation["reason"],
        "model": model_name,
        "classification_label": None,
        "risk_score": None,
        "prediction": None,
        "recommended_action": recommended_action,
        "relevance_explanation": validation.get("relevance_explanation"),
    }


def apply_validation_to_prediction(
    validation: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Attach validation fields to a successful model prediction."""
    return {
        "status": "success",
        "relevance_explanation": validation.get("relevance_explanation"),
        **prediction,
    }
