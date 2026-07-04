"""Pre-prediction validation pipeline."""

from __future__ import annotations

from typing import Any

from final_model_pipelines.input_validator import validate_input_text
from final_model_pipelines.job_description_filter import check_job_description_relevance


def validate_job_input(text: str) -> dict[str, Any]:
    input_result = validate_input_text(text)
    if not input_result["is_valid"]:
        return {
            "is_valid": False,
            "status": input_result["status"],
            "reason": input_result["reason"],
            "job_relevance_score": 0.0,
        }

    relevance_result = check_job_description_relevance(text)
    if not relevance_result["is_job_related"]:
        return {
            "is_valid": False,
            "status": relevance_result["status"],
            "reason": relevance_result["reason"],
            "job_relevance_score": relevance_result["job_relevance_score"],
        }

    return {
        "is_valid": True,
        "status": relevance_result["status"],
        "reason": relevance_result["reason"],
        "job_relevance_score": relevance_result["job_relevance_score"],
    }


def build_rejection_response(model_name: str, validation: dict[str, Any]) -> dict[str, Any]:
    status = validation.get("status", "invalid_input")
    recommended_action = (
        "Please enter a valid job posting or job description."
        if status == "not_job_related"
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
        "job_relevance_score": validation.get("job_relevance_score"),
    }


def apply_validation_to_prediction(
    validation: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    if validation.get("status") == "success_with_warning":
        return {
            "status": "success_with_warning",
            "message": validation["reason"],
            "job_relevance_score": validation.get("job_relevance_score"),
            **prediction,
        }
    return {
        "status": "success",
        "message": "",
        "job_relevance_score": validation.get("job_relevance_score"),
        **prediction,
    }
