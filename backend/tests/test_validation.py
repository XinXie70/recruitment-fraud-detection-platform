from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.validation.input_validator import validate_input_text
from backend.validation.job_description_filter import (
    check_job_description_relevance,
    compute_job_relevance_score,
    compute_keyword_score,
)
from backend.validation.pipeline import (
    apply_validation_to_prediction,
    build_rejection_response,
    validate_job_input,
)


@pytest.mark.parametrize(
    ("value", "reason_fragment"),
    [
        (None, "must be a string"),
        ("   ", "cannot be empty"),
        ("https://example.com/jobs", "only a URL"),
        ("12345 !!! ### $$$", "numbers, symbols"),
        ("def calculate_score(value):\n    return value * 2", "code snippet"),
        ("hello!", "casual chat"),
        ("A short but readable sentence.", "too short"),
    ],
)
def test_basic_input_validation_rejects_invalid_categories(value, reason_fragment):
    result = validate_input_text(value)  # type: ignore[arg-type]

    assert result["is_valid"] is False
    assert result["status"] == "invalid_input"
    assert reason_fragment.lower() in result["reason"].lower()


def test_basic_input_validation_accepts_readable_job_text():
    result = validate_input_text(
        "We are hiring an experienced engineer to build reliable software for our customers."
    )

    assert result == {"is_valid": True, "status": "valid", "reason": ""}


def test_keyword_and_relevance_scores_are_bounded_and_penalize_unrelated_topics():
    job_text = (
        "We are looking for a software engineer to join our company. Responsibilities include "
        "building applications. Requirements include Python experience and communication skills. "
        "The permanent role offers salary and benefits from our Sydney office. Apply today."
    )
    unrelated_text = " ".join(
        [
            "This travel article reviews a hotel restaurant movie and sightseeing itinerary.",
            "The news report discusses an election, parliament, and government policy.",
        ]
        * 6
    )

    assert compute_keyword_score(job_text) > 0.5
    assert 0.55 <= compute_job_relevance_score(job_text) <= 1.0
    assert compute_job_relevance_score(unrelated_text) == 0.0


def test_relevance_filter_returns_rejected_warning_and_valid_states():
    rejected = check_job_description_relevance(
        "This detailed travel article compares coastal hotels, restaurants, museums, and films."
    )
    warning = check_job_description_relevance(
        "We are hiring a software engineer for a permanent remote role. Apply with your resume."
    )
    valid = check_job_description_relevance(
        "We are looking for a software engineer to join our company. Responsibilities include "
        "building applications. Requirements include Python experience. Benefits include salary "
        "and flexible work. The successful candidate will work full time in our Sydney office and "
        "should apply with a resume."
    )

    assert rejected["status"] == "not_job_related"
    assert rejected["is_job_related"] is False
    assert warning["status"] == "success_with_warning"
    assert warning["is_job_related"] is True
    assert valid["status"] == "valid"
    assert valid["is_job_related"] is True


def test_validation_pipeline_short_circuits_and_preserves_relevance_result():
    invalid = validate_job_input("hello")
    unrelated = validate_job_input(
        "This travel article reviews hotels, restaurants, museums, beaches, and local films."
    )

    assert invalid["status"] == "invalid_input"
    assert invalid["job_relevance_score"] == 0.0
    assert unrelated["status"] == "not_job_related"
    assert unrelated["job_relevance_score"] < 0.4

    with patch(
        "backend.validation.pipeline.check_job_description_relevance",
        return_value={
            "is_job_related": True,
            "status": "success_with_warning",
            "reason": "Missing common fields.",
            "job_relevance_score": 0.42,
        },
    ):
        result = validate_job_input(
            "We are hiring a developer with relevant professional skills and experience today."
        )

    assert result == {
        "is_valid": True,
        "status": "success_with_warning",
        "reason": "Missing common fields.",
        "job_relevance_score": 0.42,
    }


def test_validation_response_helpers_cover_warning_and_rejection_contracts():
    unrelated = build_rejection_response(
        "fp-gate", {"status": "not_job_related", "reason": "Not a job.", "job_relevance_score": 0.1}
    )
    invalid = build_rejection_response("fp-gate", {"reason": "Invalid."})

    assert unrelated["recommended_action"] == "Please enter a valid job posting or job description."
    assert unrelated["prediction"] is None
    assert invalid["status"] == "invalid_input"
    assert invalid["recommended_action"] == "Please enter a valid job description."

    prediction = {"prediction": 1, "risk_score": 0.8}
    warned = apply_validation_to_prediction(
        {
            "status": "success_with_warning",
            "reason": "Limited detail.",
            "job_relevance_score": 0.45,
        },
        prediction,
    )
    success = apply_validation_to_prediction(
        {"status": "valid", "job_relevance_score": 0.8}, prediction
    )

    assert warned["status"] == "success_with_warning"
    assert warned["message"] == "Limited detail."
    assert success == {
        "status": "success",
        "job_relevance_score": 0.8,
        "prediction": 1,
        "risk_score": 0.8,
    }
