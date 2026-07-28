from unittest.mock import Mock, patch

import pytest

from final_model_pipelines.validation_pipeline import validate_job_input
from services.analysis_service import AnalysisService, InputRejectedError


def build_service() -> AnalysisService:
    """Build the smallest service needed to test pre-inference validation."""
    return AnalysisService(
        ensemble=Mock(),
        xai=Mock(),
        gentle_ai=Mock(),
        url_analyzer=Mock(),
    )


UNRELATED_TEXT = """
This travel article describes a weekend visit to a coastal town. The itinerary
begins with breakfast near the harbour, followed by sightseeing at a historic
museum and a walk along the beach. In the evening, visitors can choose a local
restaurant and watch a film. The article also reviews several hotels, public
transport options, popular tourist attractions, and seasonal events for
families.
"""

VALID_JOB_TEXT = """
We are hiring a software engineer to join our established technology company.
Responsibilities include developing web applications, reviewing code, and
working with product teams. Requirements include professional programming
experience, communication skills, and knowledge of Python. The successful
candidate will receive a competitive salary and benefits, complete formal
interviews, and work full time from our Sydney office.
"""


def test_unrelated_text_is_rejected_before_model_prediction():
    service = build_service()
    service.validator = validate_job_input

    with patch.object(service.ensemble, "predict") as predict:
        with pytest.raises(InputRejectedError) as error:
            service.analyze(UNRELATED_TEXT)

    predict.assert_not_called()
    assert error.value.status == "not_job_related"
    assert "does not appear to be a job posting" in error.value.reason


def test_valid_job_advertisement_still_passes_validation():
    result = validate_job_input(VALID_JOB_TEXT)

    assert result["is_valid"] is True
    assert result["status"] in {"valid", "success_with_warning"}
    assert result["job_relevance_score"] >= 0.4


def test_short_input_returns_structured_invalid_input():
    result = validate_job_input("This is only a short sentence.")

    assert result["is_valid"] is False
    assert result["status"] == "invalid_input"
    assert "too short" in result["reason"].lower()
