import pytest
from pydantic import ValidationError

from schemas.analysis import EnsembleResult


def test_ensemble_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        EnsembleResult(
            status="success",
            risk_score=0.5,
            classification_label="Suspicious",
            risk_level="medium",
            prediction="fake",
            recommended_action="Review Required",
            low_threshold=0.8,
            high_threshold=0.3,
            active_model_count=1,
            failed_model_count=0,
            version="test",
            fitted=False,
            weight_source="test",
        )
