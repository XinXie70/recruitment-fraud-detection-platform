"""Database-level integrity tests."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from models import AnalysisHistory, User


def test_analysis_history_rejects_invalid_risk_score(db_session) -> None:
    user = User(
        email="constraints@example.com",
        username="constraints-user",
        password_hash="not-used-in-this-test",
    )
    db_session.add(user)
    db_session.flush()
    db_session.add(
        AnalysisHistory(
            user_id=user.id,
            input_preview="Listing",
            input_hash="f" * 64,
            risk_score=1.5,
            risk_level="high",
            status="success",
            ensemble_available=1,
            ensemble_total=1,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()
