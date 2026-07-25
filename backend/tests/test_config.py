"""Configuration safety tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Settings


def test_production_rejects_default_secret() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key="change-this-secret-key-for-local-development",
        )


def test_production_accepts_long_random_secret() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        secret_key="a-secure-production-secret-with-more-than-32-bytes",
    )
    assert settings.app_env == "production"
