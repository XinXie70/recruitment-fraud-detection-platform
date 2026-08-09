"""Configuration safety tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.config import Settings


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


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match="explicit trusted origins"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key="a-secure-production-secret-with-more-than-32-bytes",
            cors_origins="*",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("db_pool_size", 0),
        ("db_max_overflow", -1),
        ("db_pool_timeout_seconds", 0),
        ("db_pool_recycle_seconds", 30),
    ],
)
def test_database_pool_settings_reject_unsafe_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
