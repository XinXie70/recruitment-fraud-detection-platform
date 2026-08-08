

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["development", "test", "production"] = "development"


    # Database

    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/fake_job_detection"
    )
    async_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/fake_job_detection"
    )
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_pool_timeout_seconds: int = Field(default=10, ge=1, le=120)
    db_pool_recycle_seconds: int = Field(default=1800, ge=60, le=86400)


    # Authentication

    secret_key: str = "change-this-secret-key-for-local-development"
    access_token_expire_minutes: int = 120
    jwt_issuer: str = "fake-job-detection-api"
    jwt_audience: str = "fake-job-detection-client"

    # Dedicated administrator account.
    admin_email: str = ""
    admin_username: str = ""
    admin_password: str = ""


    # CORS

    cors_origins: str = (
        "http://127.0.0.1:5190,http://localhost:5190,"
        "https://capstone-project-26t2-9900-h09c-almond.onrender.com"
    )


    # Model / Ensemble

    model_timeout_seconds: float = 30.0
    model_max_workers: int = 3
    model_server_url: str = ""
    model_server_timeout: float = 120.0
    ensemble_config_path: str | None = None
    max_input_chars: int = 50000


    # XAI

    xai_max_items: int = Field(default=10, ge=1, le=50)
    xai_max_evals: int = Field(default=200, ge=8)
    xai_hierarchical_min_words: int = Field(default=250, ge=1)
    xai_hierarchical_top_segments: int = Field(default=2, ge=1)
    xai_hierarchical_segment_words: int = Field(default=35, ge=1)
    xai_hierarchical_max_evals: int = Field(default=160, ge=8)


    # Ollama

    ollama_enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout: float = 8.0


    # Observability

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "json"


    # Cache

    analysis_cache_ttl_seconds: int = 300  # 5 minutes
    analysis_cache_max_entries: int = 1024


    # Rate Limiting

    rate_limit_enabled: bool = True
    rate_limit_global: str = "60/minute"  # per-IP global limit
    rate_limit_auth_login: str = "10/minute"  # stricter for login
    rate_limit_auth_register: str = "5/minute"  # strictest for registration
    rate_limit_analyze: str = "20/minute"  # per-IP analysis limit

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def production_secrets_are_safe(self) -> "Settings":
        if self.app_env != "production":
            return self
        if (
            self.secret_key == "change-this-secret-key-for-local-development"
            or len(self.secret_key.encode("utf-8")) < 32
        ):
            raise ValueError("Production SECRET_KEY must contain at least 32 bytes.")
        if "*" in self.cors_origin_list:
            raise ValueError(
                "Production CORS_ORIGINS must list explicit trusted origins; '*' is not allowed."
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        """Return CORS origins as a cleaned list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root ."""
        return Path(__file__).resolve().parent.parent


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton safe to call repeatedly."""
    return Settings()


# Convenience module-level
settings = get_settings()
