"""
Centralised application configuration via Pydantic Settings.

All environment variables are declared once here with types, defaults, and
descriptions.  Other modules import ``settings`` instead of calling
``os.getenv()`` directly.

.. note::
   Add ``pydantic-settings`` to requirements if not already present:
   ``pip install pydantic-settings``
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/fake_job_detection"
    )
    async_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/fake_job_detection"
    )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    secret_key: str = "change-this-secret-key-for-local-development"
    access_token_expire_minutes: int = 120

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: str = (
        "http://127.0.0.1:5190,http://localhost:5190,"
        "https://capstone-project-26t2-9900-h09calmond.onrender.com"
    )

    # ------------------------------------------------------------------
    # Model / Ensemble
    # ------------------------------------------------------------------
    model_timeout_seconds: float = 30.0
    model_max_workers: int = 3
    model_server_url: str = ""
    model_server_timeout: float = 120.0
    ensemble_config_path: str | None = None
    max_input_chars: int = 50000

    # ------------------------------------------------------------------
    # XAI (Explainable AI)
    # ------------------------------------------------------------------
    xai_use_shap: bool = True
    xai_max_items: int = 10
    xai_max_segments: int = 80
    xai_max_evals: int = 200

    # ------------------------------------------------------------------
    # Ollama (optional external LLM for Gentle AI rewrites)
    # ------------------------------------------------------------------
    ollama_enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout: float = 8.0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "json"

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    analysis_cache_ttl_seconds: int = 300  # 5 minutes

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------
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

    @property
    def cors_origin_list(self) -> list[str]:
        """Return CORS origins as a cleaned list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root (parent of backend/)."""
        return Path(__file__).resolve().parent.parent


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — safe to call repeatedly."""
    return Settings()


# Convenience module-level instance (preferred import target).
settings = get_settings()
