"""
Dependency injection — FastAPI ``Depends`` callables that replace module-level
singletons, making the system testable and the lifecycle explicit.

All dependencies are cached with ``lru_cache`` so they act as process-level
singletons while remaining replaceable via ``app.dependency_overrides`` in tests.
"""

from __future__ import annotations

from functools import lru_cache

from services.analysis_service import AnalysisService
from services.cache import TTLCache
from services.ensemble_predictor import EnsemblePredictor
from services.model_adapter import ModelRegistry
from xai_gentle import GentleAIService, XAIService


@lru_cache()
def get_model_registry() -> ModelRegistry:
    """Cached singleton of the model registry."""
    return ModelRegistry.default()


@lru_cache()
def get_ensemble_predictor() -> EnsemblePredictor:
    """Cached singleton of the ensemble predictor."""
    return EnsemblePredictor.from_environment(get_model_registry())


@lru_cache()
def get_xai_service() -> XAIService:
    """Cached singleton of the XAI service."""
    return XAIService()


@lru_cache()
def get_gentle_ai_service() -> GentleAIService:
    """Cached singleton of the Gentle AI (education) service."""
    return GentleAIService()


@lru_cache()
def get_cache() -> TTLCache:
    """Cached singleton of the analysis result cache."""
    return TTLCache()


@lru_cache()
def get_analysis_service() -> AnalysisService:
    """Compose the full AnalysisService from its sub-dependencies.

    ``lru_cache`` ensures ML models are loaded exactly once per process.
    FastAPI ``Depends(get_analysis_service)`` keeps the dependency explicit
    and replaceable in tests.
    """
    return AnalysisService(
        ensemble=get_ensemble_predictor(),
        xai=get_xai_service(),
        gentle_ai=get_gentle_ai_service(),
        cache=get_cache(),
    )
