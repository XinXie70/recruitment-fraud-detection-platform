

from __future__ import annotations

from functools import lru_cache

from services.analysis_service import AnalysisService
from services.cache import TTLCache
from services.ensemble_predictor import EnsemblePredictor
from services.model_adapter import ModelRegistry
from xai_gentle import GentleAIService, XAIService


@lru_cache()
def get_model_registry() -> ModelRegistry:

    return ModelRegistry.default()


@lru_cache()
def get_ensemble_predictor() -> EnsemblePredictor:

    return EnsemblePredictor.from_environment(get_model_registry())


@lru_cache()
def get_xai_service() -> XAIService:

    return XAIService()


@lru_cache()
def get_gentle_ai_service() -> GentleAIService:

    return GentleAIService()


@lru_cache()
def get_cache() -> TTLCache:

    return TTLCache()


@lru_cache()
def get_analysis_service() -> AnalysisService:

    return AnalysisService(
        ensemble=get_ensemble_predictor(),
        xai=get_xai_service(),
        gentle_ai=get_gentle_ai_service(),
        cache=get_cache(),
    )
