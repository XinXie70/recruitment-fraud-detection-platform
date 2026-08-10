

from __future__ import annotations

from functools import lru_cache

from backend.config import settings
from backend.services.analysis_service import AnalysisService
from backend.services.cache import TTLCache
from backend.services.fp_gate_predictor import FPGatePredictor
from backend.xai_gentle import GentleAIService, XAIService


@lru_cache()
def get_ensemble_predictor() -> FPGatePredictor:
    return FPGatePredictor(
        settings.required_model_server_url,
        timeout_seconds=settings.model_server_timeout,
    )


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
