from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from final_model_pipelines.validation_pipeline import validate_job_input

from schemas.analysis import AnalysisResponse, URLAnalysis
from services.cache import TTLCache
from services.ensemble_predictor import EnsemblePredictor
from services.model_adapter import ModelRegistry
from url_analyzer import analyze_urls
from backend.xai_gentle import GentleAIService, RiskContext, XAIService

logger = logging.getLogger("fake_job_detection_api.analysis")


Validator = Callable[[str], dict[str, Any]]
URLAnalyzer = Callable[[str], dict[str, Any]]
HistorySaver = Callable[[str, AnalysisResponse, int | None], None]


@dataclass(frozen=True)
class InputRejectedError(ValueError):
    status: str
    reason: str
    job_relevance_score: float | None = None

    def __str__(self) -> str:
        return self.reason


def empty_url_analysis(reason: str) -> URLAnalysis:
    return URLAnalysis(
        urls_found=0,
        risk_score=0,
        risk_level="low",
        high_risk_count=0,
        medium_risk_count=0,
        urls=[],
        reasons=[reason],
    )


class AnalysisService:
    def __init__(
        self,
        ensemble: EnsemblePredictor,
        xai: XAIService,
        gentle_ai: GentleAIService,
        validator: Validator = validate_job_input,
        url_analyzer: URLAnalyzer = analyze_urls,
        cache: TTLCache | None = None,
    ):
        self.ensemble = ensemble
        self.xai = xai
        self.gentle_ai = gentle_ai
        self.validator = validator
        self.url_analyzer = url_analyzer
        self.cache = cache or TTLCache()
        self.ready = False
        self.warm_up_outcomes: dict[str, str | None] = {}
        self.on_analysis_complete: HistorySaver | None = None

    @classmethod
    def from_environment(cls) -> "AnalysisService":
        registry = ModelRegistry.default()
        return cls(
            ensemble=EnsemblePredictor.from_environment(registry),
            xai=XAIService(),
            gentle_ai=GentleAIService(),
        )

    def warm_up(self) -> dict[str, str | None]:
        sample = (
            "Software engineer role with clear requirements, company benefits, "
            "and a standard interview process."
        )
        self.warm_up_outcomes = self.ensemble.warm_up(sample)
        self.ready = any(error is None for error in self.warm_up_outcomes.values())
        return dict(self.warm_up_outcomes)

    def analyze(self, text: str) -> AnalysisResponse:
        # Check cache first — avoid re-running the full pipeline for duplicates.
        cache_key = TTLCache.text_key(text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Analysis cache hit", extra={"cache_key": cache_key[:16]})
            return cached

        validation = self.validator(text)
        if not validation.get("is_valid", False):
            raise InputRejectedError(
                status=str(validation.get("status", "invalid_input")),
                reason=str(validation.get("reason", "Input is not a valid job posting.")),
                job_relevance_score=validation.get("job_relevance_score"),
            )

        computation = self.ensemble.predict(text)
        xai_result = self.xai.explain(
            text=text,
            score_batch=computation.score_batch,
            expected_output=computation.ensemble.risk_score,
        )
        risk_context = RiskContext(
            risk_score=computation.ensemble.risk_score,
            risk_level=computation.ensemble.risk_level,
            classification_label=computation.ensemble.classification_label,
            recommended_action=computation.ensemble.recommended_action,
        )
        gentle_result = self.gentle_ai.generate(risk_context, xai_result)

        url_failed = False
        try:
            url_result = URLAnalysis.model_validate(self.url_analyzer(text))
        except Exception as exc:
            url_failed = True
            url_result = empty_url_analysis(
                f"URL analysis was unavailable: {type(exc).__name__}: {exc}"
            )

        degraded = (
            computation.ensemble.status == "degraded"
            or xai_result.status == "unavailable"
            or url_failed
        )
        result = AnalysisResponse(
            status="degraded" if degraded else "success",
            job_relevance_score=validation.get("job_relevance_score"),
            ensemble=computation.ensemble,
            member_outputs=computation.members,
            xai=xai_result,
            gentle_ai=gentle_result,
            url_analysis=url_result,
        )

        # Cache successful results for `analysis_cache_ttl_seconds`.
        self.cache.set(cache_key, result)

        # Persist to analysis history (fire-and-forget via callback).
        if self.on_analysis_complete is not None:
            try:
                self.on_analysis_complete(text, result, None)
            except Exception:
                logger.exception("Failed to persist analysis history")

        return result
