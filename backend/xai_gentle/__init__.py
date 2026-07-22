"""Public interface for the independently owned XAI and Gentle AI module."""

from .contracts import (
    EducationItem,
    EducationTopic,
    EvidenceSpan,
    GentleAIResult,
    GentleEvidenceExplanation,
    RiskContext,
    XAIResult,
)
from .gentle_ai_service import GentleAIService
from .xai_service import BatchScorer, XAIService

__all__ = [
    "BatchScorer",
    "EducationTopic",
    "EducationItem",
    "EvidenceSpan",
    "GentleAIResult",
    "GentleAIService",
    "GentleEvidenceExplanation",
    "RiskContext",
    "XAIResult",
    "XAIService",
]
