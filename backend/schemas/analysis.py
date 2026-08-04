from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from config import settings
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from xai_gentle import EducationItem, GentleAIResult, XAIResult


API_VERSION: Final[Literal["1.0"]] = "1.0"
MAX_INPUT_CHARS = settings.max_input_chars


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Job description text cannot be empty.")
        return cleaned


class ModelMemberOutput(BaseModel):
    key: str
    display_name: str
    status: Literal["success", "error", "timeout"]
    raw_score: float | None = Field(None, ge=0, le=1)
    calibrated_score: float | None = Field(None, ge=0, le=1)
    configured_weight: float = Field(..., ge=0, le=1)
    effective_weight: float = Field(..., ge=0, le=1)
    weighted_contribution: float = Field(..., ge=0, le=1)
    error: str | None = None
    error_code: (
        Literal["not_registered", "artifact_unavailable", "inference_failed", "timeout"]
        | None
    ) = None


class EnsembleResult(BaseModel):
    status: Literal["success", "degraded"]
    risk_score: float = Field(..., ge=0, le=1)
    classification_label: Literal["Likely Legitimate", "Suspicious", "Likely Deceptive"]
    risk_level: Literal["low", "medium", "high"]
    prediction: Literal["real", "fake"]
    recommended_action: Literal["Safe", "Review Required", "High Risk Warning"]
    low_threshold: float = Field(..., ge=0, le=1)
    high_threshold: float = Field(..., ge=0, le=1)
    active_model_count: int = Field(..., ge=1)
    failed_model_count: int = Field(..., ge=0)
    version: str
    fitted: bool
    weight_source: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_score_100(self) -> float:
        """Percentage-scale risk score for API consumers and UI display."""
        return round(self.risk_score * 100, 2)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "EnsembleResult":
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be lower than high_threshold")
        return self


class URLResult(BaseModel):
    url: str
    domain: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    flags: list[str]
    flag_codes: list[str]


class URLAnalysis(BaseModel):
    urls_found: int = Field(..., ge=0)
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    high_risk_count: int = Field(..., ge=0)
    medium_risk_count: int = Field(..., ge=0)
    urls: list[URLResult]
    reasons: list[str]


class AnalysisResponse(BaseModel):
    api_version: Literal["1.0"] = API_VERSION
    phase: Literal["score", "complete"] = "complete"
    status: Literal["success", "degraded"]
    job_relevance_score: float | None = Field(None, ge=0, le=1)
    ensemble: EnsembleResult
    member_outputs: list[ModelMemberOutput]
    xai: XAIResult
    gentle_ai: GentleAIResult
    url_analysis: URLAnalysis


class UserAnalysisHistoryItem(BaseModel):
    id: int
    input_preview: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    status: Literal["success", "degraded"]
    ensemble_available: int = Field(..., ge=0)
    ensemble_total: int = Field(..., ge=0)
    has_result: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class UserAnalysisHistoryPage(BaseModel):
    items: list[UserAnalysisHistoryItem]
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1, le=100)
    total_pages: int = Field(..., ge=0)


class UserAnalysisHistoryDetail(BaseModel):
    id: int
    analysis_result: dict[str, Any]


class EducationListResponse(BaseModel):
    api_version: Literal["1.0"] = API_VERSION
    items: list[EducationItem]
