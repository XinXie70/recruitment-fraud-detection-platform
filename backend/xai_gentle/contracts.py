from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RiskContext(BaseModel):
    """Stable ensemble fields Gentle AI is allowed to read."""

    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    classification_label: Literal[
        "Likely Legitimate", "Suspicious", "Likely Deceptive"
    ]
    recommended_action: Literal["Safe", "Review Required", "High Risk Warning"]


class EvidenceSpan(BaseModel):
    text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)
    contribution: float
    direction: Literal["raises_risk", "lowers_risk"]

    @model_validator(mode="after")
    def range_and_direction_are_valid(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.direction == "raises_risk" and self.contribution <= 0:
            raise ValueError("raises_risk evidence needs a positive contribution")
        if self.direction == "lowers_risk" and self.contribution >= 0:
            raise ValueError("lowers_risk evidence needs a negative contribution")
        return self


class XAIResult(BaseModel):
    status: Literal["success", "unavailable"]
    method: Literal["shap_partition", "occlusion_fallback", "unavailable"]
    target: Literal["ensemble_fake_probability"] = "ensemble_fake_probability"
    version: str = "xai-v1"
    base_value: float | None = None
    output_value: float | None = Field(None, ge=0, le=1)
    items: list[EvidenceSpan] = Field(default_factory=list)
    message: str | None = None


class GentleEvidenceExplanation(BaseModel):
    text: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)
    direction: Literal["raises_risk", "lowers_risk"]
    explanation: str


class GentleAIResult(BaseModel):
    status: Literal["success", "fallback"]
    provider: Literal["template", "ollama"]
    summary: str
    evidence_explanations: list[GentleEvidenceExplanation] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    learning_item_ids: list[str] = Field(default_factory=list)
    disclaimer: str
    message: str | None = None
    version: str = "gentle-v1"


class EducationItem(BaseModel):
    id: str
    topic: Literal["fake_jobs", "misinformation", "phishing", "scam_patterns"]
    title: str
    summary: str
    warning_signs: list[str]
    example: str
    best_practices: list[str]
    indicator_terms: list[str] = Field(default_factory=list, exclude=True)
    source_name: str
    source_url: str
