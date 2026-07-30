from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AnalysisHistoryItem(BaseModel):
    id: int
    input_preview: str
    risk_score: float
    risk_level: str
    status: str
    ensemble_available: int
    ensemble_total: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int


class AdminStats(BaseModel):
    total_users: int
    total_analyses: int
    analyses_today: int
    avg_risk_score: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class AdminUserItem(BaseModel):
    id: int
    email: str
    username: str
    is_admin: bool
    analysis_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelMetric(BaseModel):
    model: str
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    category: Literal["classic", "dl", "transformer"]


class ModelMetricsResponse(BaseModel):
    version: str
    dataset: str
    models: list[ModelMetric]
