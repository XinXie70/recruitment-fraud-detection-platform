"""Pydantic request/response models for the fraud detection API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Job advertisement text to score")


class BasePredictionResponse(BaseModel):
    model: str
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    prediction: Literal[0, 1]
    predicted_label: Literal["Legitimate", "Fraudulent"]


class EnsemblePredictionResponse(BasePredictionResponse):
    weights: dict[str, float]
    lr_fraud_score: float
    bert_fraud_score: float


class RiskPredictionResponse(EnsemblePredictionResponse):
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: Literal["Low", "Suspicious", "High"]
    binary_threshold: float
    low_suspicious_threshold: float
    suspicious_high_threshold: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    device: str


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictionItem(BaseModel):
    index: int
    fraud_score: float
    prediction: Literal[0, 1]
    predicted_label: Literal["Legitimate", "Fraudulent"]


class BatchPredictResponse(BaseModel):
    model: str
    threshold: float
    predictions: list[BatchPredictionItem]
