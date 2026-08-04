"""FastAPI service exposing LR, BERT (class-weighted), and LR+BERT ensemble."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .inference import ModelService
from .risk import build_risk_result
from .schemas import (
    BasePredictionResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictionItem,
    EnsemblePredictionResponse,
    HealthResponse,
    PredictRequest,
    RiskPredictionResponse,
)

service = ModelService(allow_cpu=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        service.load_lr()
        service.load_bert()
        service.load_ensemble_config()
    except FileNotFoundError as exc:
        print(f"Warning: startup preload skipped — {exc}")
    yield


app = FastAPI(
    title="Fake Job Ad Detection API",
    description=(
        "Serve three locked models: Logistic Regression, "
        "BERT (class-weighted), and LR+BERT ensemble with optional risk bands."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=service.models_loaded(),
        device=str(service.device),
    )


@app.post("/predict/lr", response_model=BasePredictionResponse)
def predict_lr(body: PredictRequest) -> BasePredictionResponse:
    try:
        return BasePredictionResponse(**service.predict_lr(body.text))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/bert", response_model=BasePredictionResponse)
def predict_bert(body: PredictRequest) -> BasePredictionResponse:
    try:
        return BasePredictionResponse(**service.predict_bert(body.text))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
def predict_ensemble(body: PredictRequest) -> EnsemblePredictionResponse:
    try:
        return EnsemblePredictionResponse(**service.predict_ensemble(body.text))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/ensemble/risk", response_model=RiskPredictionResponse)
def predict_ensemble_risk(body: PredictRequest) -> RiskPredictionResponse:
    try:
        ensemble = service.predict_ensemble(body.text)
        risk = build_risk_result(ensemble["fraud_score"])
        return RiskPredictionResponse(**ensemble, **risk)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/lr/batch", response_model=BatchPredictResponse)
def predict_lr_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    try:
        service.load_lr()
        items = []
        for index, text in enumerate(body.texts):
            result = service.predict_lr(text)
            items.append(
                BatchPredictionItem(
                    index=index,
                    fraud_score=result["fraud_score"],
                    prediction=result["prediction"],
                    predicted_label=result["predicted_label"],
                )
            )
        return BatchPredictResponse(
            model="logistic_regression_baseline",
            threshold=service._lr_threshold,
            predictions=items,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
