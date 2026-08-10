from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.rate_limit import limiter
from backend.services.resilience import ServiceStatus, SystemHealth


class ReadyResponse(BaseModel):
    status: str
    model_ready: bool
    database_connected: bool


def create_health_router(
    analysis_service: Any,
    database_check: Callable[[], bool],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/live", include_in_schema=False)
    def liveness_check() -> dict[str, str]:
        return {"status": "alive"}

    @router.get("/api/health")
    @limiter.limit(settings.rate_limit_global)
    def health_check(request: Request):
        health = SystemHealth.from_analysis_service(analysis_service)
        health.database_connected = database_check()
        health.ollama_available = analysis_service.gentle_ai.ollama_enabled
        status_code = 503 if health.status == ServiceStatus.UNAVAILABLE else 200
        return JSONResponse(
            status_code=status_code,
            content={
                "status": health.status.value,
                "service": health.service,
                "version": health.version,
                "ensemble": {
                    "available": health.ensemble_members_available,
                    "total": health.ensemble_members_total,
                    "failed_members": sorted(health.failed_members),
                },
                "database_connected": health.database_connected,
                "ollama_available": health.ollama_available,
            },
        )

    @router.get("/api/ready", response_model=ReadyResponse)
    def readiness_check() -> ReadyResponse:
        model_ready = analysis_service.refresh_readiness()
        database_connected = database_check()
        if not model_ready or not database_connected:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service dependencies are not ready.",
                    "model_ready": model_ready,
                    "database_connected": database_connected,
                },
            )
        return ReadyResponse(
            status="ready",
            model_ready=True,
            database_connected=True,
        )

    return router
