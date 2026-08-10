"""
Resilience primitives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class SystemHealth:
    """Snapshot of overall system health for the ``/api/health`` endpoint."""

    status: ServiceStatus
    service: str = "fake_job_detection_api"
    version: str = "2.1.0"

    # Mandatory FP-gate service
    ensemble_members_available: int = 0
    ensemble_members_total: int = 1

    # External services
    ollama_available: bool = False
    database_connected: bool = False

    # Detail
    failed_members: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_analysis_service(cls, service: Any) -> "SystemHealth":
        """Build a SystemHealth snapshot from the live AnalysisService."""
        refresh_readiness = getattr(service, "refresh_readiness", None)
        if callable(refresh_readiness):
            refresh_readiness()
        warm_up = getattr(service, "warm_up_outcomes", {}) or {}
        available = sum(1 for err in warm_up.values() if err is None)
        total = max(len(warm_up), 1)

        if available == 0:
            status = ServiceStatus.UNAVAILABLE
        elif available < total:
            status = ServiceStatus.DEGRADED
        else:
            status = ServiceStatus.HEALTHY

        return cls(
            status=status,
            ensemble_members_available=available,
            ensemble_members_total=total,
            failed_members={k: v for k, v in warm_up.items() if v is not None},
        )
