

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import settings

# Global limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_global] if settings.rate_limit_enabled else [],
    enabled=settings.rate_limit_enabled,
)
