"""
Rate limiting via ``slowapi`` — in-memory, no Redis dependency.

Provides a pre-configured ``Limiter`` instance and per-route dependency
helpers.  Limits are configured in ``config.py`` under ``rate_limit_*``.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings

# Global limiter — keyed by client IP via ``get_remote_address``.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_global] if settings.rate_limit_enabled else [],
    enabled=settings.rate_limit_enabled,
)
