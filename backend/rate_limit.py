

from __future__ import annotations

import hashlib

from slowapi import Limiter
from fastapi import Request
from slowapi.util import get_remote_address

from backend.config import settings

def get_client_address(request: Request) -> str:
    """Return the proxy-provided client IP only when explicitly trusted."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        # Authenticated users receive independent buckets even when every
        # request reaches the API through the same frontend reverse proxy.
        return "token:" + hashlib.sha256(authorization.encode()).hexdigest()
    if settings.trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        client_address = forwarded_for.split(",", 1)[0].strip()
        if client_address:
            return client_address
    return get_remote_address(request)


# Global limiter
limiter = Limiter(
    key_func=get_client_address,
    default_limits=[settings.rate_limit_global] if settings.rate_limit_enabled else [],
    enabled=settings.rate_limit_enabled,
)
