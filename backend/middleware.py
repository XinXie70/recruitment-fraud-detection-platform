"""
Middleware — request-id tracing, body validation, and content-type enforcement.
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config import settings

logger = logging.getLogger("fake_job_detection_api")

# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request-id to every response and log context."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id

        logger.info(
            "request.start",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def get_request_id(request: Request) -> str:
    """Dependency that returns the current request-id (or a fallback)."""
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# Request body validation middleware (ASGI-level, wraps receive)
# ---------------------------------------------------------------------------

_MAX_BODY_BYTES = 200 * 1024  # 200 KiB hard cap for any request
_JSON_METHODS = {"POST", "PUT", "PATCH"}


class RequestBodyGuardMiddleware:
    """Enforce Content-Type and maximum body size at ASGI level.

    Applied *before* Pydantic validation so malformed / oversized payloads are
    rejected early with a clear error, without touching the route handlers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        # Only enforce for JSON-bearing methods
        if method not in _JSON_METHODS:
            await self.app(scope, receive, send)
            return

        # Collect body (up to the cap) for validation.
        body_chunks: list[bytes] = []
        total = 0
        more_body = True

        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            more_body = message.get("more_body", False)
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > _MAX_BODY_BYTES:
                await self._send_error(send, 413, "Request body too large (max 200 KiB)")
                # Drain remaining chunks
                while more_body:
                    msg = await receive()
                    more_body = msg.get("more_body", False)
                return
            body_chunks.append(chunk)

        body = b"".join(body_chunks)

        # Validate Content-Type for non-empty bodies
        if body:
            content_type = self._get_header(scope, "content-type")
            if not content_type or "application/json" not in content_type:
                await self._send_error(
                    send, 415, "Content-Type must be application/json"
                )
                return

        # Never log request bodies: authentication payloads and analyzed job
        # advertisements can contain passwords or other sensitive data.

        # Reconstruct receive so downstream can read the body
        async def _wrapped_receive() -> Message:
            nonlocal body_chunks
            if body_chunks:
                chunk = body_chunks.pop(0)
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": bool(body_chunks),
                }
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, _wrapped_receive, send)

    @staticmethod
    def _get_header(scope: Scope, name: str) -> str:
        for key, value in scope.get("headers", []):
            if key == name.encode("latin-1"):
                return value.decode("latin-1")
        return ""

    @staticmethod
    async def _send_error(send: Send, status_code: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
