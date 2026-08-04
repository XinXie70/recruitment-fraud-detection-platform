

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger("fake_job_detection_api")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# Request ID middleware



class RequestIDMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if _SAFE_REQUEST_ID.fullmatch(supplied_request_id)
            else str(uuid.uuid4())[:8]
        )
        request.state.request_id = request_id

        logger.info(
            "request.start",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        started_at = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logger.info(
                "request.complete",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply baseline browser security headers to every API response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith("/api/auth"):
            response.headers["Cache-Control"] = "no-store"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


def get_request_id(request: Request) -> str:

    return getattr(request.state, "request_id", "unknown")



# Request body validation middleware


_MAX_BODY_BYTES = 200 * 1024  # 200 KiB hard cap
_JSON_METHODS = {"POST", "PUT", "PATCH"}
_FORM_ENDPOINTS = {"/api/auth/token"}


class RequestBodyGuardMiddleware:


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

        # Collect body for validation.
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
                await self._send_error(send, 413, "Request body too large ")
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
            path = scope.get("path", "")
            accepts_form = (
                path in _FORM_ENDPOINTS
                and "application/x-www-form-urlencoded" in content_type
            )
            if "application/json" not in content_type and not accepts_form:
                await self._send_error(
                    send, 415, "Unsupported Content-Type"
                )
                return


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
