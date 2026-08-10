from __future__ import annotations

import logging
import secrets
from typing import Any

from flask import Flask, jsonify, request

from settings import MAX_CONTENT_LENGTH, MODEL_API_KEY
from services.text_utils import ResolvedTexts

logger = logging.getLogger("model_service")

def error_response(message: str, status: int = 400, details: str | None = None):
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status

def server_error(exc: Exception):
    logger.exception("Unhandled prediction error: %s", exc)
    return error_response("Internal server error", status=500)

def parse_payload() -> dict[str, Any]:
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError("Invalid or empty JSON body")
    return payload

def with_meta(result: dict[str, Any], texts: ResolvedTexts) -> dict[str, Any]:
    output = {"ok": True, **result}
    if texts.get("record_id"):
        output["record_id"] = texts["record_id"]
    return output

def _extract_api_key() -> str | None:
    header_key = request.headers.get("X-API-Key")
    if header_key:
        return header_key.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None

def register_http_handlers(app: Flask) -> None:
    @app.before_request
    def require_api_key():
        if not request.path.startswith("/predict") or not MODEL_API_KEY:
            return None
        provided = _extract_api_key()
        if not provided or not secrets.compare_digest(provided, MODEL_API_KEY):
            return error_response("Unauthorized", status=401)
        return None

    @app.errorhandler(413)
    def request_entity_too_large(_exc):
        return error_response(
            f"Request body exceeds limit of {MAX_CONTENT_LENGTH} bytes", status=413
        )
