"""Security regression tests for HTTP middleware."""

from __future__ import annotations

import logging
import re

from starlette.requests import Request

from backend import rate_limit


def _request(headers: list[tuple[bytes, bytes]], client: tuple[str, int]) -> Request:
    return Request({"type": "http", "headers": headers, "client": client})


def test_rate_limit_uses_token_bucket_for_authenticated_proxy_users() -> None:
    first = _request(
        [(b"authorization", b"Bearer first-token")],
        ("shared-proxy", 443),
    )
    second = _request(
        [(b"authorization", b"Bearer second-token")],
        ("shared-proxy", 443),
    )

    assert rate_limit.get_client_address(first).startswith("token:")
    assert rate_limit.get_client_address(first) != rate_limit.get_client_address(second)


def test_forwarded_address_requires_explicit_proxy_trust(monkeypatch) -> None:
    request = _request(
        [(b"x-forwarded-for", b"203.0.113.8, 10.0.0.2")],
        ("trusted-proxy", 443),
    )
    monkeypatch.setattr(rate_limit.settings, "trust_proxy_headers", False)
    assert rate_limit.get_client_address(request) == "trusted-proxy"

    monkeypatch.setattr(rate_limit.settings, "trust_proxy_headers", True)
    assert rate_limit.get_client_address(request) == "203.0.113.8"

def test_authentication_body_is_not_written_to_logs(client, caplog) -> None:
    password = "unique-secret-that-must-not-be-logged"

    with caplog.at_level(logging.INFO):
        client.post(
            "/api/auth/login",
            json={"identifier": "missing-user", "password": password},
        )

    assert password not in caplog.text


def test_security_headers_are_applied(client) -> None:
    response = client.get("/api/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_authentication_responses_are_not_cacheable(client) -> None:
    response = client.post(
        "/api/auth/login",
        json={"identifier": "missing-user", "password": "not-a-password"},
    )

    assert response.headers["Cache-Control"] == "no-store"


def test_unsafe_request_id_is_replaced(client) -> None:
    response = client.get(
        "/api/health",
        headers={"X-Request-ID": "attacker controlled value"},
    )

    assert re.fullmatch(r"[0-9a-f]{8}", response.headers["X-Request-ID"])


def test_request_completion_is_logged(client, caplog) -> None:
    with caplog.at_level(logging.INFO):
        response = client.get("/api/live", headers={"X-Request-ID": "trace-123"})

    completion = next(
        record for record in caplog.records if record.getMessage() == "request.complete"
    )
    assert completion.request_id == "trace-123"
    assert completion.status_code == response.status_code
    assert completion.duration_ms >= 0
