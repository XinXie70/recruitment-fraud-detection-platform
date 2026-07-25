"""Security regression tests for HTTP middleware."""

from __future__ import annotations

import logging


def test_authentication_body_is_not_written_to_logs(client, caplog) -> None:
    password = "unique-secret-that-must-not-be-logged"

    with caplog.at_level(logging.INFO):
        client.post(
            "/api/auth/login",
            json={"identifier": "missing-user", "password": password},
        )

    assert password not in caplog.text
