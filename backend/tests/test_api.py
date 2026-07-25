"""Tests for core API endpoints."""

from __future__ import annotations

import pytest
from jose import jwt

from config import settings


class TestHealthEndpoints:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        # 200 = healthy, 503 = degraded (models not fully loaded).
        # Both are valid depending on model artifact availability.
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "status" in data
        assert "ensemble" in data

    def test_ready_returns_503_when_models_not_warm(self, client):
        resp = client.get("/api/ready")
        # Models are not warmed up in test — expect 503
        assert resp.status_code == 503


class TestAuthEndpoints:
    def test_register_creates_user(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "new@example.com",
            "username": "newuser",
            "password": "Securepass123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "new@example.com"

    def test_register_duplicate_rejected(self, client):
        payload = {"email": "dup@example.com", "username": "dupuser", "password": "Securepass123"}
        client.post("/api/auth/register", json=payload)
        resp = client.post("/api/auth/register", json=payload)
        assert resp.status_code == 409

    def test_register_rejects_password_over_bcrypt_byte_limit(self, client):
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "long-password@example.com",
                "username": "long-password",
                "password": "密" * 25,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        "password",
        [
            "lowercase123",
            "UPPERCASE123",
            "NoNumbersHere",
            "ValidPassword1234567890123456789",
        ],
    )
    def test_register_rejects_password_that_breaks_policy(self, client, password):
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "policy@example.com",
                "username": "policy-user",
                "password": password,
            },
        )
        assert resp.status_code == 422

    def test_login_returns_token(self, auth_headers):
        assert auth_headers["Authorization"].startswith("Bearer ")

    def test_me_returns_user(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "testuser"

    def test_invalid_login_rejected(self, client):
        resp = client.post("/api/auth/login", json={
            "identifier": "nobody", "password": "wrong",
        })
        assert resp.status_code == 401

    def test_non_numeric_token_subject_is_rejected(self, client):
        token = jwt.encode(
            {"sub": "not-a-user-id"},
            settings.secret_key,
            algorithm="HS256",
        )
        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401

    def test_registration_does_not_leak_internal_database_error(
        self, client, monkeypatch
    ):
        from sqlalchemy.orm import Session

        def fail_commit(_session):
            raise RuntimeError("database-password=do-not-leak")

        monkeypatch.setattr(Session, "commit", fail_commit)
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "failure@example.com",
                "username": "failure-user",
                "password": "Securepass123",
            },
        )

        assert resp.status_code == 500
        assert "do-not-leak" not in resp.text


class TestAnalysisEndpoints:
    def test_analyze_requires_auth(self, client):
        resp = client.post("/api/v1/analyze", json={"text": "test"})
        assert resp.status_code == 401

    def test_empty_text_rejected(self, client, auth_headers):
        resp = client.post("/api/v1/analyze", json={"text": ""}, headers=auth_headers)
        assert resp.status_code == 422


class TestAdminEndpoints:
    def test_non_admin_rejected(self, client, auth_headers):
        resp = client.get("/api/admin/stats", headers=auth_headers)
        assert resp.status_code == 403
