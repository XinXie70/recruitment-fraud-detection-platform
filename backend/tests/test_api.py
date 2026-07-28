"""Tests for core API endpoints."""

from __future__ import annotations

import pytest
from jose import jwt

from config import settings


class TestHealthEndpoints:
    def test_live_does_not_depend_on_services(self, client):
        resp = client.get("/api/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "alive"}

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
        assert resp.json()["detail"]["database_connected"] is True


class TestAuthEndpoints:
    def test_register_creates_user(self, client):
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "new@example.com",
                "username": "newuser",
                "password": "Securepass123",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "new@example.com"

    def test_register_duplicate_rejected(self, client):
        payload = {
            "email": "dup@example.com",
            "username": "dupuser",
            "password": "Securepass123",
        }
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
        assert "密" not in resp.text

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
        token = auth_headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
        assert claims["sub"]
        assert claims["iat"] < claims["exp"]
        assert claims["jti"]

    def test_me_returns_user(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "testuser"

    def test_invalid_login_rejected(self, client):
        resp = client.post(
            "/api/auth/login",
            json={
                "identifier": "nobody",
                "password": "wrong",
            },
        )
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

    def test_token_for_wrong_audience_is_rejected(self, client, auth_headers):
        token = auth_headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.get_unverified_claims(token)
        claims["aud"] = "different-client"
        wrong_audience_token = jwt.encode(
            claims,
            settings.secret_key,
            algorithm="HS256",
        )

        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {wrong_audience_token}"},
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
        assert resp.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"

    def test_score_phase_requires_auth(self, client):
        resp = client.post("/api/v1/analyze/score", json={"text": "test"})
        assert resp.status_code == 401


class TestUserHistoryEndpoints:
    def test_user_can_list_and_delete_only_own_history(
        self, client, auth_headers, db_session
    ):
        from models import AnalysisHistory, User

        owner = db_session.query(User).filter(User.username == "testuser").one()
        other = User(
            email="other@example.com",
            username="other-user",
            password_hash="not-used-in-this-test",
        )
        db_session.add(other)
        db_session.flush()
        own_items = [
            AnalysisHistory(
                user_id=owner.id,
                input_preview=f"Own listing {index}",
                input_hash=str(index) * 64,
                risk_score=0.2,
                risk_level="low",
                status="success",
                ensemble_available=8,
                ensemble_total=8,
            )
            for index in (1, 2)
        ]
        other_item = AnalysisHistory(
            user_id=other.id,
            input_preview="Other user's listing",
            input_hash="3" * 64,
            risk_score=0.8,
            risk_level="high",
            status="success",
            ensemble_available=8,
            ensemble_total=8,
        )
        db_session.add_all([*own_items, other_item])
        db_session.commit()

        listing = client.get("/api/v1/history?page=1&page_size=1", headers=auth_headers)
        assert listing.status_code == 200
        assert listing.json()["total"] == 2
        assert listing.json()["total_pages"] == 2
        assert len(listing.json()["items"]) == 1
        assert "Other user's listing" not in listing.text

        forbidden_delete = client.delete(
            f"/api/v1/history/{other_item.id}", headers=auth_headers
        )
        assert forbidden_delete.status_code == 404

        own_delete = client.delete(
            f"/api/v1/history/{own_items[0].id}", headers=auth_headers
        )
        assert own_delete.status_code == 204

        remaining = client.get("/api/v1/history", headers=auth_headers)
        assert remaining.json()["total"] == 1

    def test_history_requires_authentication(self, client):
        assert client.get("/api/v1/history").status_code == 401
        assert client.delete("/api/v1/history/1").status_code == 401


class TestAdminEndpoints:
    def test_non_admin_rejected(self, client, auth_headers):
        resp = client.get("/api/admin/stats", headers=auth_headers)
        assert resp.status_code == 403

    def test_admin_dashboard_queries(self, client, auth_headers, db_session):
        from models import AnalysisHistory, User

        user = db_session.query(User).filter(User.username == "testuser").one()
        user.is_admin = True
        db_session.add_all(
            [
                AnalysisHistory(
                    user_id=user.id,
                    input_preview="First listing",
                    input_hash="a" * 64,
                    risk_score=0.9,
                    risk_level="high",
                    status="success",
                    ensemble_available=8,
                    ensemble_total=8,
                ),
                AnalysisHistory(
                    user_id=user.id,
                    input_preview="Second listing",
                    input_hash="b" * 64,
                    risk_score=0.2,
                    risk_level="low",
                    status="success",
                    ensemble_available=7,
                    ensemble_total=8,
                ),
            ]
        )
        db_session.commit()

        stats = client.get("/api/admin/stats", headers=auth_headers)
        assert stats.status_code == 200
        assert stats.json()["total_analyses"] == 2
        assert stats.json()["high_risk_count"] == 1
        assert stats.json()["low_risk_count"] == 1

        users = client.get("/api/admin/users", headers=auth_headers)
        assert users.status_code == 200
        assert users.json()["items"][0]["analysis_count"] == 2

        analyses = client.get(
            f"/api/admin/analyses?user_id={user.id}",
            headers=auth_headers,
        )
        assert analyses.status_code == 200
        assert analyses.json()["total"] == 2
