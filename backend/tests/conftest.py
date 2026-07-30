"""
Shared test fixtures — FastAPI TestClient with an isolated test database.

Tests never fall back to the application ``DATABASE_URL``. Set
``TEST_DATABASE_URL`` to use a dedicated CI database; otherwise a temporary
SQLite database is created for the test session.

Usage::

    pytest backend/tests/ -v
    TEST_DATABASE_URL=postgresql://... pytest backend/tests/ -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Select the test database before importing application configuration. Never
# allow a test run to silently connect to the development or production DB.
_temporary_db_dir: tempfile.TemporaryDirectory[str] | None = None
_test_db_url = os.getenv("TEST_DATABASE_URL", "").strip()
if not _test_db_url:
    _temporary_db_dir = tempfile.TemporaryDirectory(prefix="fake-job-api-tests-")
    test_db_path = Path(_temporary_db_dir.name) / "test.db"
    _test_db_url = f"sqlite:///{test_db_path}"
os.environ["DATABASE_URL"] = _test_db_url
os.environ["APP_ENV"] = "test"

# Disable rate limiting during tests.
os.environ["RATE_LIMIT_ENABLED"] = "false"

from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from database import Base, engine, get_db

# Ensure all tables (including new models) exist on the test database.
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Per-test DB session (transactional rollback)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def db_session():
    """Yields a fresh SQLAlchemy session; creates tables then rolls back."""
    Base.metadata.create_all(bind=engine)
    connection = engine.connect()
    transaction = connection.begin()
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


# ---------------------------------------------------------------------------
# FastAPI TestClient with overridden DB dependency
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient whose ``get_db`` dependency returns the test session."""
    from main import app

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auth helper — register + get token
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def auth_headers(client):
    """Register a test user and return ``{"Authorization": "Bearer <token>"}``."""
    client.post("/api/auth/register", json={
        "email": "test@example.com",
        "username": "testuser",
        "password": "Testpass123",
    })
    resp = client.post("/api/auth/login", json={
        "identifier": "testuser",
        "password": "Testpass123",
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
