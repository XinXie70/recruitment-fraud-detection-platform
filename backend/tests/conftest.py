"""
Shared test fixtures — FastAPI TestClient, real database (Supabase / local PG).

Uses the configured ``DATABASE_URL`` (from .env or config.py).
Set ``TEST_DATABASE_URL`` to override for CI / isolated test runs.

Usage::

    pytest backend/tests/ -v
    TEST_DATABASE_URL=postgresql://... pytest backend/tests/ -v
"""

from __future__ import annotations

import os

import pytest

# Optionally override the database URL for tests.
_test_db_url = os.getenv("TEST_DATABASE_URL")
if _test_db_url:
    os.environ["DATABASE_URL"] = _test_db_url

# Disable rate limiting during tests.
os.environ["RATE_LIMIT_ENABLED"] = "false"

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from database import Base, engine, get_db

# Ensure all tables (including new models) exist on the test database.
Base.metadata.create_all(bind=engine)

# Add any missing columns to existing tables (non-destructive).
from sqlalchemy import inspect, text as sa_text
inspector = inspect(engine)
if "analysis_history" in inspector.get_table_names():
    existing_cols = {c["name"] for c in inspector.get_columns("analysis_history")}
    missing_cols = {
        "input_preview": "VARCHAR(500) NOT NULL DEFAULT ''",
        "input_hash": "VARCHAR(64) NOT NULL DEFAULT ''",
        "status": "VARCHAR(20) NOT NULL DEFAULT 'success'",
        "ensemble_available": "INTEGER NOT NULL DEFAULT 0",
        "ensemble_total": "INTEGER NOT NULL DEFAULT 8",
    }
    with engine.connect() as conn:
        for col_name, col_def in missing_cols.items():
            if col_name not in existing_cols:
                conn.execute(sa_text(
                    f'ALTER TABLE analysis_history ADD COLUMN {col_name} {col_def}'
                ))
                print(f"  🔧 Added missing column: analysis_history.{col_name}")
        conn.commit()


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
        "password": "testpass123",
    })
    resp = client.post("/api/auth/login", json={
        "identifier": "testuser",
        "password": "testpass123",
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
