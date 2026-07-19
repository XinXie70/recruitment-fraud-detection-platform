import os
from urllib.parse import quote_plus, urlparse, urlunparse

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _build_database_url() -> str:
    """Build the DATABASE_URL, supporting three modes:

    1. DATABASE_URL env var (standard) — local dev / Docker Compose
       postgresql+psycopg2://user:pass@host:5432/dbname

    2. Cloud SQL via Unix socket — when INSTANCE_CONNECTION_NAME is set:
       postgresql+psycopg2://user:pass@/dbname?host=/cloudsql/<INSTANCE>
       This uses the Cloud SQL Auth Proxy sidecar injected by Cloud Run.

    3. Cloud SQL via cloud-sql-python-connector — when USE_CLOUD_SQL_CONNECTOR=true:
       Uses IAM-based automatic auth + TLS.  Requires the pg8000 driver.
    """
    raw = os.getenv("DATABASE_URL", "")
    if raw:
        return raw

    # Fallback: build from individual env vars
    user = os.getenv("DB_USER", "postgres")
    password = quote_plus(os.getenv("DB_PASSWORD", "postgres"))
    dbname = os.getenv("DB_NAME", "fake_job_detection")

    instance = os.getenv("INSTANCE_CONNECTION_NAME", "")
    if instance:
        # Cloud SQL via Unix socket (mounted by Cloud Run / Cloud SQL Proxy)
        host = f"/cloudsql/{instance}"
        return f"postgresql+psycopg2://{user}:{password}@/{dbname}?host={host}"

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


DATABASE_URL = _build_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
