"""Run the real backend against a disposable database for Playwright smoke tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import uvicorn


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repository_root))

    database_fd, database_name = tempfile.mkstemp(prefix="almond-playwright-", suffix=".sqlite")
    os.close(database_fd)
    database_path = Path(database_name)

    os.environ.update(
        {
            "APP_ENV": "test",
            "DATABASE_URL": f"sqlite:///{database_path}",
            "MODEL_SERVER_URL": "http://model-server.invalid",
            "RATE_LIMIT_ENABLED": "false",
            "LOG_FORMAT": "text",
        }
    )

    try:
        # Test-mode startup intentionally skips migrations. Create the same ORM
        # schema used by backend integration tests in this disposable database.
        from backend import models as _models  # noqa: F401
        from backend.database import Base, engine

        Base.metadata.create_all(bind=engine)
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8001, log_level="warning")
    finally:
        database_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
