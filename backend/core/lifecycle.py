from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


def initialise_database(app_env: str, alembic_config_path: Path) -> None:
    if app_env in {"test", "production"}:
        return
    command.upgrade(Config(str(alembic_config_path)), "head")


def provision_admin(
    *,
    app_env: str,
    email: str,
    username: str,
    password: str,
    session_factory: Callable[[], Any],
    user_model: Any,
    hash_password: Callable[[str], str],
    logger: logging.Logger,
) -> None:
    if app_env == "test":
        logger.info("Skipping administrator provisioning in test environment.")
        return

    credentials = (email.strip().lower(), username.strip(), password)
    if not any(credentials):
        logger.info("Dedicated administrator provisioning is not configured.")
        return
    if not all(credentials):
        raise RuntimeError(
            "ADMIN_EMAIL, ADMIN_USERNAME and ADMIN_PASSWORD must all be configured."
        )

    normalized_email, normalized_username, normalized_password = credentials
    password_is_valid = (
        len(normalized_password) >= 8
        and len(normalized_password.encode("utf-8")) <= 72
        and any(char.isupper() for char in normalized_password)
        and any(char.islower() for char in normalized_password)
        and any(char.isdigit() for char in normalized_password)
    )
    if not password_is_valid:
        raise RuntimeError(
            "ADMIN_PASSWORD must be 8-72 bytes and contain uppercase, lowercase and digits."
        )

    db = session_factory()
    try:
        by_email = (
            db.query(user_model).filter(user_model.email == normalized_email).one_or_none()
        )
        by_username = (
            db.query(user_model)
            .filter(user_model.username == normalized_username)
            .one_or_none()
        )
        if by_email and by_username and by_email.id != by_username.id:
            raise RuntimeError("Administrator email and username belong to different users.")

        admin = by_email or by_username
        if admin is None:
            admin = user_model(
                email=normalized_email,
                username=normalized_username,
                password_hash=hash_password(normalized_password),
                is_admin=True,
            )
            db.add(admin)
        else:
            admin.email = normalized_email
            admin.username = normalized_username
            admin.is_admin = True
        db.commit()
        logger.info("Dedicated administrator account is ready: %s", normalized_username)
    except IntegrityError:
        db.rollback()
        logger.warning("Dedicated administrator provisioning hit a race; continuing.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_database(session_factory: Callable[[], Any]) -> bool:
    db = None
    try:
        db = session_factory()
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        if db is not None:
            db.close()


def warm_up_models(analysis_service: Any, logger: logging.Logger) -> None:
    try:
        outcomes = analysis_service.warm_up()
        failed = {key: error for key, error in outcomes.items() if error}
        if analysis_service.ready:
            logger.info("Ensemble runtime is ready. Failed members: %s", failed or "none")
        else:
            logger.error("No ensemble model could be loaded: %s", failed)
    except Exception:
        logger.exception("Background model warm-up failed.")
