from __future__ import annotations

import json
import logging
import sys

from backend.config import settings


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attribute in (
            "request_id",
            "path",
            "method",
            "body_len",
            "status_code",
            "duration_ms",
        ):
            if hasattr(record, attribute):
                payload[attribute] = getattr(record, attribute)
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.handlers.clear()
    root.addHandler(handler)
