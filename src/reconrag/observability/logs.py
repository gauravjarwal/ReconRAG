"""Structured logs and an audit trail. Structured PHI is redacted before anything is written."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from reconrag.security.phi_detect import redact_structured_phi

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_audit = logging.getLogger("reconrag.audit")
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_structured_phi(record.getMessage()),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                entry[key] = redact_structured_phi(value) if isinstance(value, str) else value
        if record.exc_info:
            entry["exception"] = redact_structured_phi(self.formatException(record.exc_info))
        return json.dumps(entry, default=str)


class PlainFormatter(JsonFormatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = json.loads(super().format(record))
        extras = {
            k: v for k, v in entry.items() if k not in ("ts", "level", "logger", "message") and v
        }
        return f"{entry['level']:7} {entry['logger']}: {entry['message']} {extras or ''}".rstrip()


def configure_logging(level: str = "INFO", *, as_json: bool = True) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if as_json else PlainFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)


def audit(event: str, **fields: Any) -> None:
    """Who did what to which collection. Counts and ids only, never content."""
    _audit.info(event, extra={"audit": True, **fields})
