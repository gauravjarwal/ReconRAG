"""
Logging, audit trail, and request correlation.

- Every log record passes through a PHI filter that redacts structured
  identifiers (SSN, MRN, phone, dates, ...) before it is written anywhere.
- Audit events (who did what to which collection) go to a dedicated logger as
  JSON lines, ready to ship to a SIEM. They carry counts, never content.
"""
from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pipelines.phi import redact_structured_phi

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_audit_logger = logging.getLogger("reconrag.audit")


class PhiRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_structured_phi(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact_structured_phi(logging.Formatter().formatException(record.exc_info))
        record.request_id = request_id_var.get() or "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] — %(message)s"))
    handler.addFilter(PhiRedactingFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Route uvicorn's own loggers through the same filtered handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def audit(event: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "request_id": request_id_var.get(),
        **fields,
    }
    _audit_logger.info("AUDIT %s", json.dumps(record, default=str, sort_keys=True))


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request IDs, a request-size cap, and security headers."""

    def __init__(self, app, max_body_bytes: int) -> None:
        super().__init__(app)
        self._max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if 0 < len(incoming) <= 64 and incoming.isprintable() else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > self._max_body_bytes:
                response: Response = JSONResponse(status_code=413, content={"detail": "Request body too large."})
            else:
                response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"  # responses may contain PHI
        return response
