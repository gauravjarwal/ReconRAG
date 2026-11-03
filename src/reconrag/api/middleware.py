"""Plain ASGI middleware, because BaseHTTPMiddleware wraps streaming in a task group."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from reconrag.observability.logs import request_id_var
from reconrag.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
)
_TOO_LARGE = b'{"error":{"code":"request_too_large","message":"The request body is too large."}}'


class _BodyTooLarge(Exception):
    pass


class RequestContext:
    """Request ids, security headers, metrics and a hard limit on body size."""

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = {"code": 500, "sent": False}
        seen = 0

        async def counting_receive() -> Any:
            # Counted as bytes arrive, because Content-Length is optional and can lie.
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > self.max_body_bytes:
                raise _BodyTooLarge
            return message

        async def tagged_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                status["code"], status["sent"] = message["status"], True
                message.setdefault("headers", [])
                message["headers"].extend(_HEADERS)
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            length = headers.get(b"content-length", b"0")
            if length.isdigit() and int(length) > self.max_body_bytes:
                raise _BodyTooLarge
            await self.app(scope, counting_receive, tagged_send)
        except _BodyTooLarge:
            if not status["sent"]:
                await tagged_send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": _TOO_LARGE})
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            elapsed = time.perf_counter() - started
            HTTP_REQUESTS.labels(route, str(status["code"])).inc()
            HTTP_LATENCY.labels(route).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "route": route,
                    "status": status["code"],
                    "ms": round(elapsed * 1000, 1),
                },
            )
            request_id_var.reset(token)
