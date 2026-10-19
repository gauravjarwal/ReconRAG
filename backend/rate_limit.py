from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from auth import key_fingerprint
from config import settings


def _client_key(request: Request) -> str:
    """Limit per API key when one is sent (many clients can share a NAT IP), else per IP."""
    api_key = request.headers.get("x-api-key")
    return f"key:{key_fingerprint(api_key)}" if api_key else f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_client_key, default_limits=[settings.rate_limit_default])
