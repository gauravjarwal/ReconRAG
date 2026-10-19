from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from config import settings
from observability import audit

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    key_id: str            # non-reversible fingerprint, safe for logs
    can_reidentify: bool


def key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


def _matches(candidate: str, keys: frozenset[str]) -> bool:
    # Compare against every key so timing doesn't reveal which one matched.
    found = False
    for key in keys:
        found |= hmac.compare_digest(candidate.encode(), key.encode())
    return found


def require_api_key(request: Request, api_key: str | None = Depends(_api_key_header)) -> Principal:
    if api_key:
        if _matches(api_key, settings.phi_access_key_set):
            principal = Principal(key_fingerprint(api_key), can_reidentify=True)
        elif _matches(api_key, settings.api_key_set):
            principal = Principal(key_fingerprint(api_key), can_reidentify=False)
        else:
            principal = None
        if principal:
            request.state.key_id = principal.key_id
            return principal
    audit("auth_failed", key_id=key_fingerprint(api_key) if api_key else None, path=request.url.path)
    raise HTTPException(status_code=401, detail="Missing or invalid API key.", headers={"WWW-Authenticate": "APIKey"})
