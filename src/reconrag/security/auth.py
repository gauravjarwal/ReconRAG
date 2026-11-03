"""API keys. Keys are compared in constant time, and only fingerprints reach the logs."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from reconrag.config import Settings


@dataclass(frozen=True)
class Principal:
    key_id: str
    can_reidentify: bool


def fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


def _matches(candidate: str, keys: frozenset[str]) -> bool:
    # Checks every key, so timing doesn't reveal how many matched or where.
    found = False
    for key in keys:
        found |= hmac.compare_digest(candidate.encode(), key.encode())
    return found


def authenticate(api_key: str | None, settings: Settings) -> Principal | None:
    if settings.disable_auth:
        return Principal("anonymous", can_reidentify=False)
    if not api_key:
        return None
    if _matches(api_key, settings.phi_key_set):
        return Principal(fingerprint(api_key), can_reidentify=True)
    if _matches(api_key, settings.api_key_set):
        return Principal(fingerprint(api_key), can_reidentify=False)
    return None
