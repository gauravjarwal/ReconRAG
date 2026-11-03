from __future__ import annotations

from types import SimpleNamespace

from reconrag.api import deps
from reconrag.security.auth import Principal


def test_limits_with_the_same_number_count_separately():
    keys: list[str] = []
    cache = SimpleNamespace(over_limit=lambda key, limit: keys.append(key))
    settings = SimpleNamespace(
        rate_limit_query_per_min=30,
        rate_limit_upload_per_min=30,
        rate_limit_default_per_min=30,
    )
    services = SimpleNamespace(cache=cache, settings=settings)
    principal = Principal("key", can_reidentify=False)
    deps.limit_query(None, services, principal)  # type: ignore[arg-type]
    deps.limit_upload(None, services, principal)  # type: ignore[arg-type]
    deps.limit_default(None, services, principal)  # type: ignore[arg-type]
    assert len(set(keys)) == 3
