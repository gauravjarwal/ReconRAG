"""Request dependencies: who is calling, are they within limits, which collection."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request
from fastapi.security import APIKeyHeader

from reconrag.api.schemas import COLLECTION_NAME
from reconrag.api.services import Services
from reconrag.domain import Collection
from reconrag.observability.logs import audit
from reconrag.security.auth import Principal, authenticate, fingerprint
from reconrag.storage import repo

_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
_UNSAFE_NAME = re.compile(r"[^\w.\- ()]")


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def current_principal(
    request: Request, services: ServicesDep, api_key: Annotated[str | None, Depends(_api_key)]
) -> Principal:
    principal = authenticate(api_key, services.settings)
    if principal is None:
        audit(
            "auth_failed",
            key_id=fingerprint(api_key) if api_key else None,
            path=request.url.path,
        )
        raise HTTPException(401, "Missing or invalid API key.", {"WWW-Authenticate": "APIKey"})
    return principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]


def _limit(services: Services, principal: Principal, scope: str, limit: int) -> None:
    retry_after = services.cache.over_limit(f"{principal.key_id}:{scope}", limit)
    if retry_after is not None:
        raise HTTPException(429, "Too many requests.", {"Retry-After": str(retry_after)})


def limit_default(request: Request, services: ServicesDep, principal: PrincipalDep) -> None:
    _limit(services, principal, "default", services.settings.rate_limit_default_per_min)


def limit_query(request: Request, services: ServicesDep, principal: PrincipalDep) -> None:
    _limit(services, principal, "query", services.settings.rate_limit_query_per_min)


def limit_upload(request: Request, services: ServicesDep, principal: PrincipalDep) -> None:
    _limit(services, principal, "upload", services.settings.rate_limit_upload_per_min)


def collection_dep(
    name: Annotated[str, Path(pattern=COLLECTION_NAME)], services: ServicesDep
) -> Collection:
    with services.pool.connection() as conn:
        return repo.get_collection(conn, name)


CollectionDep = Annotated[Collection, Depends(collection_dep)]


def safe_filename(name: str) -> str:
    """Display only, never a path: blobs are stored by hash."""
    name = unicodedata.normalize("NFKC", name)
    name = re.split(r"[/\\]", name)[-1]
    name = "".join(c for c in name if unicodedata.category(c)[0] != "C")
    return _UNSAFE_NAME.sub("_", name).strip().lstrip(".")[:255]
