"""Builds the FastAPI app."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout
from starlette.exceptions import HTTPException

from reconrag import __version__
from reconrag.api.middleware import RequestContext
from reconrag.api.routes import router, system
from reconrag.api.services import Services, build_services
from reconrag.config import Settings, get_settings
from reconrag.errors import (
    AllProvidersFailed,
    Conflict,
    InvalidInput,
    NotFound,
    ReconRagError,
    UpstreamRejected,
    UpstreamUnavailable,
)
from reconrag.observability.logs import configure_logging, request_id_var

logger = logging.getLogger(__name__)

_STATUS: dict[type[ReconRagError], tuple[int, str]] = {
    NotFound: (404, "not_found"),
    Conflict: (409, "conflict"),
    InvalidInput: (400, "invalid_input"),
    UpstreamRejected: (502, "upstream_rejected"),
    UpstreamUnavailable: (503, "upstream_unavailable"),
    AllProvidersFailed: (503, "no_provider_available"),
}
_HTTP_CODES = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
}


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    headers = extra.pop("headers", None)
    body = {
        "error": {"code": code, "message": message, "request_id": request_id_var.get(), **extra}
    }
    return JSONResponse(body, status_code=status, headers=headers)


def _register_errors(app: FastAPI) -> None:
    @app.exception_handler(ReconRagError)
    def known(_: Request, exc: ReconRagError) -> JSONResponse:
        status, code = _STATUS.get(type(exc), (400, "invalid_input"))
        return _error(status, code, str(exc))

    @app.exception_handler(HTTPException)
    def http(_: Request, exc: HTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "error")
        return _error(exc.status_code, code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations only. The raw input may hold PHI, so it isn't echoed back.
        fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        return _error(422, "validation_error", "The request is invalid.", fields=fields)

    @app.exception_handler(PoolTimeout)
    @app.exception_handler(psycopg.OperationalError)
    def overloaded(_: Request, exc: Exception) -> JSONResponse:
        logger.warning("database unavailable: %s", type(exc).__name__)
        return _error(
            503, "overloaded", "The service is busy. Try again.", headers={"Retry-After": "1"}
        )

    @app.exception_handler(Exception)
    def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return _error(500, "internal_error", "Something went wrong on our side.")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, as_json=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = services is None
        app.state.services = services or build_services(settings)
        if owned:
            app.state.services.pool.open(wait=True, timeout=30)
            await run_in_threadpool(app.state.services.retriever.warm)
        yield
        if owned:
            app.state.services.pool.close()
            app.state.services.cache.close()

    app = FastAPI(title="ReconRAG", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
    app.add_middleware(RequestContext, max_body_bytes=settings.max_request_bytes)
    _register_errors(app)
    app.include_router(router)
    app.include_router(system)
    return app


def app_factory() -> FastAPI:
    return create_app()
