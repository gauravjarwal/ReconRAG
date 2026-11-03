"""Connection pool. Every connection knows pgvector types and has a statement timeout."""

from __future__ import annotations

from typing import Any, cast

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from reconrag.config import Settings

Conn = psycopg.Connection[dict[str, Any]]
Pool = ConnectionPool[Conn]


def create_pool(settings: Settings, *, min_size: int | None = None) -> Pool:
    def configure(conn: Conn) -> None:
        register_vector(conn)
        conn.commit()

    # psycopg prepares a statement after 5 runs, and Postgres may then keep one generic plan. Search
    # needs a plan per collection: HNSW suits a large one, an exact sort suits a small one.
    options = f"-c statement_timeout={settings.db_statement_timeout_ms} "
    options += "-c plan_cache_mode=force_custom_plan"
    pool = ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min if min_size is None else min_size,
        max_size=settings.db_pool_max,
        timeout=settings.db_pool_timeout_s,
        configure=configure,
        check=ConnectionPool.check_connection,
        kwargs={
            "row_factory": dict_row,
            "options": options,
            "application_name": "reconrag",
        },
        open=False,
    )
    return cast(Pool, pool)
