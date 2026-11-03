"""Migrations are raw SQL, so there is no ORM metadata to autogenerate from."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from reconrag.config import get_settings


def _url() -> str:
    override = context.config.get_main_option("sqlalchemy.url")
    url = override or get_settings().database_url
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def run() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline migrations aren't supported.")
run()
