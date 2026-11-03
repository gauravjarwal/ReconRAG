"""Record which request and entry point created a job, so its logs can be traced back.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE ingest_jobs ADD COLUMN request_id text")
    op.execute("ALTER TABLE ingest_jobs ADD COLUMN entry_point text NOT NULL DEFAULT 'unknown'")


def downgrade() -> None:
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN entry_point")
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN request_id")
