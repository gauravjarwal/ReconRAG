"""A job queue in Postgres.

`FOR UPDATE SKIP LOCKED` lets workers claim different jobs without blocking each other. A claim is
a lease: if a worker dies its lease expires and the job runs again. `attempts` is a fencing token,
so a worker that lost its lease can never finish the job, even if it wakes up later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from reconrag.domain import Job, JobStatus
from reconrag.errors import LeaseLost, NotFound
from reconrag.storage.db import Conn


def _job(row: dict[str, Any]) -> Job:
    return Job(
        id=row["id"],
        collection_id=row["collection_id"],
        filename=row["filename"],
        blob_key=row["blob_key"],
        content_hash=bytes(row["content_hash"]),
        size_bytes=row["size_bytes"],
        attempts=row["attempts"],
        document_key=row["document_key"],
        status=row["status"],
        request_id=row["request_id"],
        entry_point=row["entry_point"],
    )


def enqueue(
    conn: Conn,
    collection_id: UUID,
    filename: str,
    blob_key: str,
    content_hash: bytes,
    size_bytes: int,
    document_key: str | None = None,
    *,
    request_id: str | None = None,
    entry_point: str = "unknown",
) -> UUID:
    row = conn.execute(
        """
        INSERT INTO ingest_jobs (
            collection_id, filename, blob_key, content_hash, size_bytes, document_key,
            request_id, entry_point
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        """,
        (
            collection_id,
            filename,
            blob_key,
            content_hash,
            size_bytes,
            document_key,
            request_id,
            entry_point,
        ),
    ).fetchone()
    assert row is not None
    return row["id"]


def bury_exhausted(conn: Conn, max_attempts: int) -> int:
    """A job that keeps killing workers has an expired lease and no attempts left."""
    return conn.execute(
        """
        UPDATE ingest_jobs SET status = 'dead', stage = 'dead', finished_at = now(),
            updated_at = now(), worker_id = NULL, lease_expires_at = NULL,
            error_code = 'attempts_exhausted',
            error_detail = coalesce(error_detail, 'A worker died on every attempt.')
        WHERE status = 'running' AND lease_expires_at < now() AND attempts >= %s
        """,
        (max_attempts,),
    ).rowcount


def claim(conn: Conn, worker_id: str, lease_s: float, max_attempts: int) -> Job | None:
    """Takes the oldest runnable job, which includes jobs whose worker's lease has expired."""
    bury_exhausted(conn, max_attempts)
    row = conn.execute(
        """
        UPDATE ingest_jobs SET status = 'running', stage = 'claimed', worker_id = %(worker)s,
            attempts = attempts + 1, updated_at = now(),
            lease_expires_at = now() + make_interval(secs => %(lease)s)
        WHERE id = (
            SELECT id FROM ingest_jobs
            WHERE not_before <= now()
              AND (status = 'queued' OR (status = 'running' AND lease_expires_at < now()))
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED LIMIT 1
        )
        RETURNING *
        """,
        {"worker": worker_id, "lease": lease_s},
    ).fetchone()
    return _job(row) if row else None


def _require_ownership(rowcount: int) -> None:
    if rowcount == 0:
        raise LeaseLost("This worker no longer owns the job.")


def heartbeat(conn: Conn, job: Job, worker_id: str, lease_s: float, stage: str | None) -> None:
    result = conn.execute(
        """
        UPDATE ingest_jobs SET updated_at = now(), stage = coalesce(%s, stage),
            lease_expires_at = now() + make_interval(secs => %s)
        WHERE id = %s AND worker_id = %s AND attempts = %s AND status = 'running'
        """,
        (stage, lease_s, job.id, worker_id, job.attempts),
    )
    _require_ownership(result.rowcount)


def finish(
    conn: Conn,
    job: Job,
    worker_id: str,
    status: JobStatus,
    *,
    version_id: UUID | None = None,
    stats: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> None:
    """Fenced. Run in the publish transaction, a lost lease rolls the whole publish back."""
    result = conn.execute(
        """
        UPDATE ingest_jobs SET status = %s, stage = %s, version_id = %s, stats = %s,
            error_code = %s, error_detail = %s, finished_at = now(), updated_at = now(),
            lease_expires_at = NULL
        WHERE id = %s AND worker_id = %s AND attempts = %s AND status = 'running'
        """,
        (
            status.value,
            status.value,
            version_id,
            Jsonb(stats or {}),
            error_code,
            error_detail,
            job.id,
            worker_id,
            job.attempts,
        ),
    )
    _require_ownership(result.rowcount)


def retry_later(
    conn: Conn, job: Job, worker_id: str, delay_s: float, error_code: str, error_detail: str
) -> None:
    result = conn.execute(
        """
        UPDATE ingest_jobs SET status = 'queued', stage = 'retrying', worker_id = NULL,
            lease_expires_at = NULL, updated_at = now(), error_code = %s, error_detail = %s,
            not_before = now() + make_interval(secs => %s)
        WHERE id = %s AND worker_id = %s AND attempts = %s AND status = 'running'
        """,
        (error_code, error_detail, delay_s, job.id, worker_id, job.attempts),
    )
    _require_ownership(result.rowcount)


def get_job(conn: Conn, job_id: UUID) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM ingest_jobs WHERE id = %s", (job_id,)).fetchone()
    if row is None:
        raise NotFound("Job not found.")
    return row


def list_jobs(
    conn: Conn, collection_id: UUID, status: str | None, limit: int, offset: int
) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT * FROM ingest_jobs
        WHERE collection_id = %s AND (%s::text IS NULL OR status = %s)
        ORDER BY created_at DESC LIMIT %s OFFSET %s
        """,
        (collection_id, status, status, limit, offset),
    ).fetchall()


@dataclass(frozen=True)
class QueueStats:
    depth: int
    oldest_age_s: float


def stats(conn: Conn) -> QueueStats:
    """Age counts only jobs that could run now: one waiting out its backoff isn't stuck."""
    row = conn.execute(
        """
        SELECT count(*) AS depth, coalesce(extract(epoch FROM now() - min(created_at)
            FILTER (WHERE status = 'queued' AND not_before <= now())), 0)::float AS oldest
        FROM ingest_jobs WHERE status IN ('queued', 'running')
        """
    ).fetchone()
    assert row is not None
    return QueueStats(row["depth"], row["oldest"])


def retry_dead(conn: Conn, job_id: UUID) -> None:
    """An operator's call, after fixing what killed the job."""
    result = conn.execute(
        """
        UPDATE ingest_jobs SET status = 'queued', stage = 'queued', attempts = 0,
            not_before = now(), error_code = NULL, error_detail = NULL, finished_at = NULL,
            updated_at = now()
        WHERE id = %s AND status = 'dead'
        """,
        (job_id,),
    )
    if result.rowcount == 0:
        raise NotFound("No dead job with that id.")
