"""Runs real worker processes, then kills them the way production does."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest

from reconrag.config import Settings
from reconrag.storage import repo
from reconrag.storage.blobs import BlobStore
from reconrag.storage.db import Pool
from reconrag.worker import queue

LONG_DOCUMENT = "\n\n".join(
    f"Article {i}. Obligation {i} requires inspection of unit {i * 3} every {i % 90 + 1} days."
    for i in range(1, 6000)
).encode()


def wait_for(predicate, timeout: float = 90.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for the condition")


@pytest.fixture
def start_worker(settings: Settings, tmp_path: Path) -> Iterator:
    started: list[subprocess.Popen[bytes]] = []

    def start() -> subprocess.Popen[bytes]:
        env = {
            **os.environ,
            "DATABASE_URL": settings.database_url,
            "BLOB_DIR": str(settings.blob_dir),
            "MODEL_CACHE_DIR": str(Path("evals/.cache/models").resolve()),
            "WORKER_LEASE_S": "2",
            "WORKER_HEARTBEAT_S": "0.5",
            "WORKER_POLL_S": "0.1",
            "OCR_ENABLED": "false",
            "LOG_LEVEL": "WARNING",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "reconrag.worker.main"],
            env=env,
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(process)
        return process

    yield start
    for process in started:
        if process.poll() is None:
            process.kill()
            process.wait(10)


def submit(pool: Pool, blobs: BlobStore, name: str) -> UUID:
    with pool.connection() as conn:
        collection = repo.create_collection(conn, "drill", phi_protection=False)
    key, size = blobs.put([LONG_DOCUMENT], 10_000_000)
    with pool.connection() as conn:
        return queue.enqueue(conn, collection.id, name, key, bytes.fromhex(key), size)


def job(pool: Pool, job_id: UUID) -> dict:
    with pool.connection() as conn:
        return queue.get_job(conn, job_id)


def chunk_counts(pool: Pool) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT count(*) AS chunks, count(DISTINCT ordinal) AS ordinals,
                   (SELECT count(*) FROM document_versions) AS versions
            FROM chunks WHERE active
            """
        ).fetchone()
    assert row is not None
    return row


def test_sigkill_during_embedding_loses_nothing_and_duplicates_nothing(pool, blobs, start_worker):
    job_id = submit(pool, blobs, "contract.txt")
    first = start_worker()
    wait_for(lambda: job(pool, job_id)["stage"] == "embedding")

    first.send_signal(signal.SIGKILL)
    first.wait(10)
    assert job(pool, job_id)["status"] == "running", "a dead worker leaves the job claimed"
    assert chunk_counts(pool)["chunks"] == 0, "nothing was published by the dead worker"

    start_worker()
    wait_for(lambda: job(pool, job_id)["status"] == "succeeded")
    row = job(pool, job_id)
    counts = chunk_counts(pool)
    assert row["attempts"] == 2
    assert counts["versions"] == 1
    assert counts["chunks"] == counts["ordinals"] == row["stats"]["chunks"]


def test_sigterm_finishes_the_current_job_then_exits(pool, blobs, start_worker):
    job_id = submit(pool, blobs, "contract.txt")
    worker = start_worker()
    wait_for(lambda: job(pool, job_id)["stage"] == "embedding")

    worker.send_signal(signal.SIGTERM)
    assert worker.wait(120) == 0
    assert job(pool, job_id)["status"] == "succeeded"
