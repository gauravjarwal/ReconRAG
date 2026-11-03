"""An operator can follow a request into its job and see a stalled queue."""

from __future__ import annotations

import json
import logging

import pytest

from reconrag.domain import Collection
from reconrag.observability.logs import JsonFormatter, request_id_var
from reconrag.storage import repo
from reconrag.storage.db import Pool
from reconrag.worker import queue
from reconrag.worker.processor import Processor


@pytest.fixture
def collection(pool: Pool) -> Collection:
    with pool.connection() as conn:
        return repo.create_collection(conn, "main", phi_protection=False)


def enqueue(pool: Pool, collection: Collection, **fields) -> None:  # type: ignore[no-untyped-def]
    with pool.connection() as conn:
        queue.enqueue(conn, collection.id, "a.txt", "ab" * 32, b"\xab" * 32, 1, **fields)


def test_a_job_carries_the_request_and_the_entry_point_that_created_it(pool, collection):
    enqueue(pool, collection, request_id="req-1", entry_point="cli")
    with pool.connection() as conn:
        job = queue.claim(conn, "w", 30, 3)
    assert job is not None
    assert (job.request_id, job.entry_point) == ("req-1", "cli")


def test_worker_logs_use_the_id_of_the_originating_request(
    pool, collection, blobs, parser, embedder, settings, caplog
):
    key, size = blobs.put([b"Some plain text long enough to be indexed as a document here."], 1000)
    with pool.connection() as conn:
        queue.enqueue(
            conn,
            collection.id,
            "a.txt",
            key,
            bytes.fromhex(key),
            size,
            request_id="req-9",
            entry_point="api",
        )
        job = queue.claim(conn, "w", 30, 3)
    assert job is not None
    processor = Processor(pool, blobs, parser, embedder, settings, "w")
    seen: list[tuple[str | None, str | None]] = []

    class Spy(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            entry = json.loads(JsonFormatter().format(record))
            seen.append((entry["request_id"], entry.get("entry_point")))

    spy = Spy()
    logger = logging.getLogger("reconrag.worker.processor")
    previous = logger.level
    logger.addHandler(spy)
    logger.setLevel(logging.INFO)
    try:
        processor.run(job)
    finally:
        logger.removeHandler(spy)
        logger.setLevel(previous)
    assert ("req-9", "api") in seen
    assert request_id_var.get() is None


def test_queue_stats_report_the_age_of_the_oldest_runnable_job(pool, collection):
    enqueue(pool, collection)
    with pool.connection() as conn:
        conn.execute("UPDATE ingest_jobs SET created_at = now() - interval '10 minutes'")
        stats = queue.stats(conn)
    assert stats.depth == 1
    assert 590 < stats.oldest_age_s < 620


def test_a_job_waiting_out_its_backoff_does_not_look_stuck(pool, collection):
    enqueue(pool, collection)
    with pool.connection() as conn:
        conn.execute(
            "UPDATE ingest_jobs SET created_at = now() - interval '10 minutes', "
            "not_before = now() + interval '1 minute'"
        )
        assert queue.stats(conn).oldest_age_s == 0


def test_pooled_connections_never_share_one_plan_across_collections(pool):
    with pool.connection() as conn:
        assert conn.execute("SHOW plan_cache_mode").fetchone() == {
            "plan_cache_mode": "force_custom_plan"
        }
