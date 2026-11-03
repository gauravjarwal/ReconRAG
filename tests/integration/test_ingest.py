from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest

from reconrag.config import Settings
from reconrag.domain import Collection, Job, JobStatus
from reconrag.errors import LeaseLost, UpstreamUnavailable
from reconrag.storage import repo
from reconrag.storage.blobs import BlobStore
from reconrag.storage.db import Pool
from reconrag.worker import queue
from reconrag.worker.processor import Processor

BODY = "\n\n".join(
    f"Section {i} explains that the deductible for event type {i} is {i * 100} dollars per claim."
    for i in range(1, 60)
)


@pytest.fixture
def collection(pool: Pool) -> Collection:
    with pool.connection() as conn:
        return repo.create_collection(conn, "main", phi_protection=False)


@pytest.fixture
def processor(pool, blobs, parser, embedder, settings) -> Processor:
    return Processor(pool, blobs, parser, embedder, settings, "worker-a")


def submit(pool: Pool, blobs: BlobStore, collection: Collection, name: str, body: bytes) -> UUID:
    key, size = blobs.put([body], 10_000_000)
    with pool.connection() as conn:
        return queue.enqueue(conn, collection.id, name, key, bytes.fromhex(key), size)


def claim(pool: Pool, worker: str = "worker-a", lease: float = 30, attempts: int = 3) -> Job | None:
    with pool.connection() as conn:
        return queue.claim(conn, worker, lease, attempts)


def drain(pool: Pool, processor: Processor) -> None:
    while (job := claim(pool)) is not None:
        processor.run(job)


def job_row(pool: Pool, job_id: UUID) -> dict:
    with pool.connection() as conn:
        return queue.get_job(conn, job_id)


def scalar(pool: Pool, sql: str, *params) -> int:
    with pool.connection() as conn:
        row = conn.execute(sql, params).fetchone()
    assert row is not None
    return next(iter(row.values()))


def active_chunks(pool: Pool) -> int:
    return scalar(pool, "SELECT count(*) FROM chunks WHERE active")


def test_a_file_becomes_active_chunks(pool, blobs, collection, processor):
    job_id = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    drain(pool, processor)
    row = job_row(pool, job_id)
    assert row["status"] == "succeeded"
    assert row["stats"]["chunks"] == active_chunks(pool) > 0
    assert row["finished_at"] is not None


def test_reuploading_identical_bytes_does_no_embedding(
    pool, blobs, collection, processor, embedder
):
    submit(pool, blobs, collection, "guide.txt", BODY.encode())
    drain(pool, processor)
    embedder.embedded.clear()
    second = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    drain(pool, processor)
    assert job_row(pool, second)["status"] == "duplicate"
    assert embedder.embedded == []
    assert scalar(pool, "SELECT count(*) FROM document_versions") == 1


def test_same_text_in_another_file_is_a_duplicate(pool, blobs, collection, processor, embedder):
    submit(pool, blobs, collection, "guide.txt", BODY.encode())
    drain(pool, processor)
    embedder.embedded.clear()
    second = submit(pool, blobs, collection, "copy_of_notes.txt", b"  " + BODY.encode())
    drain(pool, processor)
    row = job_row(pool, second)
    assert row["status"] == "duplicate"
    assert row["stats"]["reason"] == "identical_text"
    assert embedder.embedded == []


def test_new_version_replaces_old_and_reembeds_only_changes(
    pool, blobs, collection, processor, embedder
):
    submit(pool, blobs, collection, "guide.txt", BODY.encode())
    drain(pool, processor)
    embedder.embedded.clear()
    edited = BODY + "\n\nA brand new closing section covers appeals and arbitration of disputes."
    job_id = submit(pool, blobs, collection, "guide_v2.txt", edited.encode())
    drain(pool, processor)

    stats = job_row(pool, job_id)["stats"]
    assert stats["version"] == 2
    assert stats["reused_embeddings"] > 0
    assert len(embedder.embedded) == stats["embedded"] < stats["chunks"]
    assert scalar(pool, "SELECT count(*) FROM documents") == 1
    assert scalar(pool, "SELECT count(*) FROM document_versions WHERE active") == 1
    assert active_chunks(pool) == stats["chunks"]


def test_older_edition_arriving_late_does_not_win(pool, blobs, collection, processor):
    submit(pool, blobs, collection, "rates_2026.txt", BODY.encode())
    drain(pool, processor)
    older = BODY.replace("deductible for event type 1 is", "fee for event type 1 was")
    job_id = submit(pool, blobs, collection, "rates_2025.txt", older.encode())
    drain(pool, processor)

    assert job_row(pool, job_id)["stats"]["superseded_on_arrival"]
    assert scalar(pool, "SELECT count(*) FROM documents") == 1
    with pool.connection() as conn:
        row = conn.execute("SELECT filename FROM document_versions WHERE active").fetchone()
    assert row is not None and row["filename"] == "rates_2026.txt"
    assert (
        scalar(pool, "SELECT count(*) FROM chunks WHERE active AND text LIKE %s", "%was 100%") == 0
    )


def test_near_duplicate_under_another_name_stays_indexed_and_linked(
    pool, blobs, collection, processor
):
    submit(pool, blobs, collection, "plan_gold.txt", BODY.encode())
    drain(pool, processor)
    similar = BODY.replace("event type 5 is 500", "event type 5 is 550")
    assert similar != BODY
    submit(pool, blobs, collection, "plan_silver.txt", similar.encode())
    drain(pool, processor)
    assert scalar(pool, "SELECT count(*) FROM documents") == 2
    assert (
        scalar(pool, "SELECT count(*) FROM document_versions WHERE near_duplicate_of IS NOT NULL")
        == 1
    )
    assert (
        scalar(
            pool, "SELECT count(*) FROM chunks WHERE active AND text LIKE %s", "%type 5 is 550 %"
        )
        == 1
    )


def test_bad_file_is_quarantined_with_a_reason(pool, blobs, collection, processor):
    job_id = submit(pool, blobs, collection, "evil.pdf", b"MZ\x90\x00not a pdf")
    drain(pool, processor)
    row = job_row(pool, job_id)
    assert row["status"] == "quarantined"
    assert row["error_code"] == "unsupported_type"
    assert row["attempts"] == 1


def test_worker_death_mid_job_is_recovered_without_duplicates(pool, blobs, collection, processor):
    job_id = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    assert claim(pool, "dead-worker", lease=0.01) is not None  # Claimed, then the process dies.
    threading.Event().wait(0.05)

    drain(pool, processor)
    row = job_row(pool, job_id)
    assert row["status"] == "succeeded"
    assert row["attempts"] == 2
    assert scalar(pool, "SELECT count(*) FROM document_versions") == 1
    assert active_chunks(pool) == row["stats"]["chunks"]


def test_zombie_worker_cannot_finish_or_publish_a_reclaimed_job(
    pool, blobs, collection, processor, parser, embedder, settings
):
    submit(pool, blobs, collection, "guide.txt", BODY.encode())
    zombie_job = claim(pool, "zombie", lease=0.01)
    assert zombie_job is not None
    threading.Event().wait(0.05)
    fresh_job = claim(pool, "worker-a")
    assert fresh_job is not None

    with pytest.raises(LeaseLost), pool.connection() as conn:
        queue.finish(conn, zombie_job, "zombie", JobStatus.SUCCEEDED)
    processor.run(fresh_job)
    assert scalar(pool, "SELECT count(*) FROM document_versions") == 1


def test_lost_lease_rolls_back_the_publish(pool, blobs, collection, parser, embedder, settings):
    job_id = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    job = claim(pool, "slow-worker")
    assert job is not None
    processor = Processor(pool, blobs, parser, embedder, settings, "slow-worker")
    with pool.connection() as conn:
        conn.execute("UPDATE ingest_jobs SET worker_id = 'other' WHERE id = %s", (job_id,))
    processor.run(job)
    assert active_chunks(pool) == 0
    assert scalar(pool, "SELECT count(*) FROM document_versions") == 0


def test_two_workers_never_claim_the_same_job(pool, blobs, collection):
    for i in range(20):
        submit(pool, blobs, collection, f"f{i}.txt", f"Body {i}".encode() * 10)

    def take(worker: int) -> list[UUID]:
        taken = []
        while (job := claim(pool, f"w{worker}")) is not None:
            taken.append(job.id)
        return taken

    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(take, range(4)))
    all_ids = [job_id for ids in results for job_id in ids]
    assert len(all_ids) == 20
    assert len(set(all_ids)) == 20


def test_provider_outage_retries_then_dead_letters(pool, blobs, collection, parser, settings):
    class Down:
        name, dim = "down", 384

        def embed_documents(self, texts):
            raise UpstreamUnavailable("503")

        def embed_query(self, text):
            raise UpstreamUnavailable("503")

    job_id = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    processor = Processor(pool, blobs, parser, Down(), settings, "worker-a")
    job = claim(pool)
    assert job is not None
    processor.run(job)
    row = job_row(pool, job_id)
    assert (row["status"], row["error_code"]) == ("queued", "transient")
    assert claim(pool) is None, "a backoff delay must apply"

    for _ in range(2):
        with pool.connection() as conn:
            conn.execute("UPDATE ingest_jobs SET not_before = now()")
        job = claim(pool)
        assert job is not None
        processor.run(job)
    assert job_row(pool, job_id)["status"] == "dead"


def test_job_that_keeps_killing_workers_is_buried(pool, blobs, collection):
    job_id = submit(pool, blobs, collection, "poison.pdf", b"%PDF-1.7")
    for _ in range(3):
        assert claim(pool, lease=0.01) is not None
        threading.Event().wait(0.03)
    assert claim(pool) is None
    row = job_row(pool, job_id)
    assert (row["status"], row["error_code"]) == ("dead", "attempts_exhausted")


def test_dead_job_can_be_requeued_by_an_operator(pool, blobs, collection, processor):
    job_id = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    with pool.connection() as conn:
        conn.execute("UPDATE ingest_jobs SET status = 'dead' WHERE id = %s", (job_id,))
        queue.retry_dead(conn, job_id)
    drain(pool, processor)
    assert job_row(pool, job_id)["status"] == "succeeded"


def test_phi_collection_needs_a_pseudonymizer(pool, blobs, parser, embedder, settings):
    with pool.connection() as conn:
        protected = repo.create_collection(conn, "phi", phi_protection=True)
    job_id = submit(pool, blobs, protected, "note.txt", b"Patient: John Smith, SSN 123-45-6789.")
    processor = Processor(pool, blobs, parser, embedder, settings, "worker-a", None)
    drain(pool, processor)
    assert job_row(pool, job_id)["error_code"] == "phi_unavailable"
    assert embedder.embedded == []


def test_blob_store_is_content_addressed(blobs: BlobStore):
    first, size = blobs.put([b"ab", b"c"], 100)
    second, _ = blobs.put([b"abc"], 100)
    assert first == second
    assert size == 3
    assert blobs.get(first) == b"abc"


def test_retiring_a_document_removes_it_from_search(pool, blobs, collection, processor):
    submit(pool, blobs, collection, "guide.txt", BODY.encode())
    drain(pool, processor)
    with pool.connection() as conn:
        document = repo.list_documents(conn, collection.id, 10, 0)[0]
        repo.retire_document(conn, collection.id, document["id"])
    assert active_chunks(pool) == 0


def test_sandbox_survives_a_crashing_parse(settings: Settings, monkeypatch):
    from reconrag.errors import RejectedFile
    from reconrag.worker.sandbox import ParseFailed, ParseSandbox

    sandbox = ParseSandbox(settings)
    try:
        assert sandbox.prepare(b"Hello world, this is text.", "a.txt", 30).kind == "txt"
        sandbox._process.kill()
        sandbox._process.join(5)
        assert sandbox.prepare(b"Back again after a crash.", "a.txt", 30).kind == "txt"
        with pytest.raises(RejectedFile):
            sandbox.prepare(b"", "a.txt", 30)
        with pytest.raises(ParseFailed) as caught:
            sandbox.prepare(b"x " * 100, "a.txt", 0)
        assert caught.value.code == "parse_timeout"
    finally:
        sandbox.stop()


def test_internal_errors_stay_out_of_the_job_record(pool, blobs, collection, parser, settings):
    class Broken:
        name, dim = "broken", 384

        def embed_documents(self, texts):
            raise RuntimeError("password=hunter2 at /srv/secret/path")

    job_id = submit(pool, blobs, collection, "guide.txt", BODY.encode())
    job = claim(pool)
    assert job is not None
    Processor(pool, blobs, parser, Broken(), settings, "worker-a").run(job)
    row = job_row(pool, job_id)
    assert row["error_code"] == "internal_error"
    assert "hunter2" not in row["error_detail"]
