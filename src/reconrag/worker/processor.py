"""One ingest job, from blob to published chunks."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import numpy as np
import psycopg
from psycopg_pool import PoolTimeout

from reconrag.config import Settings
from reconrag.domain import Chunk, Collection, Job, JobStatus
from reconrag.errors import LeaseLost, RejectedFile, UpstreamRejected, UpstreamUnavailable
from reconrag.ingestion.chunking import chunk_document
from reconrag.ingestion.dedup import (
    document_key,
    edition,
    simhash,
    text_digest,
    undated_key,
)
from reconrag.ingestion.prepare import Prepared
from reconrag.ingestion.quality import document_text
from reconrag.llm.embeddings import Embedder
from reconrag.observability.logs import request_id_var
from reconrag.observability.metrics import INGEST_SECONDS, JOB_RETRIES, JOBS
from reconrag.security.phi import Pseudonymizer
from reconrag.storage import repo
from reconrag.storage.blobs import BlobStore
from reconrag.storage.db import Conn, Pool
from reconrag.worker import queue
from reconrag.worker.sandbox import ParseFailed

logger = logging.getLogger(__name__)

_SAFE_DETAIL = "Processing failed. See the worker logs for this job id."
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 300.0


class Parser(Protocol):
    """ParseSandbox in production, an in-process stand-in in tests."""

    def prepare(self, data: bytes, filename: str, timeout_s: float) -> Prepared: ...


@dataclass
class Lease:
    """Keeps a job's lease alive from a background thread while work runs."""

    pool: Pool
    job: Job
    worker_id: str
    settings: Settings
    stage: str | None = None
    lost: bool = False

    def beat(self) -> None:
        with self.pool.connection() as conn:
            queue.heartbeat(
                conn, self.job, self.worker_id, self.settings.worker_lease_s, self.stage
            )

    def enter(self, stage: str) -> None:
        self.stage = stage
        if self.lost:
            raise LeaseLost("This worker no longer owns the job.")
        self.beat()


@contextmanager
def _keep_alive(lease: Lease) -> Any:
    done = threading.Event()

    def run() -> None:
        while not done.wait(lease.settings.worker_heartbeat_s):
            try:
                lease.beat()
            except LeaseLost:
                lease.lost = True
                return
            except (psycopg.Error, PoolTimeout):
                logger.warning("heartbeat failed", extra={"job_id": str(lease.job.id)})

    thread = threading.Thread(target=run, daemon=True, name="lease-heartbeat")
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(5)


@contextmanager
def _timed(stage: str) -> Any:
    started = time.perf_counter()
    try:
        yield
    finally:
        INGEST_SECONDS.labels(stage).observe(time.perf_counter() - started)


def embed_with_reuse(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    reusable: dict[tuple[bytes, str], np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Returns one vector per chunk and how many were reused instead of computed."""
    missing = [c for c in chunks if (c.text_hash, c.context) not in reusable]
    fresh = iter(embedder.embed_documents([c.embed_text for c in missing]))
    vectors = [
        reusable[(c.text_hash, c.context)] if (c.text_hash, c.context) in reusable else next(fresh)
        for c in chunks
    ]
    return vectors, len(chunks) - len(missing)


class Processor:
    def __init__(
        self,
        pool: Pool,
        blobs: BlobStore,
        parser: Parser,
        embedder: Embedder,
        settings: Settings,
        worker_id: str,
        pseudonymizer: Pseudonymizer | None = None,
    ) -> None:
        self.pool = pool
        self.blobs = blobs
        self.parser = parser
        self.embedder = embedder
        self.settings = settings
        self.worker_id = worker_id
        self.pseudonymizer = pseudonymizer

    def run(self, job: Job) -> None:
        """Logs under the request that created the job, so one id follows it across processes."""
        token = request_id_var.set(job.request_id or str(job.id))
        logger.info(
            "job started",
            extra={"job_id": str(job.id), "entry_point": job.entry_point, "attempt": job.attempts},
        )
        try:
            self._run(job)
        finally:
            request_id_var.reset(token)

    def _run(self, job: Job) -> None:
        """Sorts every failure into quarantine, retry or dead. Nothing escapes to the loop."""
        lease = Lease(self.pool, job, self.worker_id, self.settings)
        try:
            with _keep_alive(lease):
                self._process(job, lease)
        except RejectedFile as exc:
            self._finish(job, JobStatus.QUARANTINED, error_code=exc.code, error_detail=str(exc))
        except LeaseLost:
            logger.warning("lease lost, abandoning job", extra={"job_id": str(job.id)})
        except UpstreamRejected as exc:
            self._finish(job, JobStatus.DEAD, error_code="upstream_rejected", error_detail=str(exc))
        except ParseFailed as exc:
            logger.warning("parse failed: %s", exc, extra={"job_id": str(job.id)})
            self._retry(job, exc.code, _SAFE_DETAIL)
        except (UpstreamUnavailable, psycopg.OperationalError, PoolTimeout) as exc:
            logger.warning("transient failure: %s", exc, extra={"job_id": str(job.id)})
            self._retry(job, "transient", "A dependency was unavailable. The job will retry.")
        except Exception:
            # The job record is client-visible. Details stay in the (redacted) logs.
            logger.exception("job failed", extra={"job_id": str(job.id)})
            self._retry(job, "internal_error", _SAFE_DETAIL)

    def _finish(self, job: Job, status: JobStatus, **fields: Any) -> None:
        try:
            with self.pool.connection() as conn:
                queue.finish(conn, job, self.worker_id, status, **fields)
            JOBS.labels(status.value).inc()
        except LeaseLost:
            logger.warning("lease lost before finish", extra={"job_id": str(job.id)})

    def _retry(self, job: Job, code: str, detail: str) -> None:
        if job.attempts >= self.settings.worker_max_attempts:
            self._finish(job, JobStatus.DEAD, error_code=code, error_detail=detail)
            return
        ceiling = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * 2**job.attempts)
        delay = random.uniform(ceiling / 2, ceiling)
        JOB_RETRIES.labels(code).inc()
        try:
            with self.pool.connection() as conn:
                queue.retry_later(conn, job, self.worker_id, delay, code, detail)
        except LeaseLost:
            logger.warning("lease lost before retry", extra={"job_id": str(job.id)})

    def _duplicate(self, job: Job, of: dict[str, Any], reason: str) -> None:
        self._finish(
            job,
            JobStatus.DUPLICATE,
            version_id=of["id"],
            stats={"duplicate_of": of["filename"], "reason": reason},
        )

    def _pseudonymize(self, collection: Collection, prepared: Prepared) -> Counter[str]:
        if not collection.phi_protection:
            return Counter()
        if self.pseudonymizer is None:
            # Fail closed: raw PHI must never reach the embedder.
            raise RejectedFile("phi_unavailable", "PHI protection isn't available on this worker.")
        with self.pool.connection() as conn:
            prepared.document, counts = self.pseudonymizer.document(
                conn, collection.id, prepared.document
            )
        prepared.text = document_text(prepared.document)
        return counts

    def _resolve_identity(
        self, conn: Conn, job: Job, fingerprint: int
    ) -> tuple[str, int, UUID | None]:
        """Decides which document this file belongs to."""
        key = job.document_key or document_key(job.filename)
        file_edition = edition(job.filename)
        near = repo.find_near_duplicate(
            conn, job.collection_id, fingerprint, self.settings.near_duplicate_bits
        )
        if near is None or near["document_key"] == key:
            return key, file_edition, None
        same_series = undated_key(near["document_key"]) == undated_key(key)
        if not job.document_key and same_series and file_edition != near["edition"]:
            # "rates 2025" and "rates 2026" with almost the same text: editions of one document.
            return near["document_key"], file_edition, None
        return key, file_edition, near["id"]

    def _process(self, job: Job, lease: Lease) -> None:
        with self.pool.connection() as conn:
            collection = repo.get_collection_by_id(conn, job.collection_id)
            same_bytes = repo.find_active_by_content(conn, job.collection_id, job.content_hash)
        if same_bytes:
            self._duplicate(job, same_bytes, "identical_bytes")
            return

        lease.enter("parsing")
        data = self.blobs.get(job.blob_key)
        with _timed("parse"):
            prepared = self.parser.prepare(data, job.filename, self.settings.worker_parse_timeout_s)

        lease.enter("deduplicating")
        phi_counts = self._pseudonymize(collection, prepared)
        text_hash = text_digest(prepared.text)
        fingerprint = simhash(prepared.text)
        with self.pool.connection() as conn:
            same_text = repo.find_active_by_text(conn, job.collection_id, text_hash)
            if same_text:
                self._duplicate(job, same_text, "identical_text")
                return
            key, file_edition, near_id = self._resolve_identity(conn, job, fingerprint)

        lease.enter("chunking")
        # The key, not the filename: a rename must not change context and force re-embedding.
        # PHI collections get no title at all: names and filenames can hold identifiers, and the
        # title reaches the embedder and the model.
        title = "" if collection.phi_protection else prepared.document.title or key
        chunks = chunk_document(prepared.document, title, self.settings)
        if not chunks:
            raise RejectedFile("no_text", "No readable text was found in the file.")

        lease.enter("embedding")
        with self.pool.connection() as conn:
            reusable = repo.reusable_embeddings(conn, job.collection_id, chunks)
        with _timed("embed"):
            vectors, reused = embed_with_reuse(self.embedder, chunks, reusable)

        lease.enter("publishing")
        new = repo.NewVersion(
            collection_id=job.collection_id,
            document_key=key,
            title=title,
            edition=file_edition,
            filename=job.filename,
            blob_key=job.blob_key,
            content_hash=job.content_hash,
            text_hash=text_hash,
            simhash=fingerprint,
            near_duplicate_of=near_id,
            parser=prepared.document.parser,
            language=prepared.language,
            page_count=prepared.document.page_count,
            ocr_pages=prepared.document.ocr_pages,
            quality=prepared.quality,
            quality_flags=prepared.quality_flags,
        )
        with _timed("publish"), self.pool.connection() as conn:
            published = repo.publish_version(conn, new, chunks, vectors)
            stats = {
                "kind": prepared.kind,
                "chunks": len(chunks),
                "embedded": len(chunks) - reused,
                "reused_embeddings": reused,
                "version": published.version,
                "superseded_on_arrival": not published.active,
                "quality": prepared.quality,
                "quality_flags": prepared.quality_flags,
                "phi_entities": dict(phi_counts),
            }
            queue.finish(
                conn,
                job,
                self.worker_id,
                JobStatus.SUCCEEDED,
                version_id=published.version_id,
                stats=stats,
            )
        JOBS.labels(JobStatus.SUCCEEDED.value).inc()
