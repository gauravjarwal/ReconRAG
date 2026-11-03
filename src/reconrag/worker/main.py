"""The worker process: claim, process, repeat, and stop cleanly on SIGTERM."""

from __future__ import annotations

import logging
import os
import random
import signal
import socket
import threading
import uuid

import psycopg
from prometheus_client import start_http_server
from psycopg_pool import PoolTimeout

from reconrag.config import Settings, get_settings
from reconrag.llm.embeddings import make_embedder
from reconrag.observability.logs import configure_logging
from reconrag.observability.metrics import QUEUE_AGE, QUEUE_DEPTH
from reconrag.security.bootstrap import make_pseudonymizer
from reconrag.storage.blobs import BlobStore
from reconrag.storage.db import Pool, create_pool
from reconrag.worker import queue
from reconrag.worker.processor import Processor
from reconrag.worker.sandbox import ParseSandbox

logger = logging.getLogger(__name__)


def run_loop(processor: Processor, pool: Pool, stop: threading.Event) -> None:
    settings = processor.settings
    while not stop.is_set():
        try:
            with pool.connection() as conn:
                job = queue.claim(
                    conn, processor.worker_id, settings.worker_lease_s, settings.worker_max_attempts
                )
                current = queue.stats(conn)
            QUEUE_DEPTH.set(current.depth)
            QUEUE_AGE.set(current.oldest_age_s)
        except (psycopg.OperationalError, PoolTimeout):
            logger.warning("database unavailable, waiting")
            stop.wait(settings.worker_poll_s * 5)
            continue
        if job is None:
            # Jitter stops idle workers from polling in step.
            stop.wait(settings.worker_poll_s * random.uniform(0.75, 1.25))
            continue
        processor.run(job)


def serve(settings: Settings) -> None:
    configure_logging(settings.log_level, as_json=settings.log_json)
    stop = threading.Event()

    def on_signal(signum: int, _frame: object) -> None:
        logger.info("stopping after the current job", extra={"signal": signum})
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if settings.worker_metrics_port:
        start_http_server(settings.worker_metrics_port)

    worker_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    sandbox = ParseSandbox(settings)
    with create_pool(settings, min_size=1) as pool:
        processor = Processor(
            pool,
            BlobStore(settings.blob_dir),
            sandbox,
            make_embedder(settings),
            settings,
            worker_id,
            make_pseudonymizer(settings),
        )
        logger.info("worker started", extra={"worker_id": worker_id})
        try:
            run_loop(processor, pool, stop)
        finally:
            sandbox.stop()
    logger.info("worker stopped")


def main() -> None:
    serve(get_settings())


if __name__ == "__main__":
    main()
