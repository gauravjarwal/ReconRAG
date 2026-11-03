"""Two API replicas and two workers, as real processes, sharing one Postgres, Redis and disk."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from reconrag.config import Settings
from reconrag.storage.db import Pool

# Fresh per run, because rate-limit counters outlive the test in Redis.
API_KEY = f"replica-{uuid.uuid4().hex}"
KEY = {"X-API-Key": API_KEY}
TERMINAL = {"succeeded", "duplicate", "quarantined", "dead"}
DOCUMENTS = 12


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def document(index: int) -> bytes:
    """Different words per document, so none is a near-duplicate of another."""
    lines = [
        f"Vessel {index}-{word} carries {index * 7 + n} units of cargo type {word}{index}."
        for n, word in enumerate(("alpha", "bravo", "cobalt", "delta", "ember", "falcon"))
    ]
    return f"Manifest for voyage V{index:03d}\n\n" + "\n\n".join(lines) + "\n"


@pytest.fixture
def stack(settings: Settings, tmp_path: Path) -> Iterator[list[httpx.Client]]:
    env = {
        **os.environ,
        "DATABASE_URL": settings.database_url,
        "REDIS_URL": "redis://localhost:6380/14",
        "BLOB_DIR": str(settings.blob_dir),
        "MODEL_CACHE_DIR": str(Path("evals/.cache/models").resolve()),
        "API_KEYS": API_KEY,
        "ENVIRONMENT": "test",
        "RERANKER": "none",
        "OCR_ENABLED": "false",
        "WORKER_POLL_S": "0.1",
        "LOG_LEVEL": "WARNING",
        "RATE_LIMIT_QUERY_PER_MIN": "6",
        "RATE_LIMIT_DEFAULT_PER_MIN": "10000",
        "RATE_LIMIT_UPLOAD_PER_MIN": "10000",
    }
    ports = [free_port(), free_port()]
    commands = [[sys.executable, "-m", "reconrag.worker.main"]] * 2
    commands += [
        [str(Path(sys.executable).parent / "reconrag"), "api", "--port", str(p)] for p in ports
    ]
    processes = [
        subprocess.Popen(
            command,
            env=env,
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for command in commands
    ]
    clients = [
        httpx.Client(base_url=f"http://127.0.0.1:{p}", headers=KEY, timeout=30) for p in ports
    ]
    try:
        deadline = time.monotonic() + 90
        for client in clients:
            while True:
                try:
                    if client.get("/readyz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, "replica never became ready"
                time.sleep(0.2)
        yield clients
    finally:
        for client in clients:
            client.close()
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)


def test_replicas_share_work_data_and_limits(stack, pool: Pool):
    first, second = stack
    assert (
        first.post("/v1/collections", json={"name": "fleet", "phi_protection": False}).status_code
        == 201
    )
    # A collection made on one replica is visible on the other.
    assert second.get("/v1/collections/fleet/documents").status_code == 200

    def upload(index: int) -> str:
        client = stack[index % 2]
        files = {"files": (f"voyage{index}.txt", document(index), "text/plain")}
        response = client.post("/v1/collections/fleet/documents", files=files)
        assert response.status_code == 202, response.text
        return response.json()["jobs"][0]["id"]

    with ThreadPoolExecutor(6) as executor:
        job_ids = list(executor.map(upload, range(DOCUMENTS)))

    deadline = time.monotonic() + 120
    while True:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT status, worker_id, attempts FROM ingest_jobs WHERE id = ANY(%s::uuid[])",
                (job_ids,),
            ).fetchall()
        if all(row["status"] in TERMINAL for row in rows):
            break
        assert time.monotonic() < deadline, f"jobs stuck: {rows}"
        time.sleep(0.2)

    assert [row["status"] for row in rows] == ["succeeded"] * DOCUMENTS
    assert all(row["attempts"] == 1 for row in rows), "a job ran twice"
    assert len({row["worker_id"] for row in rows}) == 2, "one worker did all the work"
    with pool.connection() as conn:
        counts = conn.execute(
            """
            SELECT count(*) AS total, count(DISTINCT (document_id, ordinal)) AS distinct_chunks
            FROM chunks WHERE active
            """
        ).fetchone()
    assert counts["total"] == counts["distinct_chunks"]

    # The limit is a fixed window on the wall clock: don't let the checks below straddle two.
    if time.time() % 60 > 45:
        time.sleep(61 - time.time() % 60)

    # A file stored through replica 0 answers through replica 1.
    hit = second.post(
        "/v1/collections/fleet/query", json={"question": "Which voyage carries cobalt4?"}
    )
    assert hit.status_code == 200, hit.text
    assert any(s["filename"] == "voyage4.txt" for s in hit.json()["sources"])

    # One question was spent above, so 5 remain of the limit shared across both replicas.
    codes = [
        stack[n % 2]
        .post("/v1/collections/fleet/query", json={"question": "Which voyage carries ember3?"})
        .status_code
        for n in range(8)
    ]
    assert codes.count(200) <= 5
    assert 429 in codes
