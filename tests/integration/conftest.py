"""Real Postgres, each test run on a fresh database. Start it with `make services`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest
from alembic import command
from alembic.config import Config

from reconrag.config import Settings
from reconrag.ingestion.prepare import Prepared, prepare
from reconrag.storage.blobs import BlobStore
from reconrag.storage.db import Pool, create_pool

ADMIN_URL = os.environ.get(
    "TEST_DATABASE_ADMIN_URL", "postgresql://reconrag:reconrag@localhost:5433/postgres"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    name = f"reconrag_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{name}"
    config = Config()
    config.set_main_option("script_location", "migrations")
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    yield url
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture
def settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        database_url=database_url,
        blob_dir=tmp_path / "blobs",
        ocr_enabled=False,
        worker_heartbeat_s=0.05,
        worker_lease_s=30,
        environment="test",
        reranker="none",
        redis_url="redis://localhost:6380/15",
    )


@pytest.fixture
def pool(settings: Settings) -> Iterator[Pool]:
    with create_pool(settings, min_size=1) as opened:
        with opened.connection() as conn:
            conn.execute(
                "TRUNCATE collections, documents, document_versions, chunks, ingest_jobs CASCADE"
            )
        yield opened


@pytest.fixture
def blobs(settings: Settings) -> BlobStore:
    return BlobStore(settings.blob_dir)


class FakeEmbedder:
    """Deterministic bag-of-words vectors. Counts calls, so tests can prove work was skipped."""

    name = "fake"
    dim = 384

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in text.casefold().split():
            word = word.strip(".,:;?!()")
            vector[hash_word(word) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector + 1 / np.sqrt(self.dim)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def hash_word(word: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(word))


class InlineParser:
    """Runs `prepare` in this process. The sandbox has its own test."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare(self, data: bytes, filename: str, timeout_s: float) -> Prepared:
        return prepare(data, filename, self.settings)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def parser(settings: Settings) -> InlineParser:
    return InlineParser(settings)
