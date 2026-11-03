"""Embedders. Documents and queries are separate calls because the models treat them differently."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from reconrag.config import Settings

Vector = NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class LocalEmbedder:
    """Runs on CPU with no key. Loads lazily, so processes that don't embed stay light."""

    def __init__(self, settings: Settings) -> None:
        self.name = settings.embedding_model
        self.dim = settings.embedding_dim
        self._cache = settings.model_cache_dir
        self._batch = settings.embedding_batch_size
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]  # fastembed ships no usable stubs.
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.name, cache_dir=str(self._cache))
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        vectors = self._load().passage_embed(list(texts), batch_size=self._batch)
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def embed_query(self, text: str) -> Vector:
        return np.asarray(next(iter(self._load().query_embed([text]))), dtype=np.float32)


class GeminiEmbedder:
    """Hosted option. `dim` truncates the vector, which the model supports."""

    _BATCH = 100

    def __init__(self, settings: Settings) -> None:
        from reconrag.llm.gemini import make_client

        self.name = "gemini-embedding-001"
        self.dim = settings.embedding_dim
        self._client = make_client(settings)

    def _embed(self, texts: Sequence[str], task: str) -> list[Vector]:
        from google.genai import errors, types

        from reconrag.llm.gemini import translate

        vectors: list[Vector] = []
        for start in range(0, len(texts), self._BATCH):
            try:
                batch: list[Any] = list(texts[start : start + self._BATCH])
                response = self._client.models.embed_content(
                    model=self.name,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
                )
            except errors.APIError as exc:
                raise translate(exc) from exc
            vectors.extend(
                np.asarray(item.values, dtype=np.float32) for item in response.embeddings or []
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "gemini":
        return GeminiEmbedder(settings)
    return LocalEmbedder(settings)
