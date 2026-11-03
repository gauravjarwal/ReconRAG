"""Local embeddings, so retrieval evals are free, offline after the first download, and repeatable.

Source for the API used here: https://github.com/qdrant/fastembed/blob/main/fastembed/text/text_embedding.py
(`passage_embed` for documents, `query_embed` for queries.)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_DIR = Path("evals/.cache")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def passages(self, texts: list[str]) -> np.ndarray: ...

    def query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model_name=model, cache_dir=str(CACHE_DIR / "models"))

    def passages(self, texts: list[str]) -> np.ndarray:
        """Unit vectors for documents. Cached by content, so repeat runs skip the embedding work."""
        digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:24]
        path = CACHE_DIR / "embeddings" / f"{self.name.replace('/', '_')}-{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = _unit(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def query(self, text: str) -> np.ndarray:
        return _unit(np.array(list(self._model.query_embed(text)), dtype=np.float32))[0]


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)
