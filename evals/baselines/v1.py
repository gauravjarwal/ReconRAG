"""Runs v1's real parser, chunker, BM25 and fusion from the frozen copies.

Three differences from production v1, all recorded in each run's config:
- Embeddings come from a local model, not Gemini, so the baseline is free and repeatable.
- Vector search is exact (NumPy) instead of Chroma's HNSW. At this corpus size that is fair.
- PHI de-identification and the paid LLM reranker are off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evals.baselines.v1_frozen.bm25 import build_index, search_index
from evals.baselines.v1_frozen.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text
from evals.baselines.v1_frozen.fusion import RRF_K, reciprocal_rank_fusion
from evals.baselines.v1_frozen.parsing import ALLOWED_EXTENSIONS, DocumentParseError, parse_file
from evals.harness.embedder import Embedder
from evals.harness.schema import Chunk, FileResult, Retrieval

MODES = ("vector", "bm25", "hybrid", "default")
FETCH_K = 10
DEFAULT_MODE_K = 3


@dataclass
class V1Pipeline:
    """`hybrid` is the list v1 sends to its reranker. `default` is the no-reranker path."""

    mode: str
    embedder: Embedder
    name: str = ""
    config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}, expected one of {MODES}")
        self.name = f"v1_{self.mode}"
        self.config = {
            "pipeline": self.name,
            "embedding_model": self.embedder.name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "rrf_k": RRF_K,
            "fetch_k": FETCH_K,
            "vector_search": "exact",
            "reranker": "none",
            "phi": "off",
        }
        self._chunks: list[dict] = []
        self._vectors = np.zeros((0, 1))
        self._index = build_index([])

    def ingest(self, files_dir: str, file_names: list[str]) -> list[FileResult]:
        results = [self._ingest_one(Path(files_dir), name) for name in file_names]
        for result in results:
            if result.outcome == "indexed":
                self._chunks.extend(
                    {
                        "id": f"{result.file}:{i}",
                        "document": text,
                        "metadata": {"filename": result.file},
                    }
                    for i, text in enumerate(result.chunks)
                )
        self._vectors = self.embedder.passages([c["document"] for c in self._chunks])
        self._index = build_index(self._chunks)
        return results

    @staticmethod
    def _ingest_one(folder: Path, name: str) -> FileResult:
        extension = Path(name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return FileResult(name, "quarantined", error=f"unsupported file type {extension}")
        data = (folder / name).read_bytes()
        # v1's upload route sniffs MIME with libmagic. This is the same check for the cases here.
        if extension == ".pdf" and not data.startswith(b"%PDF-"):
            return FileResult(name, "quarantined", error="content does not match extension")
        try:
            text = parse_file(name, data)
        except DocumentParseError as exc:
            return FileResult(name, "quarantined", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a crash here is the measurement, not a failure.
            return FileResult(name, "crashed", error=f"{type(exc).__name__}: {exc}"[:200])
        chunks = chunk_text(text)
        if not chunks:
            return FileResult(name, "quarantined", text=text, error="no extractable text")
        return FileResult(name, "indexed", text=text, chunks=chunks)

    def _vector_search(self, question: str, top_k: int) -> tuple[list[dict], float]:
        scores = self._vectors @ self.embedder.query(question)
        order = np.argsort(-scores)[:top_k]
        return [self._chunks[i] for i in order], float(scores[order[0]]) if len(order) else 0.0

    def retrieve(self, question: str, k: int) -> Retrieval:
        started = time.perf_counter()
        fetch = DEFAULT_MODE_K if self.mode == "default" else FETCH_K
        fetch_vector = max(k, fetch) if self.mode == "vector" else fetch
        fetch_bm25 = max(k, fetch) if self.mode == "bm25" else fetch
        vector_hits, confidence = self._vector_search(question, fetch_vector)
        bm25_hits = search_index(self._index, question, top_k=fetch_bm25)
        if self.mode == "vector":
            ranked = vector_hits
        elif self.mode == "bm25":
            ranked = bm25_hits
        else:
            ranked = reciprocal_rank_fusion(vector_hits, bm25_hits)
        limit = DEFAULT_MODE_K if self.mode == "default" else k
        chunks = [
            Chunk(file=hit["metadata"]["filename"], text=hit["document"]) for hit in ranked[:limit]
        ]
        return Retrieval(
            chunks=chunks,
            confidence=confidence,
            refused=False,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
