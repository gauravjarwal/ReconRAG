"""The full query path, from question to passages or a refusal."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from reconrag.config import Settings
from reconrag.domain import Hit
from reconrag.errors import AllProvidersFailed, UpstreamUnavailable
from reconrag.llm.embeddings import Embedder
from reconrag.observability.metrics import CACHE, REFUSALS
from reconrag.observability.trace import Trace
from reconrag.retrieval.context import Passage, assemble, neighbours
from reconrag.retrieval.fusion import suppress_identical, weighted_rrf
from reconrag.retrieval.gate import decide
from reconrag.retrieval.rerank import CrossEncoderReranker, Reranker
from reconrag.retrieval.search import dense_search, fuzzy_search, lexical_search
from reconrag.storage.cache import Cache
from reconrag.storage.db import Pool

logger = logging.getLogger(__name__)


@dataclass
class Retrieved:
    passages: list[Passage]
    hits: list[Hit]
    refused: bool
    reason: str
    confidence: float
    degraded: list[str] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        pool: Pool,
        embedder: Embedder,
        cache: Cache | None,
        reranker: Reranker | None,
        settings: Settings,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._cache = cache
        self._reranker = reranker
        self._settings = settings

    def warm(self) -> None:
        """Loads models at startup, so the first user doesn't pay several seconds for it."""
        self._embedder.embed_query("warm up")
        if isinstance(self._reranker, CrossEncoderReranker):
            self._reranker.warm()

    def _query_vector(self, question: str) -> np.ndarray:
        if self._cache is None:
            return self._embedder.embed_query(question)
        material = f"{self._embedder.name}\x00{question}".encode()
        key = "qe:" + hashlib.sha256(material).hexdigest()
        cached = self._cache.get(key)
        if cached is not None and len(cached) == self._embedder.dim * 4:
            CACHE.labels("hit").inc()
            return np.frombuffer(cached, dtype=np.float32)
        CACHE.labels("miss").inc()
        vector = self._embedder.embed_query(question)
        self._cache.set(key, vector.tobytes(), self._settings.query_cache_ttl_s)
        return vector

    def _candidates(self, collection_id: UUID, question: str, trace: Trace) -> list[Hit]:
        settings = self._settings
        with trace.stage("embed_query"):
            vector = self._query_vector(question)
        with self._pool.connection() as conn:
            with trace.stage("dense"):
                dense = dense_search(conn, collection_id, vector, settings.dense_top_k, settings)
            with trace.stage("lexical"):
                lexical = lexical_search(conn, collection_id, question, settings.lexical_top_k)
            with trace.stage("fuzzy"):
                fuzzy = fuzzy_search(conn, collection_id, question, settings.fuzzy_top_k)
        fused = weighted_rrf(
            [
                (dense, settings.dense_weight),
                (lexical, settings.lexical_weight),
                (fuzzy, settings.fuzzy_weight),
            ],
            settings.rrf_k,
        )
        return suppress_identical(fused)

    def _rerank(
        self, question: str, candidates: list[Hit], trace: Trace, degraded: list[str]
    ) -> list[Hit]:
        shortlist = candidates[: self._settings.rerank_candidates]
        if self._reranker is None or not shortlist:
            return shortlist
        try:
            with trace.stage("rerank"):
                return self._reranker.rerank(question, shortlist)
        except (UpstreamUnavailable, AllProvidersFailed) as exc:
            logger.warning("reranker failed, using fused order", extra={"cause": str(exc)})
            degraded.append("reranker_unavailable")
            return shortlist

    def retrieve(self, collection_id: UUID, question: str, trace: Trace) -> Retrieved:
        settings = self._settings
        degraded: list[str] = []
        candidates = self._candidates(collection_id, question, trace)
        ranked = self._rerank(question, candidates, trace, degraded)
        gate = decide(question, ranked, settings)
        if gate.refuse:
            REFUSALS.labels(gate.reason).inc()
            return Retrieved([], ranked, True, gate.reason, gate.confidence, degraded)
        top = ranked[: settings.final_top_k]
        with trace.stage("context"), self._pool.connection() as conn:
            around = neighbours(conn, top, settings.expand_neighbours)
        passages = assemble(top, around, settings.context_max_tokens)
        return Retrieved(passages, top, False, gate.reason, gate.confidence, degraded)
