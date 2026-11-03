"""The long-lived objects a process needs. Built once at startup, then passed in."""

from __future__ import annotations

from dataclasses import dataclass

from reconrag.config import Settings
from reconrag.generation.service import Answerer
from reconrag.llm.embeddings import Embedder, make_embedder
from reconrag.llm.factory import make_router
from reconrag.llm.router import LLMRouter
from reconrag.retrieval.rerank import make_reranker
from reconrag.retrieval.service import Retriever
from reconrag.security.bootstrap import make_pseudonymizer
from reconrag.security.phi import Pseudonymizer
from reconrag.storage.blobs import BlobStore
from reconrag.storage.cache import Cache
from reconrag.storage.db import Pool, create_pool


@dataclass
class Services:
    settings: Settings
    pool: Pool
    cache: Cache
    blobs: BlobStore
    embedder: Embedder
    router: LLMRouter
    answerer: Answerer
    pseudonymizer: Pseudonymizer | None
    retriever: Retriever


def build_services(
    settings: Settings,
    *,
    pool: Pool | None = None,
    embedder: Embedder | None = None,
    router: LLMRouter | None = None,
) -> Services:
    pool = pool or create_pool(settings)
    cache = Cache(settings)
    embedder = embedder or make_embedder(settings)
    router = router or make_router(settings)
    pseudonymizer = make_pseudonymizer(settings)
    retriever = Retriever(pool, embedder, cache, make_reranker(settings, router), settings)
    return Services(
        settings=settings,
        pool=pool,
        cache=cache,
        blobs=BlobStore(settings.blob_dir),
        embedder=embedder,
        router=router,
        answerer=Answerer(pool, retriever, router, settings, pseudonymizer),
        pseudonymizer=pseudonymizer,
        retriever=retriever,
    )
