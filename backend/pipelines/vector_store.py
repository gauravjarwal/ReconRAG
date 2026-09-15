from __future__ import annotations

import logging

import chromadb
from chromadb import Collection

from config import settings

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None

RESERVED_NAMES = {"__default", "system", "admin"}


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_or_create_collection(name: str) -> Collection:
    return _get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def create_collection(name: str) -> Collection:
    """Create a new collection. Raises if it already exists."""
    return _get_client().create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def list_collections() -> list[dict]:
    """Return list of {name, document_count} dicts."""
    client = _get_client()
    results = []
    for col in client.list_collections():
        coll = client.get_collection(col.name)
        results.append({"name": col.name, "document_count": coll.count()})
    return results


def add_chunks(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    col = get_or_create_collection(collection_name)
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("Added %d chunks to collection '%s'", len(ids), collection_name)


def query_collection(
    collection_name: str,
    query_embedding: list[float],
    n_results: int,
) -> list[dict]:
    """Return list of {id, document, metadata, distance} dicts."""
    col = get_or_create_collection(collection_name)
    count = col.count()
    if count == 0:
        return []
    n_results = min(n_results, count)
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for i, doc_id in enumerate(results["ids"][0]):
        items.append({
            "id": doc_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return items


def get_document_names(collection_name: str) -> list[str]:
    """Return sorted unique filenames stored in the collection."""
    col = get_or_create_collection(collection_name)
    count = col.count()
    if count == 0:
        return []
    results = col.get(include=["metadatas"])
    names: set[str] = set()
    for meta in results["metadatas"]:
        if meta and "filename" in meta:
            names.add(meta["filename"])
    return sorted(names)


def get_all_chunks(collection_name: str) -> list[dict]:
    """Fetch every chunk from the collection (for BM25 index building)."""
    col = get_or_create_collection(collection_name)
    count = col.count()
    if count == 0:
        return []
    results = col.get(include=["documents", "metadatas"])
    items = []
    for i, doc_id in enumerate(results["ids"]):
        items.append({
            "id": doc_id,
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return items
