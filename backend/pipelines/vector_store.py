from __future__ import annotations

import logging
import threading
from collections import defaultdict

import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidCollectionException, UniqueConstraintError

from config import settings

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None

RESERVED_NAMES = {"__default", "system", "admin"}

PHI_METADATA_KEY = "phi_protection"

# Bumped on every write so in-process caches (BM25) know to rebuild.
# Single-process assumption: run one uvicorn worker per Chroma volume.
_revisions: defaultdict[str, int] = defaultdict(int)
_revision_lock = threading.Lock()


class CollectionExistsError(Exception):
    pass


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        # Telemetry off: never send usage events from a system that holds PHI.
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _bump_revision(name: str) -> None:
    with _revision_lock:
        _revisions[name] += 1


def collection_revision(name: str) -> int:
    return _revisions[name]


def get_collection(name: str) -> Collection | None:
    """Return the collection, or None if it does not exist. Never creates one."""
    try:
        return _get_client().get_collection(name)
    except (InvalidCollectionException, ValueError):
        return None


def is_phi_protected(collection: Collection) -> bool:
    return bool((collection.metadata or {}).get(PHI_METADATA_KEY, False))


def create_collection(name: str, phi_protection: bool) -> Collection:
    """Create a new collection. Raises CollectionExistsError if it already exists."""
    try:
        return _get_client().create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", PHI_METADATA_KEY: phi_protection},
        )
    except UniqueConstraintError as exc:
        raise CollectionExistsError(name) from exc


def list_collections() -> list[dict]:
    """Return list of {name, document_count, phi_protection} dicts."""
    return [
        {"name": col.name, "document_count": col.count(), "phi_protection": is_phi_protected(col)}
        for col in _get_client().list_collections()
    ]


def replace_file_chunks(
    collection: Collection,
    filenames: list[str],
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """
    Delete every existing chunk of the given files, then upsert the new ones.
    Without the delete, re-uploading a shorter version of a file leaves its
    old high-index chunks behind and they keep getting retrieved.
    """
    for filename in filenames:
        collection.delete(where={"filename": filename})
    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    _bump_revision(collection.name)
    logger.info("Stored %d chunks from %d file(s) in collection '%s'", len(ids), len(filenames), collection.name)


def query_collection(
    collection: Collection,
    query_embedding: list[float],
    n_results: int,
) -> list[dict]:
    """Return list of {id, document, metadata, distance} dicts."""
    count = collection.count()
    if count == 0:
        return []
    n_results = min(n_results, count)
    results = collection.query(
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


def get_document_names(collection: Collection) -> list[str]:
    """Return sorted unique filenames stored in the collection."""
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    names: set[str] = set()
    for meta in results["metadatas"]:
        if meta and "filename" in meta:
            names.add(meta["filename"])
    return sorted(names)


def get_all_chunks(collection: Collection) -> list[dict]:
    """Fetch every chunk from the collection (for BM25 index building)."""
    if collection.count() == 0:
        return []
    results = collection.get(include=["documents", "metadatas"])
    items = []
    for i, doc_id in enumerate(results["ids"]):
        items.append({
            "id": doc_id,
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return items
