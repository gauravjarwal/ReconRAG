from __future__ import annotations

from typing import Annotated

from chromadb import Collection
from fastapi import HTTPException, Path

from models.schemas import COLLECTION_NAME_PATTERN
from pipelines.vector_store import get_collection

CollectionName = Annotated[str, Path(pattern=COLLECTION_NAME_PATTERN)]


def existing_collection(collection_name: CollectionName) -> Collection:
    """Resolve a path collection name to a Chroma collection, 404 if missing. Reads never create collections."""
    collection = get_collection(collection_name)
    if collection is None:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    return collection
