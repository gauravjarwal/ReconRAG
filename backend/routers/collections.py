from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import CollectionInfo, CreateCollectionRequest
from pipelines.vector_store import RESERVED_NAMES, create_collection, get_document_names, list_collections

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.get("", response_model=list[CollectionInfo])
def get_collections() -> list[CollectionInfo]:
    return [CollectionInfo(**c) for c in list_collections()]


@router.post("", response_model=CollectionInfo, status_code=201)
def post_collection(body: CreateCollectionRequest) -> CollectionInfo:
    if body.name in RESERVED_NAMES:
        raise HTTPException(status_code=400, detail=f"Collection name '{body.name}' is reserved.")
    try:
        create_collection(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"Collection '{body.name}' already exists.") from exc
    except Exception as exc:
        logger.error("Failed to create collection '%s': %s", body.name, exc)
        raise HTTPException(status_code=500, detail="Failed to create collection.") from exc
    return CollectionInfo(name=body.name, document_count=0)


@router.get("/{collection_name}/documents", response_model=list[str])
def get_collection_documents(collection_name: str) -> list[str]:
    """Return sorted unique filenames in a collection."""
    return get_document_names(collection_name)
