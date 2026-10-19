from __future__ import annotations

import logging

from chromadb import Collection
from fastapi import APIRouter, Depends, HTTPException

from auth import Principal, require_api_key
from models.schemas import CollectionInfo, CreateCollectionRequest
from observability import audit
from pipelines.vector_store import (
    RESERVED_NAMES,
    CollectionExistsError,
    create_collection,
    get_document_names,
    is_phi_protected,
    list_collections,
)
from routers.dependencies import existing_collection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.get("", response_model=list[CollectionInfo])
def get_collections(principal: Principal = Depends(require_api_key)) -> list[CollectionInfo]:
    return [CollectionInfo(**c) for c in list_collections()]


@router.post("", response_model=CollectionInfo, status_code=201)
def post_collection(body: CreateCollectionRequest, principal: Principal = Depends(require_api_key)) -> CollectionInfo:
    if body.name in RESERVED_NAMES:
        raise HTTPException(status_code=400, detail=f"Collection name '{body.name}' is reserved.")
    try:
        create_collection(body.name, phi_protection=body.phi_protection)
    except CollectionExistsError as exc:
        raise HTTPException(status_code=409, detail=f"Collection '{body.name}' already exists.") from exc
    audit("collection_created", key_id=principal.key_id, collection=body.name, phi_protection=body.phi_protection)
    return CollectionInfo(name=body.name, document_count=0, phi_protection=body.phi_protection)


@router.get("/{collection_name}/documents", response_model=list[str])
def get_collection_documents(
    principal: Principal = Depends(require_api_key),
    collection: Collection = Depends(existing_collection),
) -> list[str]:
    """Return sorted unique filenames in a collection."""
    audit("documents_listed", key_id=principal.key_id, collection=collection.name, phi_protection=is_phi_protected(collection))
    return get_document_names(collection)
