"""HTTP routes. Sync `def`, so FastAPI runs them in its threadpool. See ADR 0002."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.sse import EventSourceResponse, ServerSentEvent
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from reconrag.api import deps
from reconrag.api.deps import CollectionDep, PrincipalDep, ServicesDep, safe_filename
from reconrag.api.schemas import (
    AnswerOut,
    ChunkOut,
    CollectionIn,
    CollectionOut,
    DocumentOut,
    JobOut,
    Page,
    QueryIn,
    RejectedUpload,
    UploadOut,
)
from reconrag.errors import InvalidInput, ReconRagError
from reconrag.observability.logs import audit, request_id_var
from reconrag.observability.metrics import DB_POOL_AVAILABLE, DB_POOL_SIZE, DB_POOL_WAITING
from reconrag.storage import repo
from reconrag.worker import queue

router = APIRouter(prefix="/v1", dependencies=[Depends(deps.current_principal)])
system = APIRouter()

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]
_READ_SIZE = 1024 * 1024


@system.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    """The process is up. No dependencies, so a database blip doesn't restart every replica."""
    return {"status": "ok"}


@system.get("/readyz", include_in_schema=False)
def readyz(services: ServicesDep, response: Response) -> dict[str, Any]:
    """Postgres is required. Redis is reported, but the service works without it."""
    try:
        with services.pool.connection(timeout=1) as conn:
            conn.execute("SELECT 1 FROM alembic_version")
        database = True
    except Exception:  # Any failure here means the same thing: not ready.
        database = False
    if not database:
        response.status_code = 503
    return {
        "database": database,
        "redis": services.cache.ping(),
        "models": services.router.available,
    }


@system.get("/metrics", include_in_schema=False)
def metrics(services: ServicesDep) -> Response:
    pool = services.pool.get_stats()
    DB_POOL_SIZE.set(pool["pool_size"])
    DB_POOL_AVAILABLE.set(pool["pool_available"])
    DB_POOL_WAITING.set(pool["requests_waiting"])
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post(
    "/collections",
    response_model=CollectionOut,
    status_code=201,
    dependencies=[Depends(deps.limit_default)],
)
def create_collection(
    body: CollectionIn, services: ServicesDep, principal: PrincipalDep
) -> dict[str, Any]:
    if body.phi_protection and services.pseudonymizer is None:
        raise InvalidInput(
            "PHI protection isn't configured here. It needs PHI keys and NER, "
            "or create the collection with phi_protection false."
        )
    with services.pool.connection() as conn:
        repo.create_collection(conn, body.name, body.phi_protection)
        row = next(r for r in repo.list_collections(conn) if r["name"] == body.name)
    audit("collection_created", key_id=principal.key_id, phi_protection=body.phi_protection)
    return row


@router.get("/collections", response_model=list[CollectionOut])
def list_collections(services: ServicesDep) -> list[dict[str, Any]]:
    with services.pool.connection() as conn:
        return repo.list_collections(conn)


@router.delete("/collections/{name}", status_code=204, dependencies=[Depends(deps.limit_default)])
def delete_collection(
    collection: CollectionDep, services: ServicesDep, principal: PrincipalDep
) -> None:
    """Hard delete: chunks, versions, jobs, PHI vault entries and unshared files."""
    with services.pool.connection() as conn:
        blob_keys = repo.delete_collection(conn, collection.id)
    for key in blob_keys:
        services.blobs.delete(key)
    audit("collection_deleted", key_id=principal.key_id, collection=str(collection.id))


def _chunks_of(file: UploadFile) -> Iterator[bytes]:
    while chunk := file.file.read(_READ_SIZE):
        yield chunk


def _store(
    files: Iterable[UploadFile],
    collection_id: UUID,
    document_key: str | None,
    services: ServicesDep,
) -> tuple[list[UUID], list[RejectedUpload]]:
    job_ids: list[UUID] = []
    rejected: list[RejectedUpload] = []
    for file in files:
        name = safe_filename(file.filename or "")
        if not name:
            rejected.append(RejectedUpload(filename="", error_code="bad_name", message="No name."))
            continue
        try:
            key, size = services.blobs.put(_chunks_of(file), services.settings.max_file_bytes)
        except InvalidInput as exc:
            rejected.append(
                RejectedUpload(filename=name, error_code="file_too_large", message=str(exc))
            )
            continue
        with services.pool.connection() as conn:
            job_ids.append(
                queue.enqueue(
                    conn,
                    collection_id,
                    name,
                    key,
                    bytes.fromhex(key),
                    size,
                    document_key,
                    request_id=request_id_var.get(),
                    entry_point="api",
                )
            )
    return job_ids, rejected


@router.post(
    "/collections/{name}/documents",
    response_model=UploadOut,
    status_code=202,
    dependencies=[Depends(deps.limit_upload)],
)
def upload_documents(
    collection: CollectionDep,
    services: ServicesDep,
    principal: PrincipalDep,
    files: Annotated[list[UploadFile], File()],
    document_key: Annotated[str | None, Form(max_length=255)] = None,
) -> dict[str, Any]:
    """Queues files and returns at once. Safe to retry: a repeat upload ends as `duplicate`."""
    if len(files) > services.settings.max_files_per_request:
        limit = services.settings.max_files_per_request
        raise InvalidInput(f"Send at most {limit} files per request.")
    job_ids, rejected = _store(files, collection.id, document_key, services)
    with services.pool.connection() as conn:
        jobs = [queue.get_job(conn, job_id) for job_id in job_ids]
    audit("documents_uploaded", key_id=principal.key_id, files=len(jobs), rejected=len(rejected))
    return {"jobs": jobs, "rejected": rejected}


@router.get("/collections/{name}/documents", response_model=Page[DocumentOut])
def list_documents(
    collection: CollectionDep, services: ServicesDep, limit: Limit = 50, offset: Offset = 0
) -> dict[str, Any]:
    with services.pool.connection() as conn:
        rows = repo.list_documents(conn, collection.id, limit, offset)
    return {"data": rows, "limit": limit, "offset": offset}


@router.delete(
    "/collections/{name}/documents/{document_id}",
    status_code=204,
    dependencies=[Depends(deps.limit_default)],
)
def retire_document(
    document_id: UUID, collection: CollectionDep, services: ServicesDep, principal: PrincipalDep
) -> None:
    with services.pool.connection() as conn:
        repo.retire_document(conn, collection.id, document_id)
    audit("document_retired", key_id=principal.key_id, document=str(document_id))


@router.get("/collections/{name}/jobs", response_model=Page[JobOut])
def list_jobs(
    collection: CollectionDep,
    services: ServicesDep,
    status: str | None = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> dict[str, Any]:
    with services.pool.connection() as conn:
        rows = queue.list_jobs(conn, collection.id, status, limit, offset)
    return {"data": rows, "limit": limit, "offset": offset}


@router.get("/collections/{name}/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: UUID, collection: CollectionDep, services: ServicesDep) -> dict[str, Any]:
    with services.pool.connection() as conn:
        row = queue.get_job(conn, job_id)
    if row["collection_id"] != collection.id:
        raise HTTPException(404, "Job not found.")
    return row


@router.get("/collections/{name}/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: UUID, collection: CollectionDep, services: ServicesDep) -> dict[str, Any]:
    """Opens a citation: the passage, with what surrounds it in the document."""
    with services.pool.connection() as conn:
        row = repo.get_chunk(conn, collection.id, chunk_id)
        around = repo.chunk_neighbours(conn, row["version_id"], row["ordinal"], 1)
    return {
        **row,
        "section": row["section_path"],
        "is_current": row["active"],
        "before": [r["text"] for r in around if r["ordinal"] < row["ordinal"]],
        "after": [r["text"] for r in around if r["ordinal"] > row["ordinal"]],
    }


def _authorize_query(body: QueryIn, principal: PrincipalDep, collection: CollectionDep) -> None:
    """A dependency, so it runs before a streaming response has sent its headers."""
    if body.reidentify and not principal.can_reidentify:
        audit("reidentify_denied", key_id=principal.key_id, collection=str(collection.id))
        raise HTTPException(403, "This key can't re-identify PHI.")
    audit(
        "query",
        key_id=principal.key_id,
        collection=str(collection.id),
        reidentified=body.reidentify and collection.phi_protection,
    )


@router.post(
    "/collections/{name}/query",
    response_model=AnswerOut,
    dependencies=[Depends(deps.limit_query), Depends(_authorize_query)],
)
def query(body: QueryIn, collection: CollectionDep, services: ServicesDep) -> dict[str, Any]:
    return services.answerer.answer(collection, body.question, body.reidentify)


@router.post(
    "/collections/{name}/query/stream",
    response_class=EventSourceResponse,
    dependencies=[Depends(deps.limit_query), Depends(_authorize_query)],
)
def query_stream(
    body: QueryIn, collection: CollectionDep, services: ServicesDep
) -> Iterator[ServerSentEvent]:
    """Events: `sources`, then `token`s, then `done` (authoritative), or `error`."""
    try:
        for event in services.answerer.stream(collection, body.question, body.reidentify):
            yield ServerSentEvent(event=event.kind, data=event.data)
    except ReconRagError as exc:
        # Headers have gone by now, so a failure has to travel as an event.
        yield ServerSentEvent(event="error", data={"code": type(exc).__name__, "message": str(exc)})
