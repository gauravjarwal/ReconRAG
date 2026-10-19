import logging
import re
import unicodedata

from chromadb import Collection
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from auth import Principal, require_api_key
from config import settings
from models.schemas import UploadResponse
from observability import audit
from pipelines.parsing import ALLOWED_EXTENSIONS, file_extension
from pipelines.retry import UpstreamServiceError
from pipelines.vector_store import is_phi_protected
from rate_limit import limiter
from routers.dependencies import existing_collection
from services.ingestion import ingest_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["documents"])

# Content-sniffed MIME types accepted per binary extension. libmagic reports
# DOCX as the OOXML type, as a generic zip, or (older versions) octet-stream.
_BINARY_MIMES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset({
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    }),
}

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\- ()]")


def _mb(n_bytes: int) -> str:
    mb = 1024 * 1024
    return f"{n_bytes // mb} MB" if n_bytes >= mb else f"{n_bytes} bytes"


def _sanitize_filename(name: str) -> str:
    """Keep only the basename, drop control/unsafe chars, no leading dots."""
    name = unicodedata.normalize("NFKC", name)
    name = re.split(r"[/\\]", name)[-1]
    name = "".join(c for c in name if unicodedata.category(c)[0] != "C")
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip().lstrip(".")
    return name[:255]


def _validate_file(file: UploadFile, data: bytes) -> str:
    """Validate extension, size, and MIME. Returns sanitized filename."""
    filename = _sanitize_filename(file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="A valid filename is required.")

    ext = file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    if len(data) > settings.max_file_size_bytes:
        raise HTTPException(status_code=413, detail=f"File '{filename}' exceeds {_mb(settings.max_file_size_bytes)} limit.")

    # MIME validation via python-magic — only for binary formats.
    # Text formats (.txt, .md) have no magic bytes; libmagic uses content heuristics
    # that misfire on code-heavy text (e.g., detecting Python examples as text/x-script.python).
    # Extension allowlist above is sufficient for text files.
    if ext in _BINARY_MIMES:
        try:
            import magic
        except ImportError as exc:
            # Fail closed: README promises content validation for binaries.
            logger.error("python-magic/libmagic unavailable; refusing binary upload")
            raise HTTPException(status_code=503, detail="Binary file validation is unavailable.") from exc
        detected_mime = magic.from_buffer(data, mime=True)
        if detected_mime not in _BINARY_MIMES[ext]:
            raise HTTPException(status_code=400, detail=f"File content type '{detected_mime}' does not match extension '{ext}'.")

    return filename


@router.post("/{collection_name}/documents", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_upload)
async def upload_documents(
    request: Request,
    files: list[UploadFile],
    principal: Principal = Depends(require_api_key),
    collection: Collection = Depends(existing_collection),
) -> UploadResponse:
    if len(files) > settings.max_files_per_request:
        raise HTTPException(status_code=400, detail=f"Too many files. Max {settings.max_files_per_request} per request.")

    total_size = 0
    validated: list[tuple[str, bytes]] = []

    for file in files:
        # Read at most limit+1 bytes so an oversized file can't exhaust memory.
        data = await file.read(settings.max_file_size_bytes + 1)
        total_size += len(data)
        if total_size > settings.max_request_size_bytes:
            raise HTTPException(status_code=413, detail=f"Total upload size exceeds {_mb(settings.max_request_size_bytes)} limit.")
        filename = _validate_file(file, data)
        validated.append((filename, data))

    try:
        # Parsing, NER and embedding are blocking; keep them off the event loop.
        result = await run_in_threadpool(ingest_files, collection, validated)
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail="Embedding service unavailable; nothing was stored.") from exc
    except Exception as exc:
        logger.error("Ingestion failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Ingestion failed; nothing was stored.") from exc

    audit(
        "documents_uploaded",
        key_id=principal.key_id,
        collection=collection.name,
        phi_protection=is_phi_protected(collection),
        files_processed=result["files_processed"],
        files_skipped=len(result["skipped"]),
        chunks_created=result["chunks_created"],
        phi_entities=sum(result["phi_entities_pseudonymized"].values()),
    )
    return UploadResponse(collection=collection.name, **result)
