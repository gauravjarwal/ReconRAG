from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, UploadFile
from config import settings
from models.schemas import UploadResponse
from pipelines.parsing import ALLOWED_EXTENSIONS
from services.ingestion import ingest_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["documents"])

_BINARY_EXTENSIONS = frozenset({".pdf", ".docx"})


def _sanitize_filename(name: str) -> str:
    name = name.replace("../", "").replace("..\\", "").replace("/", "_").replace("\\", "_")
    return name[:255]


def _validate_file(file: UploadFile, data: bytes) -> str:
    """Validate extension, size, and MIME. Returns sanitized filename."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    filename = _sanitize_filename(file.filename)
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    if len(data) > settings.max_file_size_bytes:
        raise HTTPException(status_code=413, detail=f"File '{filename}' exceeds 50 MB limit.")

    # MIME validation via python-magic — only for binary formats.
    # Text formats (.txt, .md) have no magic bytes; libmagic uses content heuristics
    # that misfire on code-heavy text (e.g., detecting Python examples as text/x-script.python).
    # Extension allowlist above is sufficient for text files.
    if ext in _BINARY_EXTENSIONS:
        try:
            import magic
            detected_mime = magic.from_buffer(data, mime=True)
            allowed_binary_mimes = {
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/octet-stream",  # some systems report docx as this
                "application/zip",           # docx is a zip container
            }
            if detected_mime not in allowed_binary_mimes:
                raise HTTPException(status_code=400, detail=f"File content type '{detected_mime}' does not match extension '{ext}'.")
        except ImportError:
            logger.warning("python-magic not available; skipping MIME validation")

    return filename


@router.post("/{collection_name}/documents", response_model=UploadResponse)
async def upload_documents(collection_name: str, files: list[UploadFile]) -> UploadResponse:
    if len(files) > settings.max_files_per_request:
        raise HTTPException(status_code=400, detail=f"Too many files. Max {settings.max_files_per_request} per request.")

    total_size = 0
    validated: list[tuple[str, bytes]] = []

    for file in files:
        data = await file.read()
        total_size += len(data)
        if total_size > settings.max_request_size_bytes:
            raise HTTPException(status_code=413, detail="Total upload size exceeds 200 MB limit.")
        filename = _validate_file(file, data)
        validated.append((filename, data))

    result = ingest_files(collection_name, validated)
    return UploadResponse(collection=collection_name, **result)
