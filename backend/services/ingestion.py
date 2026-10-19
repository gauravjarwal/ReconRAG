from __future__ import annotations

import hashlib
import logging
from collections import Counter

from chromadb import Collection

from pipelines.chunking import chunk_text
from pipelines.embedding import embed_texts
from pipelines.parsing import DocumentParseError, parse_file
from pipelines.retry import UpstreamServiceError
from pipelines.security import scan_chunk_for_injection
from pipelines.vector_store import is_phi_protected, replace_file_chunks
from services.deidentification import pseudonymize_document

logger = logging.getLogger(__name__)


def ingest_files(collection: Collection, files: list[tuple[str, bytes]]) -> dict:
    """
    Full ingestion pipeline: parse -> [de-identify] -> chunk -> scan -> embed -> store.

    In a PHI-protected collection the whole document is pseudonymized before
    chunking, so the embedding API and Chroma only ever see tokens.

    Args:
        collection: target ChromaDB collection
        files: list of (filename, file_bytes)

    Returns:
        dict with files_processed, chunks_created, skipped files, and PHI entity counts
    """
    phi_protection = is_phi_protected(collection)

    all_ids: list[str] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []
    processed_filenames: list[str] = []
    skipped: list[dict] = []
    phi_counts: Counter = Counter()

    for filename, data in files:
        try:
            text = parse_file(filename, data)
        except DocumentParseError as exc:
            logger.warning("Skipping file %d: %s", len(processed_filenames) + len(skipped) + 1, exc)
            skipped.append({"filename": filename, "reason": str(exc)})
            continue

        if phi_protection:
            result = pseudonymize_document(collection.name, text)
            text = result.text
            phi_counts.update(result.entity_counts)

        chunks = chunk_text(text)
        if not chunks:
            skipped.append({"filename": filename, "reason": "no extractable text"})
            continue

        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{collection.name}:{filename}:{idx}".encode()).hexdigest()[:32]
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_metadatas.append({
                "filename": filename,
                "chunk_index": idx,
                "chunk_text": chunk,
                "injection_risk": scan_chunk_for_injection(chunk),
            })

        processed_filenames.append(filename)

    if processed_filenames:
        # Embed before deleting old chunks: if the embedding API fails, the
        # previous version of the file stays searchable.
        try:
            embeddings = embed_texts(all_documents)
        except Exception as exc:
            logger.error("Embedding failed during ingestion: %s", type(exc).__name__)
            raise UpstreamServiceError("Embedding service unavailable.") from exc
        replace_file_chunks(collection, processed_filenames, all_ids, embeddings, all_documents, all_metadatas)

    logger.info(
        "Ingested %d files (%d skipped), %d chunks, %d PHI entities into '%s'",
        len(processed_filenames), len(skipped), len(all_ids), sum(phi_counts.values()), collection.name,
    )
    return {
        "files_processed": len(processed_filenames),
        "chunks_created": len(all_ids),
        "skipped": skipped,
        "phi_entities_pseudonymized": dict(phi_counts),
    }
