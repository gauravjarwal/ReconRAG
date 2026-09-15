from __future__ import annotations

import hashlib
import logging

from pipelines.chunking import chunk_text
from pipelines.embedding import embed_texts
from pipelines.parsing import parse_file
from pipelines.security import scan_chunk_for_injection
from pipelines.vector_store import add_chunks

logger = logging.getLogger(__name__)


def ingest_files(collection: str, files: list[tuple[str, bytes]]) -> dict:
    """
    Full ingestion pipeline: parse -> chunk -> scan -> embed -> store.

    Args:
        collection: target ChromaDB collection name
        files: list of (filename, file_bytes)

    Returns:
        dict with files_processed and chunks_created counts
    """
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []
    all_texts_to_embed: list[str] = []

    files_processed = 0

    for filename, data in files:
        try:
            text = parse_file(filename, data)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", filename, exc)
            continue

        chunks = chunk_text(text)
        logger.info("%s → %d chunks", filename, len(chunks))

        for idx, chunk in enumerate(chunks):
            injection_risk = scan_chunk_for_injection(chunk)
            chunk_id = hashlib.sha256(f"{collection}:{filename}:{idx}".encode()).hexdigest()[:32]

            all_ids.append(chunk_id)
            all_texts_to_embed.append(chunk)
            all_documents.append(chunk)
            all_metadatas.append({
                "filename": filename,
                "chunk_index": idx,
                "chunk_text": chunk,
                "injection_risk": injection_risk,
            })

        files_processed += 1

    if not all_texts_to_embed:
        return {"files_processed": files_processed, "chunks_created": 0}

    embeddings = embed_texts(all_texts_to_embed)
    add_chunks(collection, all_ids, embeddings, all_documents, all_metadatas)

    logger.info("Ingested %d files, %d chunks into '%s'", files_processed, len(all_ids), collection)
    return {"files_processed": files_processed, "chunks_created": len(all_ids)}
