"""SQL for collections, documents and chunks. Connections and transactions belong to the caller."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from pgvector import HalfVector
from psycopg.errors import UniqueViolation

from reconrag.domain import Chunk, Collection
from reconrag.errors import Conflict, NotFound
from reconrag.storage.db import Conn

Row = dict[str, Any]


def create_collection(conn: Conn, name: str, phi_protection: bool) -> Collection:
    try:
        row = conn.execute(
            "INSERT INTO collections (name, phi_protection) VALUES (%s, %s) RETURNING *",
            (name, phi_protection),
        ).fetchone()
    except UniqueViolation as exc:
        raise Conflict(f"Collection {name!r} already exists.") from exc
    assert row is not None
    return Collection(row["id"], row["name"], row["phi_protection"])


def get_collection(conn: Conn, name: str) -> Collection:
    row = conn.execute("SELECT * FROM collections WHERE name = %s", (name,)).fetchone()
    if row is None:
        raise NotFound(f"Collection {name!r} not found.")
    return Collection(row["id"], row["name"], row["phi_protection"])


def get_collection_by_id(conn: Conn, collection_id: UUID) -> Collection:
    row = conn.execute("SELECT * FROM collections WHERE id = %s", (collection_id,)).fetchone()
    if row is None:
        raise NotFound("Collection not found.")
    return Collection(row["id"], row["name"], row["phi_protection"])


def list_collections(conn: Conn) -> list[Row]:
    return conn.execute(
        """
        SELECT c.name, c.phi_protection, c.created_at,
               count(d.id) FILTER (WHERE d.retired_at IS NULL) AS document_count
        FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
        GROUP BY c.id ORDER BY c.name
        """
    ).fetchall()


def delete_collection(conn: Conn, collection_id: UUID) -> list[str]:
    """Erases everything, vault included, and returns the blobs nothing else references."""
    rows = conn.execute(
        """
        SELECT DISTINCT blob_key FROM ingest_jobs j
        WHERE collection_id = %(id)s AND NOT EXISTS (
            SELECT 1 FROM ingest_jobs o
            WHERE o.blob_key = j.blob_key AND o.collection_id <> %(id)s)
        """,
        {"id": collection_id},
    ).fetchall()
    conn.execute("DELETE FROM collections WHERE id = %s", (collection_id,))
    return [row["blob_key"] for row in rows]


def list_documents(conn: Conn, collection_id: UUID, limit: int, offset: int) -> list[Row]:
    return conn.execute(
        """
        SELECT d.id, d.document_key, d.title, v.id AS version_id, v.version, v.filename, v.quality,
               v.quality_flags, v.parser, v.language, v.page_count, v.ocr_pages, v.created_at,
               v.near_duplicate_of,
               (SELECT count(*) FROM document_versions x WHERE x.document_id = d.id) AS versions,
               (SELECT count(*) FROM chunks c WHERE c.version_id = v.id) AS chunks
        FROM documents d JOIN document_versions v ON v.document_id = d.id AND v.active
        WHERE d.collection_id = %s AND d.retired_at IS NULL
        ORDER BY d.document_key LIMIT %s OFFSET %s
        """,
        (collection_id, limit, offset),
    ).fetchall()


def count_documents(conn: Conn, collection_id: UUID) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM documents WHERE collection_id = %s AND retired_at IS NULL",
        (collection_id,),
    ).fetchone()
    assert row is not None
    return row["n"]


def get_chunk(conn: Conn, collection_id: UUID, chunk_id: UUID) -> Row:
    """Scoped to the collection, so a chunk id can't read across collections."""
    row = conn.execute(
        """
        SELECT c.id, c.ordinal, c.text, c.context, c.section_path, c.page_start, c.page_end,
               c.kind, c.version_id, c.active, v.filename, v.version
        FROM chunks c JOIN document_versions v ON v.id = c.version_id
        WHERE c.id = %s AND c.collection_id = %s
        """,
        (chunk_id, collection_id),
    ).fetchone()
    if row is None:
        raise NotFound("Chunk not found.")
    return row


def chunk_neighbours(conn: Conn, version_id: UUID, ordinal: int, reach: int) -> list[Row]:
    return conn.execute(
        """
        SELECT ordinal, text, page_start, page_end FROM chunks
        WHERE version_id = %s AND ordinal BETWEEN %s AND %s AND ordinal <> %s ORDER BY ordinal
        """,
        (version_id, ordinal - reach, ordinal + reach, ordinal),
    ).fetchall()


def retire_document(conn: Conn, collection_id: UUID, document_id: UUID) -> None:
    """Takes a document out of search but keeps its history."""
    found = conn.execute(
        "UPDATE documents SET retired_at = now() WHERE id = %s AND collection_id = %s RETURNING id",
        (document_id, collection_id),
    ).fetchone()
    if found is None:
        raise NotFound("Document not found.")
    conn.execute(
        "UPDATE document_versions SET active = false WHERE document_id = %s", (document_id,)
    )
    conn.execute(
        "UPDATE chunks SET active = false WHERE document_id = %s AND active", (document_id,)
    )


def find_active_by_content(conn: Conn, collection_id: UUID, content_hash: bytes) -> Row | None:
    return conn.execute(
        """
        SELECT id, filename FROM document_versions
        WHERE collection_id = %s AND content_hash = %s AND active LIMIT 1
        """,
        (collection_id, content_hash),
    ).fetchone()


def find_active_by_text(conn: Conn, collection_id: UUID, text_hash: bytes) -> Row | None:
    return conn.execute(
        """
        SELECT id, filename FROM document_versions
        WHERE collection_id = %s AND text_hash = %s AND active LIMIT 1
        """,
        (collection_id, text_hash),
    ).fetchone()


def find_near_duplicate(conn: Conn, collection_id: UUID, simhash: int, bits: int) -> Row | None:
    """A scan of the collection's active versions, which is cheap: one row per document."""
    return conn.execute(
        """
        SELECT v.id, v.document_id, d.document_key, v.edition,
               bit_count((v.simhash # %(hash)s)::bit(64)) AS distance
        FROM document_versions v JOIN documents d ON d.id = v.document_id
        WHERE v.collection_id = %(collection)s AND v.active
          AND bit_count((v.simhash # %(hash)s)::bit(64)) <= %(bits)s
        ORDER BY distance LIMIT 1
        """,
        {"collection": collection_id, "hash": simhash, "bits": bits},
    ).fetchone()


def reusable_embeddings(
    conn: Conn, collection_id: UUID, chunks: Sequence[Chunk]
) -> dict[tuple[bytes, str], np.ndarray]:
    """Unchanged chunks of a new version keep their vectors, so an edit re-embeds only the edit."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (text_hash, context) text_hash, context, embedding
        FROM chunks WHERE collection_id = %s AND text_hash = ANY(%s)
        """,
        (collection_id, [chunk.text_hash for chunk in chunks]),
    ).fetchall()
    return {(bytes(row["text_hash"]), row["context"]): row["embedding"].to_numpy() for row in rows}


@dataclass
class NewVersion:
    collection_id: UUID
    document_key: str
    title: str
    edition: int
    filename: str
    blob_key: str
    content_hash: bytes
    text_hash: bytes
    simhash: int
    near_duplicate_of: UUID | None
    parser: str
    language: str
    page_count: int | None
    ocr_pages: int
    quality: float
    quality_flags: list[str]


@dataclass(frozen=True)
class Published:
    version_id: UUID
    document_id: UUID
    version: int
    active: bool


def publish_version(
    conn: Conn, new: NewVersion, chunks: Sequence[Chunk], vectors: Sequence[np.ndarray]
) -> Published:
    """All of a document's change becomes visible at once, or none of it does.

    Runs in the caller's transaction. An advisory lock orders writers of one document.
    An older edition arriving late is stored but doesn't replace the newer one.
    """
    conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"{new.collection_id}:{new.document_key}",),
    )
    row = conn.execute(
        """
        INSERT INTO documents (collection_id, document_key, title) VALUES (%s, %s, %s)
        ON CONFLICT (collection_id, document_key)
        DO UPDATE SET title = EXCLUDED.title, retired_at = NULL
        RETURNING id
        """,
        (new.collection_id, new.document_key, new.title),
    ).fetchone()
    assert row is not None
    document_id: UUID = row["id"]
    latest = conn.execute(
        """
        SELECT coalesce(max(version), 0) AS version,
               coalesce(max(edition) FILTER (WHERE active), 0) AS active_edition
        FROM document_versions WHERE document_id = %s
        """,
        (document_id,),
    ).fetchone()
    assert latest is not None
    active = new.edition >= latest["active_edition"]
    if active:
        conn.execute(
            "UPDATE document_versions SET active = false WHERE document_id = %s AND active",
            (document_id,),
        )
        conn.execute(
            "UPDATE chunks SET active = false WHERE document_id = %s AND active", (document_id,)
        )
    version = latest["version"] + 1
    inserted = conn.execute(
        """
        INSERT INTO document_versions (
            document_id, collection_id, version, edition, filename, blob_key, content_hash,
            text_hash, simhash, near_duplicate_of, parser, language, page_count, ocr_pages,
            quality, quality_flags, active
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            document_id,
            new.collection_id,
            version,
            new.edition,
            new.filename,
            new.blob_key,
            new.content_hash,
            new.text_hash,
            new.simhash,
            new.near_duplicate_of,
            new.parser,
            new.language,
            new.page_count,
            new.ocr_pages,
            new.quality,
            new.quality_flags,
            active,
        ),
    ).fetchone()
    assert inserted is not None
    version_id: UUID = inserted["id"]
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (
                version_id, document_id, collection_id, ordinal, text, context, section_path,
                page_start, page_end, kind, token_count, text_hash, injection_risk, active,
                embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    version_id,
                    document_id,
                    new.collection_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.context,
                    chunk.section_path,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.kind,
                    chunk.token_count,
                    chunk.text_hash,
                    chunk.injection_risk,
                    active,
                    HalfVector(vector),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    return Published(version_id, document_id, version, active)
