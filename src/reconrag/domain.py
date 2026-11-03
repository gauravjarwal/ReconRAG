"""Types shared across stages. Plain dataclasses, with no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class BlockKind(StrEnum):
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"


@dataclass
class Block:
    """One structural unit from a parser. `rows` is set only for tables, first row is the header."""

    kind: BlockKind
    text: str = ""
    page: int | None = None
    level: int = 0
    rows: list[list[str]] | None = None


@dataclass
class ParsedDocument:
    blocks: list[Block]
    parser: str
    title: str = ""
    page_count: int | None = None
    ocr_pages: int = 0
    ocr_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    ordinal: int
    text: str
    context: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    kind: str
    token_count: int
    text_hash: bytes
    injection_risk: bool = False

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"
    DEAD = "dead"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.DUPLICATE, JobStatus.QUARANTINED, JobStatus.DEAD}
)


@dataclass(frozen=True)
class Collection:
    id: UUID
    name: str
    phi_protection: bool


@dataclass
class Job:
    id: UUID
    collection_id: UUID
    filename: str
    blob_key: str
    content_hash: bytes
    size_bytes: int
    attempts: int
    document_key: str | None = None
    status: str = JobStatus.QUEUED
    request_id: str | None = None
    entry_point: str = "unknown"


@dataclass
class Hit:
    """A retrieved chunk plus everything needed to cite it."""

    chunk_id: UUID
    version_id: UUID
    document_id: UUID
    filename: str
    ordinal: int
    text: str
    context: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    kind: str
    text_hash: bytes
    injection_risk: bool = False
    dense_similarity: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)
