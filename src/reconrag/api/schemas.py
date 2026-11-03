"""The API contract. Field names are a promise: extend them, don't change them."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

COLLECTION_NAME = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$"


class Page[T](BaseModel):
    data: list[T]
    limit: int
    offset: int


class CollectionIn(BaseModel):
    name: str = Field(pattern=COLLECTION_NAME)
    phi_protection: bool = Field(
        default=True, description="Pseudonymise before indexing. Set false for non-patient data."
    )


class CollectionOut(BaseModel):
    name: str
    phi_protection: bool
    document_count: int
    created_at: datetime


class JobOut(BaseModel):
    id: UUID
    filename: str
    status: str
    stage: str
    attempts: int
    size_bytes: int
    error_code: str | None = None
    error_detail: str | None = None
    version_id: UUID | None = None
    stats: dict[str, Any] = {}
    created_at: datetime
    finished_at: datetime | None = None


class RejectedUpload(BaseModel):
    filename: str
    error_code: str
    message: str


class UploadOut(BaseModel):
    jobs: list[JobOut]
    rejected: list[RejectedUpload]


class DocumentOut(BaseModel):
    id: UUID
    document_key: str
    title: str
    version_id: UUID
    version: int
    versions: int
    filename: str
    chunks: int
    quality: float
    quality_flags: list[str]
    parser: str
    language: str
    page_count: int | None
    ocr_pages: int
    near_duplicate_of: UUID | None
    created_at: datetime


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    reidentify: bool = Field(
        default=False, description="Return original values. Needs a PHI-access key."
    )


class SourceOut(BaseModel):
    ref: str
    chunk_id: UUID
    document_id: UUID
    filename: str
    section: list[str]
    page_start: int | None
    page_end: int | None
    kind: str
    text: str
    dense_similarity: float | None
    rerank_score: float | None
    injection_risk: bool
    also_in: list[str]


class AnswerOut(BaseModel):
    answer: str | None
    refused: bool
    reason: str
    confidence: float
    cited: list[str]
    sources: list[SourceOut]
    warnings: list[str] = []
    unverified_numbers: list[str] = []
    degraded: list[str]
    injection_flagged: bool
    phi_protected: bool
    reidentified: bool
    stages_ms: dict[str, float]
    model: dict[str, Any] | None = None


class ChunkOut(BaseModel):
    id: UUID
    filename: str
    version: int
    is_current: bool
    section: list[str]
    page_start: int | None
    page_end: int | None
    kind: str
    text: str
    before: list[str]
    after: list[str]
