"""Shared types. The Pipeline protocol is how v1 and any later system plug into the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    """Where an answer can be found. Files are alternatives: any one of them is enough."""

    model_config = ConfigDict(extra="forbid")

    files: list[str]
    quote: str | None = None
    row: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> Evidence:
        if bool(self.quote) == bool(self.row and self.value):
            raise ValueError("evidence needs either a quote, or a row and a value")
        return self


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    key_facts: list[list[str]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    forbidden_files: list[str] = Field(default_factory=list)
    refuse: bool = False
    # Terms that must not appear anywhere in the corpus, proving a refusal item is unanswerable.
    absent_terms: list[str] = Field(default_factory=list)
    canary: str | None = None


class QaItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    expected: Expected
    source: str = "synthetic"
    tags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    file: str
    text: str
    score: float = 0.0


@dataclass
class FileResult:
    """What a pipeline did with one file."""

    file: str
    outcome: str  # indexed, quarantined, duplicate, superseded, or crashed
    text: str = ""
    chunks: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class Retrieval:
    chunks: list[Chunk]
    confidence: float
    refused: bool = False
    latency_ms: float = 0.0


class Pipeline(Protocol):
    """One system under test. Adding v2 means adding one more of these."""

    name: str
    config: dict[str, object]

    def ingest(self, files_dir: str, file_names: list[str]) -> list[FileResult]: ...

    def retrieve(self, question: str, k: int) -> Retrieval: ...
