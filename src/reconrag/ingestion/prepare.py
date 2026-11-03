"""The CPU-heavy, side-effect-free part of ingestion. Runs in a child process.

Bytes go in, a cleaned document comes out. Nothing here touches the database, so a hostile file
that hangs or kills the process costs one retry and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconrag.config import Settings
from reconrag.domain import ParsedDocument
from reconrag.errors import RejectedFile
from reconrag.ingestion.clean import clean_document
from reconrag.ingestion.parsers import PARSERS
from reconrag.ingestion.quality import document_text, language, score
from reconrag.ingestion.sniff import sniff


@dataclass
class Prepared:
    document: ParsedDocument
    kind: str
    text: str
    language: str
    quality: float
    quality_flags: list[str]


def prepare(data: bytes, filename: str, settings: Settings) -> Prepared:
    kind = sniff(data, filename, settings)
    document = clean_document(PARSERS[kind](data, settings))
    text = document_text(document)
    if not text.strip():
        raise RejectedFile("no_text", "No readable text was found in the file.")
    quality, flags = score(document)
    if quality < settings.quality_reject_below:
        raise RejectedFile("unreadable_text", "The extracted text is too garbled to use.")
    if quality < settings.quality_flag_below:
        flags.append("low_quality")
    return Prepared(document, kind, text, language(text), quality, flags)
