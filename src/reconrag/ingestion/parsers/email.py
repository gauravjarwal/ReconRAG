"""Email: headers, the newest body text, and attachments parsed like any other file."""

from __future__ import annotations

import email
import email.policy
import logging
import re
from email.message import EmailMessage

from reconrag.config import Settings
from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.errors import RejectedFile
from reconrag.ingestion.parsers.html import html_blocks

logger = logging.getLogger(__name__)

_REPLY_MARKER = re.compile(r"^(On .{5,120} wrote:|-{2,}\s*Original Message\s*-{2,})\s*$", re.I)
_SIGNATURE = re.compile(r"^--\s*$")
_MAX_ATTACHMENT_DEPTH = 2


def _newest_text(body: str) -> str:
    """Quoted history is a duplicate of an earlier message, so it isn't indexed again."""
    kept: list[str] = []
    for line in body.splitlines():
        if _SIGNATURE.match(line) or _REPLY_MARKER.match(line.strip()):
            break
        if not line.lstrip().startswith(">"):
            kept.append(line)
    return "\n".join(kept).strip()


def _body_blocks(message: EmailMessage) -> list[Block]:
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return []
    content = str(part.get_content())
    if part.get_content_type() == "text/html":
        return html_blocks(content)[0]
    return [
        Block(BlockKind.TEXT, " ".join(para.split()))
        for para in re.split(r"\n\s*\n", _newest_text(content))
        if para.strip()
    ]


def _attachment_blocks(
    message: EmailMessage, settings: Settings, depth: int
) -> tuple[list[Block], list[str]]:
    from reconrag.ingestion.parsers import PARSERS
    from reconrag.ingestion.sniff import sniff

    blocks: list[Block] = []
    warnings: list[str] = []
    for part in message.iter_attachments():
        name = part.get_filename() or "attachment"
        payload = part.get_content()
        data = payload if isinstance(payload, bytes) else str(payload).encode()
        try:
            kind = sniff(data, name, settings)
            parsed = (
                _parse(data, settings, depth + 1)
                if kind == "eml"
                else PARSERS[kind](data, settings)
            )
        except RejectedFile as exc:
            warnings.append(f"attachment_skipped:{exc.code}")
            continue
        blocks.append(Block(BlockKind.HEADING, f"Attachment: {name}", level=2))
        blocks.extend(parsed.blocks)
    return blocks, warnings


def _parse(data: bytes, settings: Settings, depth: int) -> ParsedDocument:
    if depth > _MAX_ATTACHMENT_DEPTH:
        raise RejectedFile("too_deep", "Attachments are nested too deeply.")
    message = email.message_from_bytes(data, policy=email.policy.default)
    assert isinstance(message, EmailMessage)
    subject = str(message.get("Subject", "")).strip()
    fields = [
        f"{label}: {message[label]}"
        for label in ("From", "To", "Cc", "Date", "Subject")
        if message[label]
    ]
    if not fields and not message.is_multipart():
        raise RejectedFile("corrupt_file", "This doesn't look like an email.")
    blocks = [Block(BlockKind.TEXT, "\n".join(fields))] if fields else []
    blocks.extend(_body_blocks(message))
    attachments, warnings = _attachment_blocks(message, settings, depth)
    blocks.extend(attachments)
    return ParsedDocument(blocks, parser="email", title=subject, warnings=warnings)


def parse_eml(data: bytes, settings: Settings) -> ParsedDocument:
    return _parse(data, settings, 0)
