from __future__ import annotations

import logging
import re

from config import settings
from pipelines.phi import detect_structured_phi

logger = logging.getLogger(__name__)

# Citation pattern: [Source: filename | Chunk: N]
CITATION_PATTERN = re.compile(r"\[Source:\s*([^\|\]]+?)\s*\|\s*Chunk:\s*(\d+)\]")


def validate_citations(answer: str, sources: list[dict]) -> str:
    """
    Remove citations that refer to filenames/chunks not present in retrieved sources.
    """
    valid_refs: set[tuple[str, str]] = {
        (s["filename"], str(s["chunk_index"])) for s in sources
    }

    def _check(match: re.Match) -> str:
        filename = match.group(1).strip()
        chunk_idx = match.group(2).strip()
        if (filename, chunk_idx) in valid_refs:
            return match.group(0)
        logger.warning("Removing hallucinated citation: %s", match.group(0))
        return ""

    return CITATION_PATTERN.sub(_check, answer)


def check_pii(answer: str, source_texts: list[str]) -> str:
    """
    Redact structured PHI/PII (SSN, card, phone, MRN, dates, ...) that appears
    in the answer but NOT in the source texts. In a PHI-protected collection the
    sources are already pseudonymized, so any raw identifier in the answer was
    hallucinated or recalled from training data, and gets redacted.
    """
    source_blob = " ".join(source_texts)
    for span in reversed(detect_structured_phi(answer)):
        if span.text in source_blob:
            continue  # came from a source doc — acceptable
        logger.warning("Potential %s detected in response: [REDACTED]", span.entity_type)
        answer = answer[: span.start] + f"[{span.entity_type} REDACTED]" + answer[span.end :]
    return answer


def check_grounding(answer: str, source_texts: list[str]) -> str:
    """
    Append a warning if significant portions of the answer have no overlap with sources.
    Uses a simple word-overlap heuristic.
    """
    if not source_texts:
        return answer

    source_words: set[str] = set()
    for text in source_texts:
        source_words.update(re.findall(r"\b\w{4,}\b", text.lower()))

    answer_words = re.findall(r"\b\w{4,}\b", answer.lower())
    if not answer_words:
        return answer

    overlap = sum(1 for w in answer_words if w in source_words)
    ratio = overlap / len(answer_words)

    if ratio < 0.2:
        logger.warning("Low grounding ratio: %.2f — appending warning", ratio)
        answer += "\n\n⚠️ Some claims in this response could not be verified against the provided sources."

    return answer


def enforce_length(answer: str) -> str:
    if len(answer) > settings.max_output_length:
        answer = answer[: settings.max_output_length] + "\n\n[Response truncated]"
    return answer


def validate_output(answer: str, sources: list[dict]) -> str:
    """
    Full output validation pipeline.

    Args:
        answer: raw LLM response text
        sources: list of source dicts with keys: filename, chunk_index, chunk_text, score

    Returns:
        Validated (and possibly modified) answer string
    """
    source_texts = [s.get("chunk_text", "") for s in sources]

    answer = validate_citations(answer, sources)
    answer = check_pii(answer, source_texts)
    answer = check_grounding(answer, source_texts)
    answer = enforce_length(answer)

    return answer
