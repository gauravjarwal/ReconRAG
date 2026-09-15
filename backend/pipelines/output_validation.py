from __future__ import annotations

import logging
import re

from config import settings

logger = logging.getLogger(__name__)

# Patterns for PII / sensitive data
_CREDIT_CARD = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Citation pattern: [Source: filename | Chunk: N]
_CITATION = re.compile(r"\[Source:\s*([^\|]+?)\s*\|\s*Chunk:\s*(\d+)\]")

# Phrases that indicate the model broke character
_PERSONA_BREAK = re.compile(
    r"(?i)(as an? (ai|language model|llm|assistant)|my (training|knowledge cutoff)|"
    r"i('m| am) (not able|unable)|i (cannot|can't) (access|browse|search))"
)


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

    return _CITATION.sub(_check, answer)


def check_pii(answer: str, source_texts: list[str]) -> str:
    """
    Warn if PII appears in the answer that was NOT in the source texts.
    Emails in source docs are acceptable to surface.
    """
    source_blob = " ".join(source_texts)

    def _flag(match: re.Match, label: str) -> str:
        value = match.group(0)
        if value in source_blob:
            return value  # came from a source doc — acceptable
        logger.warning("Potential %s detected in response: [REDACTED]", label)
        return f"[{label} REDACTED]"

    answer = _CREDIT_CARD.sub(lambda m: _flag(m, "CREDIT_CARD"), answer)
    answer = _SSN.sub(lambda m: _flag(m, "SSN"), answer)
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
