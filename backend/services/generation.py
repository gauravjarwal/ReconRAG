from __future__ import annotations

import logging
import os

import anthropic
import google.generativeai as genai
import httpx

from config import settings
from models.schemas import SourceInfo
from pipelines.output_validation import CITATION_PATTERN
from pipelines.prompts import GENERATION_SYSTEM_PROMPT_V2
from pipelines.retry import with_retries
from pipelines.security import escape_prompt_tags, wrap_risky_chunk
from pipelines.usage import ActionUsage, cost_claude, cost_gemini_flash, timed

logger = logging.getLogger(__name__)

# The Anthropic SDK retries 408/409/429/5xx with exponential backoff + jitter itself.
_client_kwargs = {
    "api_key": settings.anthropic_api_key,
    "timeout": settings.llm_timeout_seconds,
    "max_retries": settings.llm_max_retries,
}
if os.environ.get("DISABLE_SSL_VERIFY") == "1":
    _client_kwargs["http_client"] = httpx.Client(verify=False)
# No key: skip Claude entirely rather than failing (and logging) on every request.
_anthropic_client = anthropic.Anthropic(**_client_kwargs) if settings.anthropic_api_key else None
if _anthropic_client is None:
    logger.warning("ANTHROPIC_API_KEY is not set: generation will use Gemini only")
genai.configure(api_key=settings.google_api_key, transport="rest")

SYSTEM_PROMPT = GENERATION_SYSTEM_PROMPT_V2


def _source_refs(sources: list[SourceInfo]) -> dict[str, str]:
    """
    Map short refs (S1, S2, ...) to filenames. The LLM only ever sees the refs:
    filenames can contain PHI ("smith_john_labs.pdf") and are an injection
    vector inside an XML attribute.
    """
    refs: dict[str, str] = {}
    for s in sources:
        if s.filename not in refs.values():
            refs[f"S{len(refs) + 1}"] = s.filename
    return refs


def _build_context(sources: list[SourceInfo], refs: dict[str, str]) -> str:
    ref_by_filename = {filename: ref for ref, filename in refs.items()}
    parts = []
    for s in sources:
        text = escape_prompt_tags(s.chunk_text)
        if s.injection_risk:
            text = wrap_risky_chunk(text)
        parts.append(
            f"<context source=\"{ref_by_filename[s.filename]}\" chunk_index=\"{s.chunk_index}\">\n"
            f"{text}\n"
            "</context>"
        )
    return "\n\n".join(parts)


def _build_user_message(question: str, sources: list[SourceInfo], refs: dict[str, str]) -> str:
    return f"{_build_context(sources, refs)}\n\n<user_query>{escape_prompt_tags(question)}</user_query>"


def _resolve_citation_refs(answer: str, refs: dict[str, str]) -> str:
    """Rewrite [Source: S1 | Chunk: 4] to [Source: report.pdf | Chunk: 4]. Unknown refs are left for validation to strip."""
    def _sub(match):
        filename = refs.get(match.group(1).strip())
        return f"[Source: {filename} | Chunk: {match.group(2)}]" if filename else match.group(0)

    return CITATION_PATTERN.sub(_sub, answer)


def generate_with_claude(user_message: str) -> tuple[str, ActionUsage]:
    with timed() as ms:
        message = _anthropic_client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    tokens_in = message.usage.input_tokens
    tokens_out = message.usage.output_tokens
    usage = ActionUsage(
        action="generation",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cost_usd=cost_claude(tokens_in, tokens_out),
        latency_ms=ms[0],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text:
        raise ValueError(f"Claude returned no text (stop_reason={message.stop_reason})")
    return text, usage


def generate_with_gemini(user_message: str) -> tuple[str, ActionUsage]:
    model = genai.GenerativeModel(
        settings.gemini_generation_model,
        system_instruction=SYSTEM_PROMPT,
    )
    with timed() as ms:
        response = with_retries(
            lambda: model.generate_content(user_message, request_options={"timeout": settings.llm_timeout_seconds}),
            operation="gemini_generation",
        )

    meta = getattr(response, "usage_metadata", None)
    tokens_in = getattr(meta, "prompt_token_count", 0) or 0
    tokens_out = getattr(meta, "candidates_token_count", 0) or 0
    usage = ActionUsage(
        action="generation",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cost_usd=cost_gemini_flash(tokens_in, tokens_out),
        latency_ms=ms[0],
    )
    return response.text, usage


def generate(question: str, sources: list[SourceInfo]) -> tuple[str, str, ActionUsage]:
    """
    Generate an answer. Tries Claude first; falls back to Gemini on any exception.

    Returns:
        (answer_text, model_used, ActionUsage) — citations already mapped back to filenames.
    """
    refs = _source_refs(sources)
    user_message = _build_user_message(question, sources, refs)

    if _anthropic_client is not None:
        try:
            answer, usage = generate_with_claude(user_message)
            return _resolve_citation_refs(answer, refs), "claude", usage
        except Exception as exc:
            logger.warning("Claude generation failed (%s); falling back to Gemini", type(exc).__name__)

    try:
        answer, usage = generate_with_gemini(user_message)
        return _resolve_citation_refs(answer, refs), "gemini-fallback", usage
    except Exception as exc:
        logger.error("Gemini fallback also failed (%s)", type(exc).__name__)
        raise RuntimeError("Both Claude and Gemini generation failed.") from exc
