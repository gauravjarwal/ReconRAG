"""Gemini adapter. The SDK retries 408, 429 and 5xx; timeouts here are in milliseconds."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors, types

from reconrag.config import Settings
from reconrag.errors import UpstreamRejected, UpstreamUnavailable
from reconrag.llm.base import Usage

_RETRYABLE = {408, 429, 500, 502, 503, 504}


def make_client(settings: Settings) -> genai.Client:
    return genai.Client(
        api_key=settings.google_api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.llm_timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=settings.llm_max_retries + 1),
        ),
    )


def translate(exc: Exception) -> Exception:
    """Sort SDK failures into 'try again' and 'never works'."""
    if isinstance(exc, errors.APIError):
        if exc.code in _RETRYABLE:
            return UpstreamUnavailable(f"gemini returned {exc.code}")
        return UpstreamRejected(f"gemini rejected the request ({exc.code})")
    return UpstreamUnavailable(f"gemini call failed ({type(exc).__name__})")


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.gemini_model
        self._client = make_client(settings)

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Usage]:
        config = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens
        )
        tokens_in = tokens_out = 0
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=user, config=config
            ):
                meta = chunk.usage_metadata
                if meta is not None:
                    tokens_in = meta.prompt_token_count or tokens_in
                    # Thinking tokens bill as output.
                    tokens_out = (meta.candidates_token_count or 0) + (
                        meta.thoughts_token_count or 0
                    )
                if chunk.text:
                    yield chunk.text
        except (errors.APIError, httpx.HTTPError) as exc:
            raise translate(exc) from exc
        yield Usage(tokens_in, tokens_out)
