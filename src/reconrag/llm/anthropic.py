"""Claude adapter. The SDK retries connection errors, 408, 409, 429 and 5xx itself."""

from __future__ import annotations

from collections.abc import Iterator

import anthropic

from reconrag.config import Settings
from reconrag.errors import UpstreamRejected, UpstreamUnavailable
from reconrag.llm.base import Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.claude_model
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Usage]:
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                yield from stream.text_stream
                message = stream.get_final_message()
        except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            raise UpstreamUnavailable(f"anthropic returned {exc.status_code}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in (408, 409) or exc.status_code >= 500:
                raise UpstreamUnavailable(f"anthropic returned {exc.status_code}") from exc
            raise UpstreamRejected(f"anthropic rejected the request ({exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            raise UpstreamUnavailable("anthropic connection failed") from exc
        if message.stop_reason == "refusal":
            raise UpstreamRejected("anthropic declined the request")
        yield Usage(message.usage.input_tokens, message.usage.output_tokens)
