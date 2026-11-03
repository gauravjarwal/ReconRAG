"""What every model provider must do. SDK details stop at the adapter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Usage:
    tokens_in: int
    tokens_out: int


class Provider(Protocol):
    name: str
    model: str

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Usage]:
        """Yield text fragments, then one Usage. Raises UpstreamUnavailable or UpstreamRejected."""
        ...
