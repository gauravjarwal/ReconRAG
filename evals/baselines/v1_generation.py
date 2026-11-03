"""v1's prompt assembly. The real module imports app settings, so the few lines are repeated here.

The system prompt itself comes verbatim from `v1_frozen/prompts.py`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from evals.baselines.v1_frozen.prompts import GENERATION_SYSTEM_PROMPT_V2
from evals.harness.schema import Chunk

SYSTEM_PROMPT = GENERATION_SYSTEM_PROMPT_V2
NO_ANSWER = (
    "I could not find sufficient relevant information in the collection to answer this question."
)
_PROMPT_TAG = re.compile(r"(?i)<(/?)(context|user_query|query|chunks)\b")


def _escape(text: str) -> str:
    return _PROMPT_TAG.sub(r"&lt;\1\2", text)


def build_user_message(question: str, chunks: Sequence[Chunk]) -> str:
    """Filenames stay out of the prompt. The model sees S1, S2 and so on, as in v1."""
    blocks = [
        f'<context source="S{i}" chunk_index="0">\n{_escape(chunk.text)}\n</context>'
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks) + f"\n\n<user_query>{_escape(question)}</user_query>"


def citation_targets(chunks: Sequence[Chunk]) -> set[tuple[str, int]]:
    return {(f"S{i}", 0) for i in range(1, len(chunks) + 1)}
