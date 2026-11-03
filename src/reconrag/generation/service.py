"""Answers a question. Streaming is the only path; the plain JSON route consumes the same events."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from reconrag.config import Settings
from reconrag.domain import Collection
from reconrag.errors import InvalidInput
from reconrag.generation.prompts import ANSWER_SYSTEM_V3, REFUSAL_MESSAGE, render_question
from reconrag.generation.validate import validate_answer
from reconrag.llm.router import CallResult, LLMRouter
from reconrag.observability.trace import Trace
from reconrag.retrieval.context import Passage
from reconrag.retrieval.service import Retriever
from reconrag.security.injection import looks_like_injection
from reconrag.security.phi import Pseudonymizer
from reconrag.storage.db import Pool

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """`kind` is one of sources, token, done."""

    kind: str
    data: dict[str, Any]


def _source(passage: Passage) -> dict[str, Any]:
    hit = passage.hit
    return {
        "ref": passage.ref,
        "chunk_id": str(hit.chunk_id),
        "document_id": str(hit.document_id),
        "filename": hit.filename,
        "section": hit.section_path,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "kind": hit.kind,
        "text": hit.text,
        "dense_similarity": hit.dense_similarity,
        "rerank_score": hit.rerank_score,
        "injection_risk": hit.injection_risk,
        "also_in": hit.signals.get("also_in", []),
    }


class Answerer:
    def __init__(
        self,
        pool: Pool,
        retriever: Retriever,
        router: LLMRouter,
        settings: Settings,
        pseudonymizer: Pseudonymizer | None = None,
    ) -> None:
        self._pool = pool
        self._retriever = retriever
        self._router = router
        self._settings = settings
        self._pseudonymizer = pseudonymizer

    def _protect(self, collection: Collection, question: str) -> str:
        if not collection.phi_protection:
            return question
        if self._pseudonymizer is None:
            raise InvalidInput("PHI protection isn't available on this server.")
        with self._pool.connection() as conn:
            return self._pseudonymizer.query(conn, collection.id, question).text

    def _reveal(self, collection: Collection, text: str) -> str:
        assert self._pseudonymizer is not None
        with self._pool.connection() as conn:
            return self._pseudonymizer.reidentify(conn, collection.id, text)

    def stream(self, collection: Collection, question: str, reidentify: bool) -> Iterator[Event]:
        """Yields sources, then tokens, then a final done event holding the validated answer.

        Tokens leave before validation, and `done` is authoritative. PHI collections send no
        tokens at all, because raw model output there must be checked before anyone sees it.
        """
        started = time.perf_counter()
        trace = Trace()
        protected = collection.phi_protection
        safe_question = self._protect(collection, question)
        flagged = looks_like_injection(question)
        retrieved = self._retriever.retrieve(collection.id, safe_question, trace)
        reveal = reidentify and protected

        def visible(text: str) -> str:
            return self._reveal(collection, text) if reveal else text

        sources = [_source(p) for p in retrieved.passages]
        if reveal:
            for source in sources:
                source["text"] = visible(source["text"])
        yield Event("sources", {"sources": sources})

        base = {
            "confidence": retrieved.confidence,
            "degraded": retrieved.degraded,
            "injection_flagged": flagged,
            "phi_protected": protected,
            "reidentified": reveal,
        }

        def done(**fields: Any) -> Event:
            trace.stages_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
            return Event("done", {**base, "stages_ms": trace.stages_ms, **fields})

        if retrieved.refused:
            yield done(answer=REFUSAL_MESSAGE, refused=True, reason=retrieved.reason, cited=[])
            return
        if not self._router.available:
            yield done(
                answer=None,
                refused=False,
                reason="no_model_configured",
                cited=[],
                warnings=["answers_disabled"],
            )
            return

        items = [(p.ref, p.text, p.hit.injection_risk) for p in retrieved.passages]
        user = render_question(safe_question, items)
        parts: list[str] = []
        result: CallResult | None = None
        for event in self._router.stream(
            ANSWER_SYSTEM_V3, user, self._settings.llm_max_output_tokens
        ):
            if isinstance(event, CallResult):
                result = event
            else:
                parts.append(event)
                if not protected:
                    yield Event("token", {"text": event})
        assert result is not None
        refs = {p.ref: p.text for p in retrieved.passages}
        checked = validate_answer("".join(parts), refs, self._settings.max_answer_chars)
        yield done(
            answer=visible(checked.answer),
            refused=False,
            reason="ok",
            cited=checked.cited,
            warnings=checked.warnings,
            unverified_numbers=checked.unverified_numbers,
            model=asdict(result),
        )

    def answer(self, collection: Collection, question: str, reidentify: bool) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        for event in self.stream(collection, question, reidentify):
            if event.kind == "sources":
                sources = event.data["sources"]
            elif event.kind == "done":
                final = event.data
        return {**final, "sources": sources}
