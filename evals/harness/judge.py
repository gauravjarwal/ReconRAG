"""LLM judge for faithfulness. It is only trusted as far as its measured agreement goes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from evals.harness.llm import LlmClient, LlmResponse
from evals.harness.metrics import cohen_kappa, refusal_scores
from evals.harness.prompts import JUDGE_SYSTEM_V1, JUDGE_USER_V1


class Verdict(BaseModel):
    faithful: bool
    reason: str = ""


def parse_verdict(raw: str) -> Verdict:
    """A verdict that can't be parsed raises. Guessing would corrupt the score silently."""
    try:
        return Verdict.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"unusable judge output: {raw[:80]!r}") from exc


def judge_answer(
    client: LlmClient, question: str, context: str, answer: str
) -> tuple[Verdict, LlmResponse]:
    response = client.generate(
        JUDGE_SYSTEM_V1,
        JUDGE_USER_V1.format(question=question, context=context, answer=answer),
        json_output=True,
    )
    return parse_verdict(response.text), response


def judge_agreement(truth: Sequence[bool], judged: Sequence[bool]) -> dict[str, Any]:
    """'Positive' means faithful. TNR is the rate that matters most: catching bad answers."""
    scores = refusal_scores(truth, judged)
    tp, fn, fp, tn = (int(scores[key]) for key in ("tp", "fn", "fp", "tn"))
    total = tp + fn + fp + tn
    return {
        "items": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
        "true_negative_rate": tn / (tn + fp) if tn + fp else 0.0,
        "cohen_kappa": cohen_kappa(truth, judged),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }
