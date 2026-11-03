"""Scores retrieval per item, so any two runs can be compared item by item."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from evals.harness.matching import evidence_rank
from evals.harness.metrics import (
    all_found_at_k,
    bootstrap_mean_ci,
    mrr,
    ndcg_at_k,
    percentile,
    recall_at_k,
    refusal_scores,
)
from evals.harness.schema import QaItem, Retrieval

KS = (1, 3, 5, 10)
SUMMARY_KEYS = (
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "all_found@5",
    "all_found@10",
    "mrr",
    "ndcg@10",
)


def category_of(item: QaItem) -> str:
    return item.tags[0]


def split_of_item(item: QaItem) -> str:
    return next(tag.split(":", 1)[1] for tag in item.tags if tag.startswith("split:"))


def score_item(item: QaItem, retrieval: Retrieval) -> dict[str, Any]:
    scored: dict[str, Any] = {
        "id": item.id,
        "category": category_of(item),
        "split": split_of_item(item),
        "confidence": retrieval.confidence,
        "refused": retrieval.refused,
        "latency_ms": retrieval.latency_ms,
    }
    if item.expected.refuse:
        return scored
    ranked = [(chunk.file, chunk.text) for chunk in retrieval.chunks]
    ranks = [evidence_rank(ranked, evidence) for evidence in item.expected.evidence]
    scored["ranks"] = ranks
    for k in KS:
        scored[f"recall@{k}"] = recall_at_k(ranks, k)
    scored["all_found@5"] = all_found_at_k(ranks, 5)
    scored["all_found@10"] = all_found_at_k(ranks, 10)
    scored["mrr"] = mrr(ranks)
    scored["ndcg@10"] = ndcg_at_k(ranks, 10)
    if item.expected.forbidden_files:
        forbidden = set(item.expected.forbidden_files)
        scored["stale_leak@5"] = 1.0 if forbidden & {c.file for c in retrieval.chunks[:5]} else 0.0
    return scored


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if "recall@5" in row]
    summary: dict[str, Any] = {"items": len(answerable)}
    for key in SUMMARY_KEYS:
        summary[key] = bootstrap_mean_ci([row[key] for row in answerable])
    stale = [row["stale_leak@5"] for row in rows if "stale_leak@5" in row]
    summary["stale_leak@5"] = bootstrap_mean_ci(stale) if stale else None
    latencies = [row["latency_ms"] for row in rows]
    summary["latency_ms"] = {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answerable:
        groups[row["category"]].append(row)
    summary["by_category"] = {
        name: {
            "items": len(group),
            **{key: sum(row[key] for row in group) / len(group) for key in SUMMARY_KEYS},
        }
        for name, group in sorted(groups.items())
    }
    return summary


def evaluate_retrieval(
    items: Sequence[QaItem], results: Mapping[str, Retrieval]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [score_item(item, results[item.id]) for item in items]
    metrics = {
        split: _summary([row for row in rows if row["split"] == split]) for split in ("dev", "test")
    }
    return metrics, rows


def calibrate_gate(items: Sequence[QaItem], confidence: Mapping[str, float]) -> dict[str, Any]:
    """Refuse when confidence is below a threshold. The threshold is chosen on dev only.

    It picks the best F1 of refusing. Ties go to the lowest threshold, which answers more often.
    """
    dev = [item for item in items if split_of_item(item) == "dev"]
    test = [item for item in items if split_of_item(item) == "test"]
    dev_refuse = [item.expected.refuse for item in dev]
    if not (any(dev_refuse) and not all(dev_refuse)):
        raise ValueError("dev split needs both answerable and refusal items to calibrate a gate")

    dev_scores = sorted({confidence[item.id] for item in dev})
    candidates = [(a + b) / 2 for a, b in zip(dev_scores, dev_scores[1:], strict=False)]
    candidates.append(dev_scores[-1] + 1e-6)

    def scores_at(group: Sequence[QaItem], threshold: float) -> dict[str, float]:
        return refusal_scores(
            [item.expected.refuse for item in group],
            [confidence[item.id] < threshold for item in group],
        )

    threshold = max(candidates, key=lambda t: (scores_at(dev, t)["f1"], -t))
    return {
        "signal": "top vector cosine",
        "threshold": threshold,
        "dev": scores_at(dev, threshold),
        "test": scores_at(test, threshold) if test else None,
    }
