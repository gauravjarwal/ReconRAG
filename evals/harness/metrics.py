"""Metric code, written by hand so every number can be explained line by line.

Retrieval metrics take `ranks`: for each evidence group in a golden item, the 1-based rank of the
first retrieved chunk that holds the evidence, or None if no chunk does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Ranks = Sequence[int | None]


def recall_at_k(ranks: Ranks, k: int) -> float:
    """Share of evidence groups found in the top k."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def all_found_at_k(ranks: Ranks, k: int) -> float:
    """1.0 only if every group is in the top k. This is what a multi-hop answer needs."""
    return 1.0 if ranks and all(r is not None and r <= k for r in ranks) else 0.0


def mrr(ranks: Ranks) -> float:
    """Reciprocal rank of the best-placed group."""
    found = [rank for rank in ranks if rank is not None]
    return 1.0 / min(found) if found else 0.0


def ndcg_at_k(ranks: Ranks, k: int) -> float:
    """Binary-gain nDCG. A chunk holding two groups counts as one relevant position."""
    if not ranks:
        return 0.0
    found = {rank for rank in ranks if rank is not None}
    missing = len(ranks) - sum(1 for rank in ranks if rank is not None)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in found if rank <= k)
    ideal_positions = min(len(found) + missing, k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_positions + 1))
    return dcg / idcg if idcg else 0.0


def refusal_scores(should_refuse: Sequence[bool], did_refuse: Sequence[bool]) -> dict[str, float]:
    """Precision and recall of refusing. 'Positive' means the system refused."""
    pairs = list(zip(should_refuse, did_refuse, strict=True))
    tp = sum(1 for want, did in pairs if want and did)
    fp = sum(1 for want, did in pairs if not want and did)
    fn = sum(1 for want, did in pairs if want and not did)
    tn = sum(1 for want, did in pairs if not want and not did)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, or None for no data."""
    return float(np.percentile(values, q)) if len(values) else None


def bootstrap_mean_ci(
    values: Sequence[float], n_resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean with a 95% percentile-bootstrap interval. Items are resampled, not runs."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, (n_resamples, data.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(data.mean()), "low": float(low), "high": float(high)}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 2000, seed: int = 0
) -> dict[str, float | bool]:
    """95% interval on mean(b - a). Pairing by item cancels out how hard each question is.

    `significant` means the interval excludes zero. This is the repo's bar for saying a change
    improved something.
    """
    if len(a) != len(b):
        raise ValueError(f"paired lists differ in length: {len(a)} vs {len(b)}")
    deltas = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    summary = bootstrap_mean_ci(deltas.tolist(), n_resamples, seed)
    return {
        "delta": summary["mean"],
        "low": summary["low"],
        "high": summary["high"],
        "significant": bool(summary["low"] > 0 or summary["high"] < 0),
    }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Agreement between two labelers, after removing the agreement expected by chance."""
    pairs = list(zip(a, b, strict=True))
    if not pairs:
        return 0.0
    observed = sum(1 for x, y in pairs if x == y) / len(pairs)
    a_true = sum(1 for x, _ in pairs if x) / len(pairs)
    b_true = sum(1 for _, y in pairs if y) / len(pairs)
    chance = a_true * b_true + (1 - a_true) * (1 - b_true)
    return (observed - chance) / (1 - chance) if chance < 1 else 0.0
