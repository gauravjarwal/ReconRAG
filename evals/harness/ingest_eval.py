"""Scores each ingestion stage against what the corpus is known to contain."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from evals.harness.matching import contains_evidence, evidence_matches, normalize
from evals.harness.schema import FileResult, QaItem

GoldenFile = dict[str, Any]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _cell_on_one_line(text: str, row: str, value: str) -> bool:
    row_norm = normalize(row)
    value_digits = value.replace(",", "").replace("Rs. ", "")
    return any(
        row_norm in normalize(line) and value_digits in line.replace(",", "")
        for line in text.splitlines()
    )


def _evidence_intact(results: dict[str, FileResult], items: Sequence[QaItem]) -> float | None:
    """Share of quotes that sit whole inside at least one chunk of a correct file."""
    scores: list[float] = []
    for item in items:
        for evidence in item.expected.evidence:
            if not evidence.quote:
                continue
            chunks = (c for f in evidence.files if f in results for c in results[f].chunks)
            scores.append(1.0 if any(evidence_matches(c, evidence) for c in chunks) else 0.0)
    return _mean(scores)


def evaluate_ingestion(
    golden: Sequence[GoldenFile], results: Sequence[FileResult], items: Sequence[QaItem]
) -> dict[str, Any]:
    by_file = {result.file: result for result in results}
    correct: list[float] = []
    outcome_scores: dict[str, list[float]] = defaultdict(list)
    text_scores: dict[str, list[float]] = defaultdict(list)
    table_scores: dict[str, list[float]] = defaultdict(list)
    boilerplate: list[float] = []

    for entry in golden:
        result = by_file.get(entry["input"]["file"], FileResult(entry["input"]["file"], "missing"))
        expected = entry["expected"]
        fmt = entry["input"]["format"]
        hit = 1.0 if result.outcome == expected["outcome"] else 0.0
        correct.append(hit)
        outcome_scores[expected["outcome"]].append(hit)
        if expected["outcome"] != "indexed":
            continue
        for sentence in expected.get("must_contain", []):
            text_scores[fmt].append(1.0 if contains_evidence(result.text, sentence) else 0.0)
        for row, _column, value in expected.get("table_cells", []):
            table_scores[fmt].append(1.0 if _cell_on_one_line(result.text, row, value) else 0.0)
        chunk_text = "\n".join(result.chunks)
        for phrase in expected.get("must_not_contain", []):
            boilerplate.append(1.0 if phrase.casefold() in chunk_text.casefold() else 0.0)

    all_text = [score for scores in text_scores.values() for score in scores]
    all_table = [score for scores in table_scores.values() for score in scores]
    return {
        "files": len(golden),
        "outcome_accuracy": _mean(correct),
        "outcome_recall": {name: _mean(scores) for name, scores in sorted(outcome_scores.items())},
        "crashed_files": sorted(r.file for r in results if r.outcome == "crashed"),
        "text_recall": _mean(all_text),
        "text_recall_by_format": {k: _mean(v) for k, v in sorted(text_scores.items())},
        "table_cell_line_recall": _mean(all_table),
        "table_cell_line_recall_by_format": {k: _mean(v) for k, v in sorted(table_scores.items())},
        "boilerplate_residual": _mean(boilerplate),
        "evidence_intact": _evidence_intact(by_file, items),
        "chunks": sum(len(r.chunks) for r in results),
    }
