"""Compares two runs item by item, and acts as the CI regression gate.

Usage:
  python -m evals.compare --a <run_id> --b <run_id>
  python -m evals.compare --gate [--b-pipeline v1_hybrid]
      Latest run against evals/baseline.json. Exits 1 on a regression.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.harness.metrics import paired_bootstrap
from evals.harness.results import load_results, load_run

EVALS_DIR = Path("evals")
RETRIEVAL_METRICS = ("recall@1", "recall@5", "recall@10", "all_found@10", "mrr", "ndcg@10")
GATED_RETRIEVAL = ("recall@5", "all_found@10", "mrr")
# Ingestion scores are one number per run, so these use a fixed tolerance. The tolerance is the
# smallest visible change: one must_contain sentence in about 230, rounded up.
INGESTION_TOLERANCE = 0.005
HIGHER_IS_BETTER = {
    "outcome_accuracy": True,
    "text_recall": True,
    "table_cell_line_recall": True,
    "evidence_intact": True,
    "boilerplate_residual": False,
}
PRIMARY = "recall@5"

Rows = list[dict[str, Any]]


def _answerable_test(rows: Rows) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows if row["split"] == "test" and PRIMARY in row}


def compare_runs(
    a_record: dict[str, Any], a_rows: Rows, b_record: dict[str, Any], b_rows: Rows
) -> dict[str, Any]:
    """Deltas are b minus a, on test-split items present in both runs."""
    a_by_id, b_by_id = _answerable_test(a_rows), _answerable_test(b_rows)
    shared = sorted(set(a_by_id) & set(b_by_id))

    retrieval = {
        metric: paired_bootstrap(
            [a_by_id[i][metric] for i in shared], [b_by_id[i][metric] for i in shared]
        )
        for metric in RETRIEVAL_METRICS
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for item_id in shared:
        groups[a_by_id[item_id]["category"]].append(item_id)
    by_category = {
        name: {
            "items": len(ids),
            "a": sum(a_by_id[i][PRIMARY] for i in ids) / len(ids),
            "b": sum(b_by_id[i][PRIMARY] for i in ids) / len(ids),
        }
        for name, ids in sorted(groups.items())
    }
    a_ingest = a_record["metrics"]["ingestion"]
    b_ingest = b_record["metrics"]["ingestion"]
    return {
        "a": a_record["run_id"],
        "b": b_record["run_id"],
        "paired_items": len(shared),
        "retrieval": retrieval,
        "by_category": by_category,
        "ingestion": {
            name: {"a": a_ingest.get(name), "b": b_ingest.get(name)} for name in HIGHER_IS_BETTER
        },
        "crashed_files": b_ingest.get("crashed_files", []),
        "same_golden": a_record.get("golden") == b_record.get("golden"),
        "worse": [i for i in shared if b_by_id[i][PRIMARY] < a_by_id[i][PRIMARY]],
        "better": [i for i in shared if b_by_id[i][PRIMARY] > a_by_id[i][PRIMARY]],
    }


def gate_failures(comparison: dict[str, Any]) -> list[str]:
    if not comparison["same_golden"]:
        return ["golden data differs between the runs, so they can't be compared"]
    failures = [
        f"retrieval {name} fell by {-result['delta']:.3f} "
        f"(95% CI {result['low']:.3f} to {result['high']:.3f})"
        for name in GATED_RETRIEVAL
        if (result := comparison["retrieval"][name])["significant"] and result["delta"] < 0
    ]
    for name, values in comparison["ingestion"].items():
        if values["a"] is None or values["b"] is None:
            continue
        change = values["b"] - values["a"]
        change = change if HIGHER_IS_BETTER[name] else -change
        if change < -INGESTION_TOLERANCE:
            failures.append(f"ingestion {name} moved from {values['a']:.3f} to {values['b']:.3f}")
    if comparison["crashed_files"]:
        failures.append(f"ingestion crashed on {len(comparison['crashed_files'])} file(s)")
    return failures


def load_comparison(a: str, b: str, evals_dir: Path = EVALS_DIR) -> dict[str, Any]:
    records = {r["run_id"]: r for r in load_results(evals_dir / "results.jsonl")}
    runs = evals_dir / "runs"
    return compare_runs(records[a], load_run(runs, a), records[b], load_run(runs, b))


def baseline_run_id(evals_dir: Path = EVALS_DIR) -> str:
    return json.loads((evals_dir / "baseline.json").read_text("utf-8"))["run_id"]


def latest_run_id(evals_dir: Path = EVALS_DIR, pipeline: str | None = None) -> str:
    """The newest retrieval run, optionally for one pipeline. Generation runs are skipped."""
    for record in reversed(load_results(evals_dir / "results.jsonl")):
        if "ingestion" not in record["metrics"]:
            continue
        if pipeline is None or record["config"].get("pipeline") == pipeline:
            return record["run_id"]
    raise SystemExit(f"No run found for pipeline {pipeline!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--b-pipeline", help="Use the latest run of this pipeline as B.")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    a = args.a or baseline_run_id()
    b = args.b or latest_run_id(pipeline=args.b_pipeline)
    comparison = load_comparison(a, b)
    for name, result in comparison["retrieval"].items():
        flag = "significant" if result["significant"] else "within noise"
        print(
            f"{name:14} {result['delta']:+.3f} [{result['low']:+.3f}, {result['high']:+.3f}] {flag}"
        )
    failures = gate_failures(comparison) if args.gate else []
    for failure in failures:
        print(f"REGRESSION: {failure}")
    if args.gate:
        print("Gate failed." if failures else "Gate passed.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
