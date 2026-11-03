"""Runs the free eval suites: ingestion, retrieval and the refusal gate.

Usage: python -m evals.run [--pipelines v1_hybrid,v1_vector] [--label baseline]
Needs no API key. The local embedding model downloads once.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.baselines.v1 import V1Pipeline
from evals.corpus.build import DEFAULT_OUT, build_corpus
from evals.harness.embedder import FastEmbedder
from evals.harness.ingest_eval import evaluate_ingestion
from evals.harness.metrics import percentile
from evals.harness.results import append_result, code_version, file_sha, save_run
from evals.harness.retrieval_eval import KS, calibrate_gate, evaluate_retrieval
from evals.harness.schema import Pipeline, QaItem

EVALS_DIR = Path("evals")
GOLDEN_DIR = EVALS_DIR / "golden"
RETRIEVE_K = max(KS)
PIPELINES = ("v1_vector", "v1_bm25", "v1_hybrid", "v1_default")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def ensure_corpus(corpus_dir: Path = DEFAULT_OUT) -> Path:
    """Always rebuilds. It takes seconds, and a stale corpus would quietly skew every number."""
    build_corpus(corpus_dir)
    return corpus_dir / "files"


def run_eval(pipeline: Pipeline, label: str, out_dir: Path = EVALS_DIR) -> dict[str, Any]:
    files_dir = ensure_corpus()
    golden_files = load_jsonl(GOLDEN_DIR / "ingest_v1.jsonl")
    items = [QaItem.model_validate(row) for row in load_jsonl(GOLDEN_DIR / "qa_v1.jsonl")]

    names = [entry["input"]["file"] for entry in golden_files]
    file_results = pipeline.ingest(str(files_dir), names)
    results = {item.id: pipeline.retrieve(item.input, RETRIEVE_K) for item in items}

    retrieval, rows = evaluate_retrieval(items, results)
    gate = calibrate_gate(items, {item_id: r.confidence for item_id, r in results.items()})
    latencies = [r.latency_ms for r in results.values()]
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{pipeline.name}-{uuid.uuid4().hex[:6]}"
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "label": label,
        **code_version(),
        "config": pipeline.config,
        "golden": {
            "qa": file_sha(GOLDEN_DIR / "qa_v1.jsonl"),
            "ingest": file_sha(GOLDEN_DIR / "ingest_v1.jsonl"),
            "corpus": file_sha(DEFAULT_OUT / "manifest.json"),
        },
        "metrics": {
            "ingestion": evaluate_ingestion(golden_files, file_results, items),
            "retrieval": retrieval,
            "refusal_gate": gate,
            "built_in_refusal_rate": sum(r.refused for r in results.values()) / len(results),
        },
        "cost_usd": 0.0,
        "latency": {
            "retrieve_p50_ms": percentile(latencies, 50),
            "retrieve_p95_ms": percentile(latencies, 95),
        },
    }
    save_run(out_dir / "runs", run_id, rows)
    append_result(out_dir / "results.jsonl", record)
    return record


def build_pipeline(name: str, embedder: FastEmbedder) -> Pipeline:
    if not name.startswith("v1_"):
        raise SystemExit(f"Unknown pipeline {name!r}. Known: {', '.join(PIPELINES)}")
    return V1Pipeline(name.removeprefix("v1_"), embedder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipelines", default=",".join(PIPELINES))
    parser.add_argument("--label", default="manual")
    args = parser.parse_args()
    embedder = FastEmbedder()
    for name in args.pipelines.split(","):
        record = run_eval(build_pipeline(name, embedder), args.label)
        test = record["metrics"]["retrieval"]["test"]
        ingestion = record["metrics"]["ingestion"]
        gate = record["metrics"]["refusal_gate"]
        print(
            f"{name:12} recall@5 {test['recall@5']['mean']:.3f} "
            f"[{test['recall@5']['low']:.3f}, {test['recall@5']['high']:.3f}]  "
            f"mrr {test['mrr']['mean']:.3f}  outcome {ingestion['outcome_accuracy']:.3f}  "
            f"gate P/R {gate['test']['precision']:.2f}/{gate['test']['recall']:.2f}  "
            f"p95 {record['latency']['retrieve_p95_ms']:.1f} ms"
        )


if __name__ == "__main__":
    main()
