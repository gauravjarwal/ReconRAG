"""Closed-loop load test: N threads, each sending the next request when the last one returns.

Usage: python scripts/load_test.py --url http://127.0.0.1:8000 --collection docs --key KEY
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

QUESTIONS = [
    "How does consensus work in distributed systems?",
    "What is the role of attention in transformers?",
    "How do massive stars end their lives?",
    "What is the CAP theorem?",
    "How does gradient descent train a network?",
    "What is dark energy?",
    "What does a load balancer do?",
    "Explain the single responsibility principle.",
]


def worker(
    url: str, headers: dict[str, str], stop_at: float, seed: int, questions: list[str]
) -> tuple[list[float], int, int]:
    latencies: list[float] = []
    errors = 0
    refused = 0
    index = seed * 37
    with httpx.Client(headers=headers, timeout=60) as client:
        while time.monotonic() < stop_at:
            question = questions[index % len(questions)]
            index += 1
            started = time.perf_counter()
            response = client.post(url, json={"question": question})
            latencies.append((time.perf_counter() - started) * 1000)
            errors += response.status_code != 200
            refused += response.status_code == 200 and response.json()["refused"]
    return latencies, errors, refused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", default="http://127.0.0.1:8000", help="Comma-separated replicas")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--concurrency", default="1,5,10,20")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--questions", type=Path, help="JSONL with a `query` field per line")
    args = parser.parse_args()
    questions = QUESTIONS
    if args.questions:
        lines = args.questions.read_text("utf-8").splitlines()
        questions = [json.loads(line)["input"] for line in lines if line]
    urls = [f"{u}/v1/collections/{args.collection}/query" for u in args.urls.split(",")]
    headers = {"X-API-Key": args.key}
    for level in (int(n) for n in args.concurrency.split(",")):
        stop_at = time.monotonic() + args.seconds
        with ThreadPoolExecutor(level) as pool:
            jobs = [
                pool.submit(worker, urls[i % len(urls)], headers, stop_at, i, questions)
                for i in range(level)
            ]
            results = [job.result() for job in jobs]
        latencies = sorted(x for r, _, _ in results for x in r)
        print(
            json.dumps(
                {
                    "concurrency": level,
                    "replicas": len(urls),
                    "requests": len(latencies),
                    "errors": sum(e for _, e, _ in results),
                    "refused": sum(r for _, _, r in results),
                    "rps": round(len(latencies) / args.seconds, 1),
                    "p50_ms": round(statistics.median(latencies), 1),
                    "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1),
                }
            )
        )


if __name__ == "__main__":
    main()
