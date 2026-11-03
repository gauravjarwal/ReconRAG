"""Run records. `results.jsonl` holds one summary per run; `runs/` holds per-item scores."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def code_version() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--short")),
    }


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def append_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def save_run(folder: Path, run_id: str, rows: list[dict[str, Any]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{run_id}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)


def load_run(folder: Path, run_id: str) -> list[dict[str, Any]]:
    with gzip.open(folder / f"{run_id}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
