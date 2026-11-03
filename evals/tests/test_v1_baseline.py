import subprocess
from pathlib import Path

import numpy as np
import pytest

from evals.baselines.v1 import V1Pipeline

FROZEN = Path("evals/baselines/v1_frozen")


class FakeEmbedder:
    """Word-hash vectors: deterministic, offline, and enough to prove the wiring."""

    name = "fake"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(64)
        for word in text.lower().split():
            vector[sum(word.encode()) % 64] += 1
        return vector / (np.linalg.norm(vector) or 1)

    def passages(self, texts):
        return np.array([self._vector(t) for t in texts])

    def query(self, text):
        return self._vector(text)


@pytest.mark.parametrize("name", ["parsing", "chunking", "bm25", "fusion", "prompts"])
def test_frozen_file_is_byte_identical_to_the_v1_commit(name):
    committed = subprocess.run(
        ["git", "show", f"f5cf1b3:backend/pipelines/{name}.py"],
        capture_output=True,
        check=True,
    ).stdout
    assert (FROZEN / f"{name}.py").read_bytes() == committed


@pytest.fixture
def files(tmp_path):
    (tmp_path / "plan.txt").write_text(
        "The excess on the Gold plan is two thousand five hundred rupees for every claim made.\n"
        * 3
    )
    (tmp_path / "other.md").write_text(
        "Our branch offices close early on the second Saturday of each and every month.\n" * 3
    )
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "table.xlsx").write_bytes(b"PK")
    return tmp_path


def test_unsupported_type_is_rejected_like_the_v1_api(files):
    pipeline = V1Pipeline("hybrid", FakeEmbedder())
    results = {r.file: r for r in pipeline.ingest(str(files), ["table.xlsx"])}
    assert results["table.xlsx"].outcome == "quarantined"
    assert "unsupported" in results["table.xlsx"].error


def test_content_check_rejects_a_disguised_pdf(files):
    pipeline = V1Pipeline("hybrid", FakeEmbedder())
    result = pipeline.ingest(str(files), ["broken.pdf"])[0]
    assert result.outcome == "quarantined"


def test_v1_indexes_text_files_and_retrieves_them(files):
    pipeline = V1Pipeline("hybrid", FakeEmbedder())
    pipeline.ingest(str(files), ["plan.txt", "other.md"])
    retrieval = pipeline.retrieve("excess on the Gold plan", k=2)
    assert retrieval.chunks[0].file == "plan.txt"


def test_default_mode_returns_at_most_three_chunks_and_never_refuses(files):
    pipeline = V1Pipeline("default", FakeEmbedder())
    pipeline.ingest(str(files), ["plan.txt", "other.md"])
    retrieval = pipeline.retrieve("what is the capital of Australia", k=10)
    assert len(retrieval.chunks) <= 3
    assert retrieval.refused is False


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        V1Pipeline("magic", FakeEmbedder())
