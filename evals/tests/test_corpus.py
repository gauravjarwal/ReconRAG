import hashlib
from collections import Counter
from pathlib import Path

import pytest

from evals.corpus.build import build_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    return out, build_corpus(out)


def _hashes(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_corpus_has_at_least_200_files_in_9_formats(corpus):
    _, docs = corpus
    assert len(docs) >= 200
    assert len({d.format for d in docs}) >= 9


def test_every_manifest_entry_exists_on_disk(corpus):
    out, docs = corpus
    assert {d.file for d in docs} == {p.name for p in (out / "files").iterdir()}


def test_rebuild_is_byte_identical(corpus, tmp_path):
    out, _ = corpus
    build_corpus(tmp_path)
    assert _hashes(out / "files") == _hashes(tmp_path / "files")


def test_corpus_includes_each_kind_of_bad_outcome(corpus):
    _, docs = corpus
    outcomes = Counter(d.outcome for d in docs)
    assert outcomes["quarantined"] >= 8
    assert outcomes["duplicate"] >= 10
    assert outcomes["superseded"] >= 10


def test_duplicates_point_at_a_real_indexed_file(corpus):
    _, docs = corpus
    by_file = {d.file: d for d in docs}
    for doc in docs:
        if doc.outcome == "duplicate":
            assert by_file[doc.duplicate_of].outcome == "indexed"


def test_superseded_files_point_at_a_newer_indexed_file(corpus):
    _, docs = corpus
    by_file = {d.file: d for d in docs}
    for doc in docs:
        if doc.outcome == "superseded":
            assert by_file[doc.superseded_by].outcome == "indexed"
