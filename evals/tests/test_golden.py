"""The frozen golden sets, checked against the corpus and the spec's coverage minimums."""

import json
from collections import Counter
from pathlib import Path

import pytest

from evals.corpus.build import CorpusBuilder
from evals.corpus.golden import build_golden, category, split_of
from evals.harness.matching import evidence_matches
from evals.harness.schema import QaItem

GOLDEN = Path("evals/golden")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    builder = CorpusBuilder(tmp_path_factory.mktemp("corpus"))
    docs = builder.build()
    return builder, docs, build_golden(builder, docs)


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (GOLDEN / name).read_text("utf-8").splitlines()]


@pytest.mark.parametrize("name", ["ingest_v1", "qa_v1", "judge_v1", "judge_human_queue_v1"])
def test_frozen_file_matches_what_the_corpus_generates(built, name):
    assert _load(f"{name}.jsonl") == built[2][name]


def test_every_qa_item_parses():
    assert all(QaItem.model_validate(item) for item in _load("qa_v1.jsonl"))


def test_qa_ids_are_unique():
    ids = [item["id"] for item in _load("qa_v1.jsonl")]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("name", "minimum"),
    [
        ("refuse", 20),
        ("multi_hop", 20),
        ("table", 15),
        ("ocr", 15),
        ("version", 10),
        ("adversarial", 10),
        ("identifier", 10),
        ("paraphrase", 10),
    ],
)
def test_category_minimums_from_the_spec(name, minimum):
    counts = Counter(category(QaItem.model_validate(i)) for i in _load("qa_v1.jsonl"))
    assert counts[name] >= minimum


def test_qa_set_has_at_least_150_items():
    assert len(_load("qa_v1.jsonl")) >= 150


def test_every_evidence_is_present_in_the_truth_of_each_file(built):
    _, docs, golden = built
    truth = {d.file: d.truth_text for d in docs}
    for raw in golden["qa_v1"]:
        item = QaItem.model_validate(raw)
        for evidence in item.expected.evidence:
            for file in evidence.files:
                assert evidence_matches(truth[file], evidence), (item.id, file)


def test_a_refusal_item_has_no_evidence_and_an_answerable_item_has_some():
    for raw in _load("qa_v1.jsonl"):
        item = QaItem.model_validate(raw)
        assert item.expected.refuse != bool(item.expected.evidence), item.id


def test_forbidden_files_never_appear_as_evidence():
    for raw in _load("qa_v1.jsonl"):
        item = QaItem.model_validate(raw)
        allowed = {f for e in item.expected.evidence for f in e.files}
        assert not allowed & set(item.expected.forbidden_files), item.id


def test_version_items_have_a_stale_file_that_states_a_different_value(built):
    _, docs, golden = built
    truth = {d.file: d.truth_text for d in docs}
    for raw in golden["qa_v1"]:
        item = QaItem.model_validate(raw)
        if category(item) != "version":
            continue
        assert item.expected.forbidden_files, item.id
        for file in item.expected.forbidden_files:
            assert not evidence_matches(truth[file], item.expected.evidence[0]), item.id


def test_refusal_questions_name_nothing_in_the_corpus(built):
    _, docs, golden = built
    text = "\n".join(d.truth_text for d in docs)
    for raw in golden["qa_v1"]:
        item = QaItem.model_validate(raw)
        for term in item.expected.absent_terms:
            assert term.casefold() not in text.casefold(), (item.id, term)


def test_split_is_stable_and_uses_both_sides():
    splits = Counter(split_of(item["id"]) for item in _load("qa_v1.jsonl"))
    assert split_of("abc") == split_of("abc")
    assert 0.3 < splits["dev"] / sum(splits.values()) < 0.5


def test_each_split_has_refusal_items():
    items = [QaItem.model_validate(i) for i in _load("qa_v1.jsonl")]
    assert sum(1 for i in items if i.expected.refuse and split_of(i.id) == "dev") >= 8
    assert sum(1 for i in items if i.expected.refuse and split_of(i.id) == "test") >= 8


def test_judge_set_is_balanced_and_large_enough():
    labels = Counter(item["expected"]["faithful"] for item in _load("judge_v1.jsonl"))
    assert labels[True] >= 50
    assert labels[False] >= 50


def test_human_queue_has_50_items_and_hides_labels():
    queue = _load("judge_human_queue_v1.jsonl")
    assert len(queue) == 50
    assert all("expected" not in item for item in queue)
