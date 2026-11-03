import pytest

from evals.harness.retrieval_eval import calibrate_gate, evaluate_retrieval, score_item
from evals.harness.schema import Chunk, Evidence, Expected, QaItem, Retrieval


def _item(item_id="q1", category="fact", split="test", refuse=False, forbidden=(), evidence=None):
    evidence = (
        [] if refuse else evidence or [Evidence(files=["a.txt"], quote="the excess is Rs. 500")]
    )
    return QaItem(
        id=item_id,
        input="?",
        tags=[category, f"split:{split}"],
        expected=Expected(
            answer="a",
            refuse=refuse,
            evidence=evidence,
            forbidden_files=list(forbidden),
        ),
    )


def _retrieval(*pairs, confidence=0.5):
    return Retrieval(chunks=[Chunk(f, t) for f, t in pairs], confidence=confidence, latency_ms=2.0)


def test_score_finds_the_rank_of_the_evidence():
    scored = score_item(_item(), _retrieval(("b.txt", "noise"), ("a.txt", "the excess is Rs. 500")))
    assert scored["ranks"] == [2]
    assert scored["recall@1"] == 0.0
    assert scored["recall@3"] == 1.0
    assert scored["mrr"] == 0.5


def test_score_flags_a_stale_file_in_the_top_five():
    scored = score_item(
        _item(forbidden=["old.pdf"]),
        _retrieval(("old.pdf", "x"), ("a.txt", "the excess is Rs. 500")),
    )
    assert scored["stale_leak@5"] == 1.0


def test_refusal_items_get_no_retrieval_score():
    scored = score_item(_item(refuse=True), _retrieval(("a.txt", "x")))
    assert "recall@5" not in scored


def test_evaluate_groups_results_by_split_and_category():
    items = [_item("q1", split="test"), _item("q2", "table", split="dev")]
    results = {
        "q1": _retrieval(("a.txt", "the excess is Rs. 500")),
        "q2": _retrieval(("b.txt", "noise")),
    }
    metrics, _ = evaluate_retrieval(items, results)
    assert metrics["test"]["recall@5"]["mean"] == 1.0
    assert metrics["dev"]["recall@5"]["mean"] == 0.0
    assert metrics["test"]["by_category"]["fact"]["recall@5"] == 1.0


def test_gate_threshold_comes_from_dev_and_is_reported_on_test():
    items = [
        _item("d1", split="dev", refuse=True),
        _item("d2", split="dev"),
        _item("t1", split="test", refuse=True),
        _item("t2", split="test"),
    ]
    confidence = {"d1": 0.2, "d2": 0.8, "t1": 0.3, "t2": 0.6}
    result = calibrate_gate(items, confidence)
    assert 0.2 < result["threshold"] <= 0.8
    assert result["test"]["recall"] == 1.0
    assert result["test"]["precision"] == 1.0


def test_gate_keeps_the_threshold_blind_to_test_items():
    items = [
        _item("d1", split="dev", refuse=True),
        _item("d2", split="dev"),
        _item("t1", split="test", refuse=True),
    ]
    first = calibrate_gate(items, {"d1": 0.2, "d2": 0.8, "t1": 0.1})
    second = calibrate_gate(items, {"d1": 0.2, "d2": 0.8, "t1": 0.95})
    assert first["threshold"] == pytest.approx(second["threshold"])


def test_gate_needs_both_classes_on_dev():
    with pytest.raises(ValueError, match="dev"):
        calibrate_gate([_item("d1", split="dev")], {"d1": 0.2})
