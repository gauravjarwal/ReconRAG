import json

from evals.compare import compare_runs, gate_failures, latest_run_id


def _rows(values, category="fact"):
    return [
        {
            "id": f"q{i}",
            "split": "test",
            "category": category,
            "recall@5": v,
            "mrr": v,
            "all_found@10": v,
            "ndcg@10": v,
            "recall@1": v,
            "recall@10": v,
        }
        for i, v in enumerate(values)
    ]


def _record(run_id="r", **ingestion):
    base = {
        "outcome_accuracy": 0.5,
        "text_recall": 0.6,
        "table_cell_line_recall": 0.1,
        "boilerplate_residual": 0.2,
        "evidence_intact": 0.7,
    }
    return {"run_id": run_id, "metrics": {"ingestion": {**base, **ingestion}}}


def test_compare_reports_a_significant_gain():
    result = compare_runs(_record("a"), _rows([0.0] * 60), _record("b"), _rows([1.0] * 60))
    assert result["retrieval"]["recall@5"]["delta"] == 1.0
    assert result["retrieval"]["recall@5"]["significant"] is True


def test_compare_only_uses_items_present_in_both_runs():
    a = _rows([1.0, 0.0])
    b = _rows([1.0])
    assert compare_runs(_record("a"), a, _record("b"), b)["paired_items"] == 1


def test_compare_lists_items_that_got_worse():
    result = compare_runs(_record("a"), _rows([1.0, 1.0]), _record("b"), _rows([1.0, 0.0]))
    assert result["worse"] == ["q1"]
    assert result["better"] == []


def test_gate_passes_identical_runs():
    comparison = compare_runs(
        _record("a"), _rows([1.0, 0.0] * 30), _record("b"), _rows([1.0, 0.0] * 30)
    )
    assert gate_failures(comparison) == []


def test_gate_fails_a_significant_retrieval_drop():
    comparison = compare_runs(_record("a"), _rows([1.0] * 60), _record("b"), _rows([0.0] * 60))
    assert any("recall@5" in failure for failure in gate_failures(comparison))


def test_gate_ignores_a_drop_inside_the_noise():
    a = _rows([1.0] * 60)
    b = _rows([1.0] * 59 + [0.0])
    assert gate_failures(compare_runs(_record("a"), a, _record("b"), b)) == []


def test_gate_fails_an_ingestion_drop_past_tolerance():
    comparison = compare_runs(
        _record("a"), _rows([1.0] * 60), _record("b", text_recall=0.5), _rows([1.0] * 60)
    )
    assert any("text_recall" in failure for failure in gate_failures(comparison))


def test_gate_treats_more_boilerplate_as_worse():
    comparison = compare_runs(
        _record("a"), _rows([1.0] * 60), _record("b", boilerplate_residual=0.4), _rows([1.0] * 60)
    )
    assert any("boilerplate_residual" in failure for failure in gate_failures(comparison))


def test_gate_fails_any_crash():
    comparison = compare_runs(
        _record("a"), _rows([1.0] * 60), _record("b", crashed_files=["x.pdf"]), _rows([1.0] * 60)
    )
    assert any("crashed" in failure for failure in gate_failures(comparison))


def test_latest_run_can_be_filtered_by_pipeline(tmp_path):
    lines = [
        {"run_id": "1", "config": {"pipeline": "v1_hybrid"}, "metrics": {"ingestion": {}}},
        {"run_id": "2", "config": {"pipeline": "v1_bm25"}, "metrics": {"ingestion": {}}},
        {"run_id": "3", "kind": "generation", "config": {}, "metrics": {}},
    ]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert latest_run_id(tmp_path) == "2"
    assert latest_run_id(tmp_path, pipeline="v1_hybrid") == "1"


def test_gate_refuses_to_compare_runs_on_different_golden_data():
    a, b = _record("a"), _record("b")
    a["golden"] = {"qa": "one", "ingest": "x", "corpus": "y"}
    b["golden"] = {"qa": "two", "ingest": "x", "corpus": "y"}
    comparison = compare_runs(a, _rows([1.0] * 60), b, _rows([1.0] * 60))
    assert any("golden" in failure for failure in gate_failures(comparison))
