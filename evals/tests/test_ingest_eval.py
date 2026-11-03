from evals.harness.ingest_eval import evaluate_ingestion
from evals.harness.schema import Evidence, Expected, FileResult, QaItem


def _golden(file, outcome="indexed", must=(), must_not=(), cells=(), fmt="txt", tags=("notes",)):
    return {
        "id": f"ingest-{file}",
        "input": {"file": file, "format": fmt},
        "expected": {
            "outcome": outcome,
            "must_contain": list(must),
            "must_not_contain": list(must_not),
            "table_cells": list(cells),
        },
        "tags": list(tags),
    }


def test_outcome_accuracy_counts_exact_matches():
    golden = [_golden("a.txt"), _golden("b.pdf", outcome="quarantined")]
    results = [
        FileResult("a.txt", "indexed", "x", ["x"]),
        FileResult("b.pdf", "indexed", "x", ["x"]),
    ]
    metrics = evaluate_ingestion(golden, results, [])
    assert metrics["outcome_accuracy"] == 0.5
    assert metrics["outcome_recall"]["quarantined"] == 0.0
    assert metrics["outcome_recall"]["indexed"] == 1.0


def test_a_crash_is_counted_and_is_never_a_correct_outcome():
    golden = [_golden("b.pdf", outcome="quarantined")]
    metrics = evaluate_ingestion(golden, [FileResult("b.pdf", "crashed", error="boom")], [])
    assert metrics["crashed_files"] == ["b.pdf"]
    assert metrics["outcome_accuracy"] == 0.0


def test_text_recall_only_scores_files_that_should_be_indexed():
    golden = [
        _golden("a.txt", must=["the excess is Rs. 500", "missing sentence here"]),
        _golden("b.txt", outcome="quarantined", must=["ignored sentence"]),
    ]
    results = [
        FileResult("a.txt", "indexed", "the excess is Rs. 500", ["c"]),
        FileResult("b.txt", "quarantined"),
    ]
    assert evaluate_ingestion(golden, results, [])["text_recall"] == 0.5


def test_a_file_that_should_exist_but_was_dropped_scores_zero_recall():
    golden = [_golden("a.html", must=["a fact"], fmt="html", tags=("faq",))]
    metrics = evaluate_ingestion(golden, [FileResult("a.html", "quarantined")], [])
    assert metrics["text_recall"] == 0.0
    assert metrics["text_recall_by_format"]["html"] == 0.0


def test_boilerplate_is_measured_in_chunks():
    golden = [_golden("a.txt", must_not=["Accept all cookies", "Sent from my phone"])]
    results = [FileResult("a.txt", "indexed", "", ["Accept all cookies now"])]
    assert evaluate_ingestion(golden, results, [])["boilerplate_residual"] == 0.5


def test_table_cell_needs_row_and_value_on_one_line():
    golden = [
        _golden("a.pdf", fmt="pdf", cells=[["Screen", "Gold", "900"], ["Keys", "Gold", "70"]])
    ]
    results = [FileResult("a.pdf", "indexed", "Screen 450 900\nKeys\n70", ["c"])]
    assert evaluate_ingestion(golden, results, [])["table_cell_line_recall"] == 0.5


def test_evidence_intact_needs_one_chunk_to_hold_the_whole_quote():
    item = QaItem(
        id="q",
        input="?",
        tags=["fact", "split:test"],
        expected=Expected(
            answer="a",
            evidence=[Evidence(files=["a.txt"], quote="the excess is Rs. 500 for each claim")],
        ),
    )
    whole = FileResult("a.txt", "indexed", "", ["the excess is Rs. 500 for each claim"])
    cut = FileResult("a.txt", "indexed", "", ["the excess is Rs.", "500 for each claim"])
    golden = [_golden("a.txt")]
    assert evaluate_ingestion(golden, [whole], [item])["evidence_intact"] == 1.0
    assert evaluate_ingestion(golden, [cut], [item])["evidence_intact"] == 0.0
