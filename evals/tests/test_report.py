from evals.compare import compare_runs
from evals.report import render_report
from evals.tests.test_compare import _record, _rows


def _html(a_id="base", b_id="new", **ingestion):
    a, b = _record(a_id), _record(b_id, **ingestion)
    a["label"], b["label"] = "baseline", "<script>x</script>"
    a["config"] = {"chunk_size": 600}
    b["config"] = {"chunk_size": 800}
    return render_report(a, b, compare_runs(a, _rows([0.0] * 60), b, _rows([1.0] * 60)))


def test_report_is_one_self_contained_page():
    html = _html()
    assert html.startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html


def test_report_names_both_runs_and_shows_the_delta():
    html = _html()
    assert "base" in html and "new" in html
    assert "+1.000" in html


def test_report_escapes_untrusted_text():
    assert "<script>x</script>" not in _html()


def test_report_shows_only_config_that_differs():
    html = _html()
    assert "chunk_size" in html and "600" in html and "800" in html


def test_report_says_when_nothing_regressed():
    assert "No regressions" in _html()


def test_report_surfaces_a_regression():
    assert "text_recall moved" in _html(text_recall=0.1)
