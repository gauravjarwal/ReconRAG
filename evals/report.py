"""Writes one self-contained HTML page comparing two runs.

Usage: python -m evals.report [--a <run_id>] [--b <run_id>]
Defaults: the baseline against the latest run. Output: evals/reports/compare.html
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

from evals.compare import (
    EVALS_DIR,
    baseline_run_id,
    gate_failures,
    latest_run_id,
    load_comparison,
)
from evals.harness.results import load_results

OUTPUT = EVALS_DIR / "reports" / "compare.html"
STYLE = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--good:#17693a;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#a0a0a0;--line:#383838;--good:#6fcf97;--bad:#ff8a80}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.5rem;margin-bottom:.2rem}h2{font-size:1.1rem;margin-top:2rem}
.meta{color:var(--muted);font-size:.9rem}
table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:.35rem .6rem;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
.up{color:var(--good)}.down{color:var(--bad)}.flat{color:var(--muted)}
.ok{color:var(--good);font-weight:600}.fail{color:var(--bad);font-weight:600}
code{font-size:.85em}details{margin-top:.5rem}
"""


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _direction(delta: float, significant: bool) -> str:
    if not significant:
        return "flat"
    return "up" if delta > 0 else "down"


def _rows(rows: list[list[str]]) -> str:
    return "".join("<tr>" + "".join(row) + "</tr>" for row in rows)


def _cell(text: object, css: str = "") -> str:
    attribute = f' class="{css}"' if css else ""
    return f"<td{attribute}>{escape(str(text))}</td>"


def _config_changes(a: dict[str, Any], b: dict[str, Any]) -> list[list[str]]:
    a_config, b_config = a.get("config", {}), b.get("config", {})
    return [
        [_cell(key), _cell(a_config.get(key, "n/a")), _cell(b_config.get(key, "n/a"))]
        for key in sorted(set(a_config) | set(b_config))
        if a_config.get(key) != b_config.get(key)
    ]


def render_report(a: dict[str, Any], b: dict[str, Any], comparison: dict[str, Any]) -> str:
    failures = gate_failures(comparison)
    if failures:
        verdict = (
            "<p class='fail'>Regressions</p><ul>"
            + "".join(f"<li>{escape(failure)}</li>" for failure in failures)
            + "</ul>"
        )
    else:
        verdict = "<p class='ok'>No regressions past measured noise.</p>"

    retrieval = [
        [
            _cell(name),
            _cell(f"{r['delta']:+.3f}", _direction(r["delta"], r["significant"])),
            _cell(f"{r['low']:+.3f} to {r['high']:+.3f}"),
            _cell("yes" if r["significant"] else "no"),
        ]
        for name, r in comparison["retrieval"].items()
    ]
    categories = []
    for name, row in comparison["by_category"].items():
        delta = row["b"] - row["a"]
        css = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        categories.append(
            [
                _cell(name),
                _cell(row["items"]),
                _cell(_number(row["a"])),
                _cell(_number(row["b"])),
                _cell(f"{delta:+.3f}", css),
            ]
        )
    ingestion = [
        [_cell(name), _cell(_number(row["a"])), _cell(_number(row["b"]))]
        for name, row in comparison["ingestion"].items()
    ]
    changes = _config_changes(a, b)
    config = (
        "<table><tr><th>Setting</th><th>A</th><th>B</th></tr>" + _rows(changes) + "</table>"
        if changes
        else "<p class='meta'>The two runs use the same config.</p>"
    )

    def id_list(title: str, ids: list[str]) -> str:
        items = "".join(f"<li><code>{escape(i)}</code></li>" for i in ids)
        return f"<details><summary>{title} ({len(ids)})</summary><ul>{items}</ul></details>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ReconRAG eval comparison</title>
<style>{STYLE}</style></head><body>
<h1>ReconRAG eval comparison</h1>
<p class="meta">A: <code>{escape(a["run_id"])}</code> ({escape(str(a.get("label", "")))})<br>
B: <code>{escape(b["run_id"])}</code> ({escape(str(b.get("label", "")))})<br>
Deltas are B minus A on {comparison["paired_items"]} answerable test-split items.
Intervals are 95% paired bootstrap.</p>
{verdict}
<h2>Retrieval</h2>
<table><tr><th>Metric</th><th>Delta</th><th>95% CI</th><th>Significant</th></tr>{_rows(retrieval)}</table>
<h2>recall@5 by category</h2>
<table><tr><th>Category</th><th>Items</th><th>A</th><th>B</th><th>Delta</th></tr>{_rows(categories)}</table>
<h2>Ingestion</h2>
<table><tr><th>Metric</th><th>A</th><th>B</th></tr>{_rows(ingestion)}</table>
<h2>Config differences</h2>
{config}
<h2>Items that moved</h2>
{id_list("Got worse", comparison["worse"])}
{id_list("Got better", comparison["better"])}
</body></html>
"""


def write_report(a_id: str, b_id: str, output: Path = OUTPUT, evals_dir: Path = EVALS_DIR) -> Path:
    records = {r["run_id"]: r for r in load_results(evals_dir / "results.jsonl")}
    html = render_report(records[a_id], records[b_id], load_comparison(a_id, b_id, evals_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--b-pipeline")
    args = parser.parse_args()
    a = args.a or baseline_run_id()
    b = args.b or latest_run_id(pipeline=args.b_pipeline)
    print(f"Wrote {write_report(a, b)}")


if __name__ == "__main__":
    main()
