import json

from evals.baselines.v1 import V1Pipeline
from evals.harness.results import append_result, code_version, load_run, save_run
from evals.run import run_eval
from evals.tests.test_v1_baseline import FakeEmbedder


def test_append_result_adds_one_json_line(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result(path, {"run_id": "a"})
    append_result(path, {"run_id": "b"})
    assert [json.loads(line)["run_id"] for line in path.read_text().splitlines()] == ["a", "b"]


def test_a_saved_run_loads_back(tmp_path):
    save_run(tmp_path, "abc", [{"id": "q1", "recall@5": 1.0}])
    assert load_run(tmp_path, "abc") == [{"id": "q1", "recall@5": 1.0}]


def test_code_version_reports_sha_and_dirty_flag():
    version = code_version()
    assert set(version) == {"git_sha", "git_dirty"}


def test_a_full_run_records_what_the_repo_rules_require(tmp_path):
    record = run_eval(V1Pipeline("hybrid", FakeEmbedder()), label="test", out_dir=tmp_path)
    assert {"timestamp", "git_sha", "config", "metrics", "cost_usd", "latency"} <= set(record)
    assert record["metrics"]["ingestion"]["files"] == 216
    assert record["metrics"]["retrieval"]["test"]["items"] > 100
    assert record["metrics"]["refusal_gate"]["threshold"] is not None
    assert len(load_run(tmp_path / "runs", record["run_id"])) == 334
    assert (tmp_path / "results.jsonl").exists()


def test_a_run_is_repeatable_when_nothing_is_random(tmp_path):
    first = run_eval(V1Pipeline("hybrid", FakeEmbedder()), label="a", out_dir=tmp_path)
    second = run_eval(V1Pipeline("hybrid", FakeEmbedder()), label="b", out_dir=tmp_path)
    assert (
        first["metrics"]["retrieval"]["test"]["recall@5"]
        == second["metrics"]["retrieval"]["test"]["recall@5"]
    )
