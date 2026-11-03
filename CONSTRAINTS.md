# Constraints

Last reviewed: 2026-11-02. Covers `evals/`, `src/`, `tests/`, `migrations/` and `frontend/`. `backend/` is v1 and frozen.

## Floor

- No new suppressions (`# noqa`, `# type: ignore`) without a reason beside them.
- No skipped or deleted tests.
- No golden item edited to make a run pass. Changes go in a new `_v2` file.
- No secrets in source, and no reading or printing `.env`.
- This file isn't weakened to make a change pass.

## Enforced

| Dimension | Rule | Checked by | Runs at |
|---|---|---|---|
| Lint and format | Zero errors | `uv run ruff check .`, `uv run ruff format --check .` | `make check`, CI |
| Types | Zero errors | `uv run mypy` | `make check`, CI |
| Unit and eval tests | All pass | `uv run pytest -q` | `make check`, CI |
| Integration tests | All pass, on real Postgres, Redis and processes | `uv run pytest tests/integration -q` | `make test-int`, CI |
| Image | Builds | `docker build .` | CI |
| Eval regression | No significant fall in recall@5, all_found@10 or MRR | `make eval-ci` | CI |
| Ingestion regression | No fall past 0.005, and no crashes | `make eval-ci` | CI |
| Dependencies (external) | No known vulnerabilities | `uvx pip-audit`, see the workflow | CI |
| Secrets (external) | No findings | `gitleaks` action | CI |

Why these numbers: 0.005 is about one sentence in the 232 checked. Retrieval uses a 95% paired
bootstrap, so noise can't fail a build and a real drop can't pass.

## Measured, not yet enforced

| Metric | Today | Direction |
|---|---|---|
| Line coverage of `evals/`, tests included | 90% | must not fall |
| recall@5, hybrid, test split | 0.500 | must not fall |
