# ReconRAG

Read `CONSTRAINTS.md` before writing code. Do not weaken it to make a change pass.

- Plan and status: `SPEC.md`, `tasks/todo.md`. Decisions: `docs/adr/`. Operations: `docs/RUNBOOK.md`.
- Baseline numbers: `docs/EVAL_REPORT.md`. They describe v1. No eval has run on v2 yet.
- v2 lives in `src/reconrag/`, with tests in `tests/`. `frontend/` is the v2 UI.
- `backend/` is v1, kept until the owner decides to delete it. Its tests are in `backend/tests/`.
- Checks: `make check` (lint, format, types, unit and eval tests). `make test-int` needs `make services`.
- Evals: `make eval` (free). They still run the v1 pipeline. `evals/baselines/v1_frozen/` must never be edited.
- Keep the bench honest: `scripts/bench_index.py` runs with prepared statements off, because a plan
  cached for the exact pass would answer the approximate pass too.
