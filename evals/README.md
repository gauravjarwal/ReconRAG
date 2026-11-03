# Evals

Everything here runs without an API key, except `make eval-full` and `make judge-calibrate`.

```
make eval             # corpus, ingestion, retrieval, refusal gate, report, regression gate
make report           # compare the baseline with the latest run
make label            # label 50 answers at http://127.0.0.1:8765
make eval-stub        # test the generation harness, free, nothing recorded
make eval-full MAX_USD=2         # live generation evals, hard spend cap
make judge-calibrate MAX_USD=1   # measure the judge on seeded errors
make check-evals      # lint, types, tests
```

| Path | Holds |
|---|---|
| `corpus/` | The fictional insurer and the messy file renderers |
| `golden/` | Frozen JSONL sets: ingestion truth, questions, judge items |
| `baselines/v1_frozen/` | v1's code, byte-identical to commit `f5cf1b3` |
| `harness/` | Metrics, matching, scoring, the LLM wrapper, the judge |
| `results.jsonl` | One summary line per run |
| `runs/` | Per-item scores, used for paired comparisons |
| `baseline.json` | The run CI compares against. Change it only after a review |

**Adding a pipeline:** implement `Pipeline` in `harness/schema.py` and register it in `run.py`.

Results and limits: `docs/EVAL_REPORT.md`.
