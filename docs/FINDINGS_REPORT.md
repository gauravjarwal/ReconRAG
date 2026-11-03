# ReconRAG v2: findings report

Every defect found while building v2, up to 2026-11-02: the root cause, the fix, and how the fix was
checked. Numbers come from `docs/BENCHMARKS.md`. Nothing here is committed yet.

**How to read the evidence column.** "Red, then green" means a failing test was run before the fix.
"Regression test" means a test now covers it, but I didn't record a failing run first. Findings 1 to 12
and 23 come from runs on 2026-11-02, with their outputs to hand. Findings 13 to 22 were reconstructed
afterwards from session notes, so their root causes are shorter and less exact.

## Summary

| Severity | Count | Examples |
|---|---|---|
| Critical | 2 | API OOM-killed under load, PHI streaming guard untested |
| High | 6 | Benchmark reported false recall, internal errors leaked to clients |
| Medium | 11 | Rate limits shared a counter, model files unreadable in the image |
| Low | 4 | Excel titles read as tables |

Four are still open, listed at the end. **No finding here shows v2 retrieves better than v1. That needs an eval run on v2, which hasn't happened.**

## Found by running the real system

These all passed unit and integration tests first. Each needed real load, real containers or a real
benchmark to appear.

### 1. Critical: API container OOM-killed under load

- **Symptom:** at 40 concurrent clients, one API container exited 137 with `OOMKilled=true`. The other
  sat at 4.2GB. Throughput peaked at 2.8 requests/s.
- **Root cause:** every request ran its own cross-encoder inference, with no limit. Inference is
  CPU-bound, so ONNX threads fought over the cores, and memory grew with concurrent runs. The memory
  growth mechanism is my reading of ONNX's arena behaviour. I didn't profile it, but bounding
  concurrency did remove it.
- **Fix:** `CrossEncoderReranker` takes a semaphore, `RERANK_CONCURRENCY=2`, with fixed
  `RERANK_THREADS=2`. A request that can't get a slot in `RERANK_WAIT_S=5` falls back to fused order,
  flagged in `degraded` and counted in `reconrag_rerank_shed_total`. Compose sets `mem_limit: 2g` on the API.
- **Measured:** same corpus, same script, real image, before and after.

| | Before | After |
|---|---|---|
| Errors at 40 clients | Container killed | 0 |
| Memory per replica | 4.2GB on the survivor, not sampled at peak | About 1.1GB peak, sampled |
| Requests/s at 10 clients | 2.8 | 3.9 |
| Requests/s at 20 clients | 2.5 | 4.3 |
| p95 at 20 clients | 13,947 ms | 6,451 ms |

- **Regression test:** `tests/unit/test_rerank_limits.py`, red, then green. Load itself isn't in CI.
- **Cost:** under overload some queries skip reranking. 172 of 657 did across the run, mostly at 40 clients.

### 2. High: first benchmark reported perfect recall, falsely (cause 1)

- **Symptom:** recall@10 of 1.0 with a 1% filter and iterative scans off, and flat latency around 47 ms.
  HNSW filtering after the scan can't do that.
- **Root cause:** `SET LOCAL enable_indexscan = off` for the exact pass ran inside `conn.transaction()`
  on a non-autocommit connection. That's a savepoint, and `SET LOCAL` outlives a savepoint, so the
  "index" queries ran as exact scans.
- **Fix:** autocommit after loading, so each transaction is real, plus a check that `EXPLAIN` names the index.
- **Measured:** the second run still showed 1.0, which led to finding 3.

### 3. High: benchmark still reported perfect recall (cause 2)

- **Symptom:** recall of 1.0 again, even with a plan check that passed.
- **Diagnosis:** a separate script on 200k vectors with `EXPLAIN ANALYZE` showed HNSW behaving correctly:
  4 of 10 rows with iterative scans off, 10 of 10 with them on, in about 2 ms.
- **Root cause:** exact and approximate passes used the same SQL text. psycopg prepares a statement
  after 5 runs, so the plan cached in the exact pass answered the approximate pass. `EXPLAIN` didn't
  share that prepared statement, which is why the check passed.
- **Fix:** `prepare_threshold=None` in the benchmark.
- **Measured:** recall became 0.758 with no filter, 0.122 at a 1% filter, and 0.910 with `relaxed_order`,
  at 4.37, 3.98 and 9.66 ms p50. Those differences are what the theory predicts for the filtered rows.
- **Lesson:** a perfect score from an approximate index should be treated as a bug until shown otherwise.

### 4. Medium: search could reuse one plan across collections

- **Symptom:** none seen in the product. Finding 3 showed the mechanism was real.
- **Root cause:** the same prepare path applies to `dense_search`. A cached generic plan can't suit both
  a large collection, which needs HNSW, and a small one, which needs an exact sort.
- **Fix:** `plan_cache_mode=force_custom_plan` on every pooled connection.
- **Measured:** `test_pooled_connections_never_share_one_plan_across_collections` proves the setting is
  on. The latency or recall effect was **not measured**. This is a preventive change.

### 5. Medium: 1M benchmark crashed on shared memory

- **Symptom:** `DiskFull: could not resize shared memory segment ... to 2144456160 bytes`.
- **Root cause:** parallel HNSW builds allocate from `/dev/shm`, and Docker's default is 64MB.
- **Fix:** `shm_size: 3gb` in both compose files and in CI.
- **Measured:** `/dev/shm` showed 3.0G, and the build finished in 88.2s.

### 6. Medium: model files unreadable in the image

- **Symptom:** `Permission denied` on the model cache at startup, logged as a "corrupted" cache.
- **Root cause:** `huggingface_hub` writes tree files as `0600`, as root, and the image runs as uid 10001.
- **Fix:** `chmod -R a+rX /app/models` in the build, and `HF_HUB_OFFLINE=1` at runtime.
- **Measured:** both models loaded in a container run with `--network none`.

### 7. Medium: load test script had never worked

- **Symptom:** 3,000 requests/s, all errors, sub-millisecond latency.
- **Root cause:** `client.post("")` with a base URL adds a trailing slash, so FastAPI answered 307.
  The script hadn't been run before. It also ignored error bodies and repeated 8 off-topic questions.
- **Fix:** post to the full URL, count refusals, and load the 334 golden questions.
- **Measured:** the query route's request counter rose, and errors fell to 0.

## Found by mutation testing

I broke 8 safety conditions on purpose, and ran the tests after each. Six failed as they should. These
two passed, so nothing was protecting them.

### 8. Critical: nothing proved PHI collections send no tokens

- **Root cause:** a test gap. The code was right, but removing `if not protected:` went unnoticed, and
  that line stops unvalidated model output reaching clients.
- **Fix:** `test_a_phi_collection_never_streams_unvalidated_tokens`.
- **Measured:** rerun of the mutation now fails the suite.

### 9. High: nothing proved shared files survive a collection delete

- **Root cause:** a test gap. Removing the `NOT EXISTS` check would delete a file another collection needs.
- **Fix:** `test_deleting_a_collection_keeps_a_file_another_collection_uses`.
- **Measured:** rerun of the mutation now fails the suite.

## Found by review

### 10. Medium: query and upload limits shared one counter

- **Root cause:** the Redis key was `key_id:limit`, and both limits default to 30, so uploads spent the
  query budget.
- **Fix:** key by scope, not by number.
- **Measured:** `test_limits_with_the_same_number_count_separately`, red, then green.

### 11. Medium: a request couldn't be followed into its job

- **Root cause:** the request id stopped at the API. Worker logs had none, and nothing recorded whether
  a job came from the API or the CLI. Found using the observability checklist.
- **Fix:** migration 0002 adds `request_id` and `entry_point` to jobs, and the worker logs under them.
  `reconrag ingest` prints a run id.
- **Measured:** `test_worker_logs_use_the_id_of_the_originating_request`, red, then green. The migration
  ran up, down and up in the real image.

### 12. Medium: a stuck queue was invisible

- **Root cause:** only depth was measured, and depth alone can't separate busy from stuck. Connection
  pool saturation wasn't measured, and worker metrics were off in compose.
- **Fix:** `reconrag_queue_oldest_age_seconds`, which ignores jobs in backoff, three pool gauges, and
  `WORKER_METRICS_PORT=9100`. `worker_id` now stays on finished jobs, so you can see who ran them.
- **Measured:** two queue tests, red, then green, and a metrics assertion added with the fix. The
  two-replica test showed both workers doing work, in 8 of 8 repeat runs. Alert rules were **not** fired.

### 13. High: internal errors leaked into the job record

- **Root cause:** the raw exception text went into `error_detail`, which clients can read.
- **Fix:** store a fixed message, and keep the detail in redacted logs.
- **Measured:** `test_internal_errors_stay_out_of_the_job_record`, red, then green.

### 14. High: refused re-identification returned 403 too late on the stream

- **Root cause:** the check ran inside the generator, after streaming headers had gone.
- **Fix:** move it to a route dependency.
- **Measured:** `test_stream_refuses_reidentify_before_any_event`, red, then green.

### 15. High: connection pool exhaustion returned 500

- **Root cause:** `PoolTimeout` and `OperationalError` had no handler.
- **Fix:** return 503 `overloaded` with `Retry-After: 1`.
- **Measured:** `test_exhausted_database_pool_is_a_503_with_retry_after`, red, then green.

### 16. Medium: eval gate could compare different golden sets

- **Root cause:** the gate didn't check that both runs used the same data, so a regression could pass.
  The corpus also wasn't always rebuilt.
- **Fix:** the gate refuses differing golden data, and the corpus is always rebuilt.
- **Measured:** `test_gate_refuses_to_compare_runs_on_different_golden_data` and
  `test_rebuild_is_byte_identical`, and 12 of 12 deliberate breaks caught in the evals slice.

## Found while building parsers and validation

| # | Severity | Defect | Root cause | Fix | Evidence |
|---|---|---|---|---|---|
| 17 | Medium | Margin removal deleted body text containing digits | Digits were normalised, so similar lines looked repeated | Only lines of 10 words or fewer, once per page | `test_repeated_header_and_footer_go_but_body_stays` |
| 18 | Medium | A renamed upload re-embedded everything | The title came from the filename, and the title is part of the context | Fall back to the document key | `test_new_version_replaces_old_and_reembeds_only_changes` covers reuse; I didn't find a test named for renames |
| 19 | Medium | Validator split decimals like 3.5 into two numbers | The sentence split treated the dot as a break | Split on a sentence-break pattern | `test_percent_and_decimal_stay_separate_numbers` |
| 20 | Low | Excel title rows read as tables | Width check too loose | Require over half the cells filled | `test_xlsx_title_rows_regions_and_empty_columns` |
| 21 | Low | Sentence split broke on abbreviations | Used the match start | Use `match.end()` | `test_sentence_split_respects_abbreviations` |
| 22 | Low | Quality score penalised "A" and "I" | Single letters counted as junk | Allow a and i | Regression coverage not confirmed by name |
| 23 | Low | Two-replica test failed on repeat runs | Rate-limit counters outlived the test in Redis | Unique API key per run, and avoid the minute boundary | Passed 8 of 8 runs, a defect in the test only |

## Still open

| Issue | Risk | Why it's open |
|---|---|---|
| Raw PHI files sit unencrypted on the `blobs` volume, and filenames are stored as uploaded | High for real patient data | Needs a design decision. The runbook requires encrypted volumes for now |
| Every API key reaches every collection | High if shared across tenants | Single-tenant by design, documented in the runbook |
| Unfiltered recall is 0.758 on synthetic data | Unknown | `ef_search` wasn't swept, and real data is needed |
| Gate thresholds are guesses | High for answer quality | Needs evals adapted to v2 |

## Not verified

- No live model call: generation, streaming, Gemini embeddings and Anthropic are all untested live.
- Retrieval quality of v2 is unmeasured. No improvement over v1 can be claimed.
- The load test stops before generation, so long model streams, the real limit in ADR 0002, weren't tested.
- CI has never run on GitHub. The compose file was only validated with `.env` absent, and the full
  `docker compose up` wasn't run: containers were started individually.
- Alert rules were never fired. The UI wasn't re-checked in a browser after its last two fixes.
