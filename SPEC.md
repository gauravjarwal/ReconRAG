# Spec: ReconRAG v2

Status: **v2 refactor in progress (2026-11-01).** `eval-golden` and `eval-harness` are done, with the v1
baseline in `docs/EVAL_REPORT.md`. The other five modules are being built in `src/reconrag/`, designed
independently of the eval harness. The evals get adapted to v2 afterwards, not the other way round.
Design decisions that replace parts of this spec are in `tasks/plan.md`.

## 1. Objective

Turn ReconRAG from a single-process demo into a service that can ingest messy real-world documents,
retrieve from them reliably, and prove it with numbers.

**User:** one engineer running it locally and defending it in interviews. The design must also hold up
as a whiteboard answer to "take this to production".

**Done means measured.** v2 is done when one command runs the evals, results are saved, and v2 is
compared with the v1 pipeline on the same data. A working demo doesn't count.

## 2. Capability map

| Module id | Responsibility | Depends on |
|---|---|---|
| `eval-golden` | Messy corpus generator, ingestion truth, golden question sets | none |
| `eval-harness` | Metrics, runner, `results.jsonl`, v1 baseline, significance tests, CI gate | `eval-golden` |
| `platform` | `src/` package, typed config, Postgres schema, migrations, LLM wrapper, logs and metrics | none |
| `ingestion` | Parsers, OCR, cleaning, dedup, versioning, structure-aware chunking, job queue, workers | `platform` |
| `retrieval` | pgvector + Postgres FTS, fusion, reranking, refusal gate, context assembly | `platform`, `ingestion` |
| `generation` | Prompts, streaming, fallback, circuit breaker, output validation | `retrieval` |
| `api-ops` | FastAPI v2, auth, PHI port, Streamlit, Docker, CI, runbook, load test | all |

**Build order:** `eval-golden` → `eval-harness` (records the v1 baseline) → `platform` → `ingestion` →
`retrieval` → `generation` → `api-ops`.

Evals go first on purpose. If the work stops early, the most valuable part already exists.

## 3. Assumptions

Correct any of these now:

1. **Postgres replaces Chroma.** pgvector + full-text search, as `learning/PLAN.md` already decided. The
   PHI vault moves there too, which removes the SQLite single-replica limit.
2. **Rewrite in place, on branch `v2-production`.** No commits until asked.
3. **Local models for offline evals.** Free, deterministic CI retrieval runs need no API key. Gemini
   stays the production embedding option.
4. **The corpus is synthetic but deliberately messy,** using a clearly fictional insurer, so ground
   truth is known by construction. It stays small enough to regenerate deterministically.
5. **Python 3.12 and `uv`,** as a standalone project, not a member of the `ai-lab` workspace.
6. **No LangChain, LlamaIndex or LangGraph.** Parsing libraries are fine; the RAG logic stays ours.
7. **Single-node Docker Compose is the deployment target.** Kubernetes is out of scope, but API and
   workers scale as separate stateless processes.

## 4. Eval design (written before the features)

### Golden sets (versioned JSONL, never edited to make a run pass)

| File | Size | Contents |
|---|---|---|
| `evals/golden/ingest_v1.jsonl` | every corpus file | Expected outcome (indexed, quarantined, duplicate, superseded), text that must survive, boilerplate that must not, table cells, language |
| `evals/golden/qa_v1.jsonl` | ≥ 150 | Question, evidence document and quote, key facts, category, dev/test split |
| `evals/golden/judge_v1.jsonl` | ≥ 100 + 50 | Answers with seeded errors (labels known by construction), plus 50 for you to label |

Minimum question coverage: ≥ 20 must-refuse, ≥ 20 multi-hop, ≥ 15 table, ≥ 15 OCR/scanned, ≥ 10
version-sensitive ("current" vs superseded), ≥ 10 adversarial, ≥ 10 exact identifier, ≥ 10 paraphrase
with no shared keywords.

Tuning uses the **dev split only**. The **test split** is reported, never tuned on.

### A metric per stage

| Stage | Metrics |
|---|---|
| Parsing | File outcome accuracy, text recall against truth, table-cell recall, OCR character accuracy |
| Cleaning | Boilerplate removed, dehyphenation and encoding repair accuracy |
| Dedup/versioning | Duplicate precision and recall, stale-version leakage rate |
| Chunking | Evidence intact in one chunk (%), table split rate |
| Retrieval | recall@k, MRR, nDCG, per category; latency p50/p95 |
| Refusal gate | Precision and recall of refusing, threshold chosen on dev, reported on test |
| Generation | Key-fact correctness, faithfulness, citation precision, injection resistance |
| System | TTFT, p50/p95, cost per query, ingestion throughput |

### Rules

- One command: `make eval`. Each run appends config, git sha, metrics, cost and latency to
  `evals/results.jsonl`.
- **Baseline:** v1's parser, chunker and BM25 + vector + RRF logic run on the same corpus with the same
  embedding model, so the comparison is fair.
- **Noise:** paired bootstrap 95% CI on every comparison. LLM stages run 3 times. A change only
  "improves" something if the interval excludes zero.
- **Judge trust:** agreement is measured on seeded-error answers now, and on your 50 labels when you
  finish them. Until then it is reported as "not validated against human labels".
- **CI gate:** every PR runs ingestion and retrieval evals with local models and fails on regression.
  Generation evals run on demand, because they cost money.

**A limit to state in interviews:** the synthetic labels were generated by code I wrote, so they prove
the pipeline against known messiness, not against your users' data. FinanceBench (open question 2) is
the outside check.

## 5. Design summary

### Ingestion

`upload → blob store → job row → worker → parse → clean → [PHI] → dedup → chunk → embed → publish`

- **Async:** upload returns `202` with a job id. Workers claim jobs with Postgres
  `FOR UPDATE SKIP LOCKED` and a lease, so a crashed worker's job is retried. Failures go to a
  dead-letter state with a reason.
- **Formats:** PDF (native, scanned, multi-column, tables), DOCX, HTML, EML, CSV, XLSX, TXT, MD, images.
- **Mess handled:** encodings and mojibake, control and zero-width characters, repeated headers and
  footers, broken hyphenation, skewed scans, corrupt, encrypted, empty or disguised files, zip bombs.
- **Per-page routing:** OCR runs only on pages with no usable text layer. A quality score is stored, and
  low-quality documents are flagged rather than silently indexed.
- **Dedup:** exact (content hash) and near-duplicate (SimHash). Same logical document with new content
  becomes a new version, and the old one stops being retrievable.
- **Chunking:** two switchable strategies, recursive (v1) and structure-aware. Structure-aware keeps
  section paths and page numbers, keeps tables whole with headers repeated, and sizes by tokens.

### Retrieval

- **Hybrid:** HNSW over `halfvec` with pgvector iterative scans (so collection filters don't starve
  results), plus Postgres FTS using OR queries, an exact-token index for identifiers, and `pg_trgm` for
  OCR typos.
- **Then:** weighted RRF → near-duplicate suppression → optional cross-encoder or LLM rerank → refusal
  gate → adjacent-chunk expansion → token-budgeted context.
- **Refusal works with and without a reranker.** Thresholds come from dev-split calibration, replacing
  the guessed 4.0.
- **Off until measured:** multi-query and HyDE ship behind flags. They stay off unless the evals justify
  their cost.

### Generation

- Async providers with explicit timeouts, retries on retryable errors only, and a circuit breaker per
  provider. Fallback no longer triggers on every exception.
- SSE streaming with TTFT recorded.
- Output validation adds a numeric grounding check: every number in an answer must appear in its cited
  source. That closes the "would it catch a wrong number?" gap for numbers, not for all claims.

### Operations

Redis-backed rate limits and embedding cache (degrades safely if Redis is down), Prometheus `/metrics`,
structured JSON logs, per-stage tracing, health and readiness probes, graceful shutdown, Alembic
migrations, multi-stage Dockerfile, GitHub Actions, runbook, ADRs and a load test.

**Scale claim:** an index benchmark at 1M synthetic vectors, plus a load test. No scale number will be
stated unless it was measured here on your laptop's Docker allocation (10 CPUs, about 8 GB).

### Likely stack (each version checked with context7 before use)

`psycopg` 3 + pool, `pgvector`, `alembic`, `fastapi`, `pymupdf`, `rapidocr-onnxruntime`,
`beautifulsoup4`, `ftfy`, `openpyxl`, `python-docx`, `fastembed`, `google-genai`, `anthropic`,
`redis`, `prometheus-client`, `reportlab` (corpus only).

**Licence flag:** PyMuPDF is AGPL. That is fine for this public repo, but a company would need a
commercial licence. It sits behind a parser interface and the ADR will say so.

## 6. Project structure

```
src/reconrag/{config,domain,llm,storage,ingestion,retrieval,generation,security,api,worker,observability}
migrations/      Alembic
evals/           corpus/ golden/ harness + results.jsonl + reports/
tests/           unit/ integration/
frontend/        Streamlit, updated for async jobs and streaming
docs/            adr/ RUNBOOK.md EVAL_REPORT.md
```

## 7. Commands (to exist when done)

```
make setup      # uv sync, create .env with generated secrets, start services, migrate
make dev        # API, worker and UI
make test       # unit tests, no network
make test-int   # integration tests against real Postgres and Redis
make eval       # corpus, ingestion, retrieval evals; appends to results.jsonl
make eval-full  # adds generation evals, needs an API key, prints estimated cost first
make check      # ruff, format check, mypy, unit tests
```

## 8. Testing and code style

- `pytest`, LLMs mocked in unit tests, real Postgres in integration tests, real model calls only in
  `evals/`. Bug fixes start with a failing test.
- Type hints everywhere, Pydantic models at every LLM and API boundary, all LLM calls through one wrapper
  logging model, tokens, latency and cost, prompts as named versioned constants.

## 9. Quality bar (`CONSTRAINTS.md`)

Defaults, unless you object: measure and hold rather than invent targets; block on the floor; 90-second
limit on the task-end check; ≥ 80% coverage of changed lines; `pip-audit` and `gitleaks` in CI as the
external checks; eval metrics may not fall past measured noise.

## 10. Boundaries

- **Always:** eval before feature, test before fix, cite docs for new libraries, redact logs.
- **Ask first:** paid eval runs over the agreed budget, new infrastructure beyond Postgres and Redis,
  anything pushed to GitHub.
- **Never:** commit secrets, edit a golden item to pass, state an unmeasured number, weaken a check to go
  green, read or print `.env` contents.

## 11. Success criteria

1. `make eval` runs from a clean checkout and writes results with no API key.
2. The corpus has ≥ 200 files across at least 9 formats. No bad file crashes ingestion.
3. Golden sets meet the coverage minimums in section 4.
4. v2 beats v1 on test-split recall@5 overall with a 95% CI excluding zero, and per-category results are
   published, including any category that got worse.
5. The refusal threshold is calibrated on dev, with precision and recall reported on test.
6. A killed worker loses no job and creates no duplicate chunks (integration test).
7. Re-uploading an unchanged file does no embedding work (integration test).
8. Two API replicas and two workers run against one Postgres (integration test).
9. Each open item in `learning/reconrag-review.md` is fixed or explicitly left open with a reason.
10. `docs/EVAL_REPORT.md` holds only measured numbers, with before/after tables and a "what I'd ship"
    decision.

## 12. Risks

- **Defensibility:** `PLAN.md` wanted you to write the chunker, fusion, refusal logic and prompts
  yourself. This rewrite means more code you didn't write, right before a project deep dive. Mitigation:
  a quiz per module and a line-by-line defence guide. You still have to do the reading.
- **Synthetic data bias:** see the limit in section 4.
- **Postgres FTS is not true BM25:** no global IDF. Measured against v1's BM25, and recorded in an ADR.
- **OCR accuracy** depends on the ONNX model on noisy scans. Measured, not assumed.
- **Laptop limits** cap the scale evidence. The benchmark is a data point, not a production guarantee.

## 13. Open questions

1. **Generation provider:** unanswered. Only the Gemini key works today, so Gemini is the only live
   provider. Claude Haiku 5.5 needs a key before it can be measured.
2. **Real-world benchmark:** unanswered, so FinanceBench is not included. Its licence is unverified.
3. **Eval budget:** not agreed, so no paid model calls have been made. `make eval-full` exists but
   hasn't run against a live model.

### Agreed friction fixes

In this slice:

- A small labelling page for your 50 judge items, instead of editing JSONL by hand.
- Eval report as one generated HTML page comparing any two runs.

Agreed for later modules:

- Generate all secrets in `make setup`, so you never hand-create keys.
- `reconrag ingest <folder>` CLI for bulk loads instead of 20 files per upload.
- Live ingest progress in the UI: per-file status and why a file was quarantined.
- Citations show page and section, and open the source passage.

## 14. addy-* skills

This table was the plan. Actual use by 2026-11-02: every row was invoked except `doubt-driven`, which
was not run. Observability and code review on v2 ran late, after the code was written.

| Skill | Used for |
|---|---|
| `spec-driven`, `planning-and-task-breakdown` | This spec, then `tasks/plan.md` |
| `constraint-driven` | `CONSTRAINTS.md` |
| `source-driven` | context7 checks for every new library |
| `incremental-implementation`, `test-driven` | Thin slices, tests first |
| `api-and-interface-design` | Async jobs, SSE, provider interfaces |
| `security-and-hardening` | Upload handling, zip bombs, injection, PHI |
| `observability-and-instrumentation` | Logs, metrics, tracing |
| `ci-cd-and-automation` | Quality gates and eval gate |
| `documentation-and-adrs` | ADRs, runbook |
| `doubt-driven` | Adversarial check on schema, queue and refusal gate |
| `code-review-and-quality`, `shipping-and-launch` | Final review and readiness check |

Not used: `git-workflow` (no commits until asked), `interview-me` (this spec covers it),
`context-engineering` (no harness change needed).
