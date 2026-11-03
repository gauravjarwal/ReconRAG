# Tasks: ReconRAG v2 refactor

Verify with `make check` (and `make test-int` where a task touches Postgres or Redis).
The evals slice is done and lives in `evals/`.

## Phase 1: Platform

- [x] **1. Package and config.** `src/reconrag`, `pyproject.toml`, typed settings, compose services.
- [x] **2. Schema.** Alembic migration, pool, repositories.
- [x] **3. Observability.** JSON logs with PHI redaction, metrics, request context.
- [x] **4. LLM layer.** Pricing, retries, circuit breaker, Gemini, Anthropic, embedders.

## Phase 2: Ingestion

- [x] **5. Sniffing and safety.** True file type, size and zip-bomb limits.
- [x] **6. Parsers.** PDF with per-page OCR, DOCX, HTML, EML, CSV, XLSX, TXT, MD, images.
- [x] **7. Cleaning.** Encoding, control characters, headers and footers, hyphenation, quality score.
- [x] **8. Dedup and versioning.** Hashes, SimHash, logical document keys, editions.
- [x] **9. Chunking.** Structure-aware, token-sized, tables whole with headers.
- [x] **10. Queue and worker.** Claim, lease, heartbeat, retry, dead letter, atomic publish.

### Checkpoint: ingest works

- [x] A killed worker loses no job and creates no duplicate chunks.
- [x] Re-uploading an unchanged file does no embedding work.

## Phase 3: Retrieval

- [x] **11. Search.** Dense, lexical, fuzzy, weighted RRF, duplicate suppression.
- [x] **12. Rerank, gate, context.** Reranker, refusal with and without it, expansion, token budget.

## Phase 4: Generation

- [x] **13. Answering.** Versioned prompts, fallback, streaming, TTFT.
- [x] **14. Output checks.** Citations, numeric grounding, PII, length.

## Phase 5: API and operations

- [x] **15. PHI port.** Detection, pseudonymisation, Postgres vault.
- [x] **16. API.** Auth, rate limits, async upload, jobs, query, SSE, citations, probes, metrics.
- [x] **17. CLI and UI.** `reconrag ingest <folder>`, generated secrets, live job status, rich citations.
- [x] **18. Packaging.** Dockerfile, compose, CI, runbook, ADRs, load test.

### Checkpoint: runs as a service

- [x] Two API replicas and two workers run against one Postgres.

## Phase 6: Finish

- [x] **19. Review.** Five-axis review, mutation checks, dependency audit. Findings: `docs/FINDINGS_REPORT.md`.
- [x] **20. Defence.** `docs/V2_DEFENSE.md` and learning records.

## Not done

- [ ] Adapt the evals to v2, then compare with v1. Nothing can be called an improvement before this.
- [ ] Live model run. Needs a key and an agreed budget.
- [ ] Calibrate the gate thresholds on the dev split.
- [ ] Load test with live model streams.
- [ ] Decide how PHI files are stored at rest.
- [ ] First CI run on GitHub. Full `docker compose up` run.
- [ ] Decide on deleting `backend/` and `frontend/components/uploader.py`.
