# ReconRAG

A RAG service for messy real-world documents. It ingests PDFs, scans, Office files, HTML, email and
spreadsheets, answers with citations to the page and section, and refuses when the evidence is weak.
It can pseudonymise PHI before anything is embedded or sent to a model.

**Status:** v2 is a full rewrite and works end to end with local models. Its retrieval quality is
**unmeasured**, because the evals haven't been adapted to it yet, and no live model has been called.
Read the limits below before relying on it.

## Quick start

Needs Docker and [uv](https://docs.astral.sh/uv/).

```bash
make setup        # Installs, generates secrets, starts Postgres and Redis, migrates
make dev          # API on :8000, worker, UI on :8501
uv run reconrag ingest ./my-folder --collection docs --wait
```

Search and refusal work with no API key. Generated answers need `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`
in `.env`. Full stack: `docker compose up -d --scale api=2 --scale worker=2`. See
[docs/RUNBOOK.md](docs/RUNBOOK.md).

## How it works

```
Upload ─→ Postgres job ─→ Worker ─→ child process: sniff, parse, OCR, clean
                            │
                            ├─→ dedup, version, PHI tokens, chunk, embed
                            └─→ publish chunks and finish job in one transaction

Question ─→ PHI tokens ─→ dense + lexical + fuzzy ─→ weighted RRF ─→ cross-encoder
          ─→ gate (refuse?) ─→ context ─→ model with fallback ─→ citation and number checks
```

- **Postgres holds everything:** chunks, vectors, search, jobs and the PHI vault. Redis is optional.
- **Ingest survives failure:** a killed worker's job returns on lease expiry, and fencing stops a stale
  worker publishing. Bad files are quarantined with a reason.
- **Parsing is sandboxed** in a child process with a timeout.
- **Refusal works with or without a reranker.**
- **PHI collections stream no tokens,** because model output must be checked before anyone sees it.

Why it's built this way: [docs/adr/](docs/adr/).

## Commands

| Command | Does |
|---|---|
| `make check` | Lint, format, types, unit and eval tests |
| `make test-int` | Integration tests on real Postgres, Redis and processes |
| `make eval` | Free evals. They still measure v1 |
| `uv run reconrag worker` / `api` | Run one worker or API process |
| `uv run reconrag retry-job <id>` | Requeue a dead job |

## Measurements

All from 2026-11-02, on one Mac, in single runs. Details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Measure | Result |
|---|---|
| HNSW at 1M synthetic vectors, 1% filter | Recall@10 0.910, p95 14.9 ms |
| Service capacity, 2 replicas, up to reranking, no model | About 4 requests/s |
| Ingest, 216 messy files, 2 workers | 104.7s, 12 quarantined with reasons |

v1's baseline, recall@5 of 0.500 on the same corpus, is in [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md).
v2 has no comparable number yet.

## Limits

- Generated answers, streaming and hosted providers are untested against live models.
- Refusal thresholds are uncalibrated guesses.
- Original uploads are unencrypted on disk, including PHI. Use encrypted volumes.
- Every API key reaches every collection. Run one deployment per tenant.
- PyMuPDF is AGPL. See [ADR 0005](docs/adr/0005-pymupdf-agpl.md).
- PHI pseudonymisation is not HIPAA compliance by itself.

Defects found and how each was fixed: [docs/FINDINGS_REPORT.md](docs/FINDINGS_REPORT.md).
