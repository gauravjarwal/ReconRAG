# ADR 0001: Postgres holds chunks, vectors, search, jobs and the PHI vault

**Status:** Accepted. **Date:** 2026-11-01.

## Context

v1 kept vectors in ChromaDB, rebuilt BM25 in memory, and kept the PHI vault in SQLite. That is three
stores with no shared transaction, and two of them can't be shared by a second replica.

## Decision

Use one Postgres with pgvector and pg_trgm. Vectors are `halfvec(384)` in an HNSW index, partial on
`WHERE active`. Redis is optional and holds only rate-limit counters and cached query embeddings.

## Alternatives

| Option | Why not |
|---|---|
| Qdrant or OpenSearch beside Postgres | Publishing a document would span two systems, which needs a reconciliation job. Better search features aren't worth that at this size |
| Keep ChromaDB | No transaction with document metadata, and no lexical search |

## Consequences

- Publishing a version, its chunks and its job status commit together, so readers never see half a document.
- Vector, lexical and queue load share one database, so a heavy ingest can slow queries.
- HNSW needs RAM, and parallel index builds need `/dev/shm`. Docker's default of 64MB fails them, so
  both compose files set `shm_size`.
- `plan_cache_mode=force_custom_plan` is set on every connection, because one cached plan can't suit
  both a large and a small collection.
- The way out is to move dense search behind `retrieval/search.py`, the only module that queries it.
