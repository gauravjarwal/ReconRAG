# ADR 0004: Postgres full-text search stands in for BM25

**Status:** Accepted, measurement pending. **Date:** 2026-11-01.

## Context

v1 rebuilt a BM25 index in memory, per collection. That doesn't scale and doesn't work across replicas.

## Decision

Use a generated `tsvector` with both `english` and `simple` configurations, a GIN index, OR queries and
`ts_rank` with length normalisation. Identifiers like `CF-7731` skip stemming. A trigram search on
identifiers only catches OCR slips.

## Alternatives

| Option | Why not |
|---|---|
| In-memory BM25 | v1's problem |
| OpenSearch | New infrastructure |
| A BM25 Postgres extension | Not in the stock pgvector image, so it means a custom image |

## Consequences

- `ts_rank` isn't BM25: it has no inverse document frequency, so rare terms get no extra weight.
- The effect on retrieval quality is **unmeasured**. The evals have to be adapted to v2 first.
- RRF uses ranks only, so the score scale doesn't matter, but the ordering does.
