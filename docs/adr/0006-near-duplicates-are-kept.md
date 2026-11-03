# ADR 0006: Near-duplicates are indexed, and identical chunks are hidden at query time

**Status:** Accepted, measurement pending. **Date:** 2026-11-01.

## Context

Real folders hold copies, renamed copies and new editions. Exact copies are safe to drop. Near-copies
aren't: two policies can differ in the one number that matters.

## Decision

- Same bytes or same normalised text: mark as `duplicate` and index nothing.
- Same logical document with a later date: the new version replaces the old one atomically. An older
  edition that arrives late is stored inactive.
- Different document with a close SimHash: index it, and link it to its neighbour.
- At query time, identical chunks collapse into one hit, which lists the others under `also_in`.

## Alternatives

| Option | Why not |
|---|---|
| Drop near-duplicates at ingest | Silently loses the document that differs |
| Keep everything | Top results fill with copies of one passage |

## Consequences

- Embeddings are reused by text and context hash, so an unchanged upload does no embedding work.
- Version detection reads file names and dates, so a badly named edition is treated as a new document.
- Its effect on stale-edition results is **unmeasured** on v2. On v1, stale editions reached the top 5
  on 73% of version questions (15 items).
