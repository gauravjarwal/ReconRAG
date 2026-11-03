# Plan: ReconRAG v2 refactor

The evals slice is finished. This plan rewrites the product itself in `src/reconrag/`.

**v2 is designed independently of the eval harness.** Nothing here exists to satisfy the eval corpus,
the golden sets or the `Pipeline` protocol, and those files aren't consulted for design. When v2 is
done, the evals are adapted to it, or changed, in a separate step. Task list: `tasks/todo.md`.

## Architecture decisions

| Decision | Why | Cost |
|---|---|---|
| Postgres holds chunks, vectors, FTS, jobs and the PHI vault | One system to back up, one transaction to publish a document, no SQLite single-replica limit | Vector and lexical load share one database |
| Synchronous core with `psycopg` and plain `def` routes | One code path for API, worker and CLI, and easier to reason about | Concurrency is capped by the threadpool and DB pool. See ADR 0002 |
| Job queue in Postgres with `FOR UPDATE SKIP LOCKED` and leases | Publishing and finishing a job commit together, so no broker is needed and no duplicates can appear | Lower peak throughput than Kafka or SQS, which single-node doesn't need |
| Blocks, not strings, leave the parser | Headings, tables and pages survive to the chunker and citations | Every parser must emit structure |
| Near-duplicates are indexed, not dropped | Two near-identical documents can differ in the one number that matters | Identical chunks are suppressed at query time instead |
| Parsing runs in a child process with a timeout | A malicious PDF can hang or kill a parser without killing the worker | Some process overhead per restart |
| Local embeddings and a cross-encoder by default | Works with no API key. Gemini embeddings are a config switch | Local models are weaker than the best hosted ones, unmeasured so far |
| PHI-protected collections don't stream tokens | Output checks must run before text leaves the service | Those collections have worse time to first token |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Sync routes under many long LLM streams | High | Bounded pool, a load test, and a documented move to async if it fails |
| Postgres FTS isn't true BM25 | Medium | OR queries, normalised rank, trigram fallback; recorded in an ADR |
| OCR quality on noisy scans | Medium | Per-page routing, stored confidence, low-quality documents flagged |
| PyMuPDF is AGPL | Medium | Behind one parser module, flagged in an ADR |
| More code the owner didn't write | High | Module quizzes in `docs/V2_DEFENSE.md` |

## After this plan

Adapt `evals/` to v2 (or redesign them), run the comparison, and only then claim any improvement.
