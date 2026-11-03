# Defending ReconRAG v2

Claude wrote this code. Answer these aloud before accepting it, then check the file named.

## Ingestion

1. Why does parsing run in a child process, and why not `ProcessPoolExecutor`? (`worker/sandbox.py`)
2. Why do exceptions cross that process boundary as tuples? (`worker/sandbox.py`)
3. Why is the file type read from magic bytes, not the extension? (`ingestion/sniff.py`)
4. How is a zip bomb caught without unzipping it? (`ingestion/sniff.py`)
5. Why does a page go to OCR per page, not per document? (`ingestion/parsers/pdf.py`)
6. Why are margin lines limited to 10 words before removal? What broke without it? (`ingestion/clean.py`)
7. Why are near-duplicates indexed and not dropped? (ADR 0006)
8. A file named `rates 2025` arrives after `rates 2026`. Trace what happens. (`worker/processor.py`, `storage/repo.py`)
9. Why is the fallback title the document key, not the filename? (`worker/processor.py`)
10. Why does a table chunk repeat its headers? (`ingestion/chunking.py`)

## Queue

11. What does `FOR UPDATE SKIP LOCKED` do that `FOR UPDATE` doesn't? (`worker/queue.py`)
12. What's the difference between a lease, a heartbeat and a fencing token?
13. A worker pauses for 2 minutes, then wakes and tries to publish. What stops it?
14. Why are publish and finish in one transaction? What breaks if they're two?
15. Is this exactly-once delivery? Say what it really is.
16. Why not Kafka, given you know Kafka well? (ADR 0003)
17. Why does `queue.stats` ignore jobs waiting out a backoff?

## Retrieval

18. Why `halfvec`, and why a partial index on `active`?
19. HNSW filters after the scan. At a 1% filter, what happens with iterative scans off, and why? (`scripts/bench_index.py`)
20. Why does `dense_search` sort its results again in Python? (`retrieval/search.py`)
21. Why OR in the tsquery, and what does `ts_rank` lack compared to BM25? (ADR 0004)
22. Why is trigram search limited to identifiers?
23. Why RRF rather than adding the scores? What does the weight do? (`retrieval/fusion.py`)
24. Why does the gate need a separate path with no reranker? (`retrieval/gate.py`)
25. Where did `-6.0` come from? Give the honest answer.
26. Why is `plan_cache_mode=force_custom_plan` set? (`storage/db.py`)

## Generation and PHI

27. When exactly may the router fall back to another provider, and why not after the first token? (`llm/router.py`)
28. Which is authoritative, the `token` events or `done`? Why?
29. Why do PHI collections send no tokens at all? (`generation/service.py`)
30. Would the numeric grounding check catch a wrong number? A wrong name? (`generation/validate.py`)
31. Why HMAC scoped by collection, and not a plain hash? (`security/phi.py`)
32. Where does raw PHI still exist on disk? (`docs/RUNBOOK.md`)
33. Is this HIPAA compliant? Answer precisely.

## API and operations

34. Why plain `def` routes, and what's the real concurrency limit? (ADR 0002)
35. Why pure ASGI middleware rather than `BaseHTTPMiddleware`? (`api/middleware.py`)
36. Why must the re-identify check be a dependency, not code in the stream? (`api/routes.py`)
37. Why does `/healthz` check nothing while `/readyz` checks Postgres?
38. What happens to rate limits when Redis goes down? (`storage/cache.py`)
39. Follow one upload from `X-Request-ID` to the worker's logs.

## What breaks if a line is removed

40. Remove `AND attempts = %s` from `queue.finish`.
41. Remove `SKIP LOCKED` from `claim`.
42. Remove `if not protected:` before the token event.
43. Remove the `collection_id` filter from `dense_search`.
44. Remove the `NOT EXISTS` from `repo.delete_collection`.
45. Put the scope back out of the rate-limit key in `api/deps.py`.

Mutation runs on 2026-11-02 confirmed that a test catches 40 to 44. Say which test, and why it fails.

## Interview pressure

46. "You benchmarked 1M vectors. Your first run showed perfect recall. Why was that wrong?"
47. "Your recall with no filter is lower than with a filter. Explain that, or say you can't."
48. "How do you know v2 retrieves better than v1?" (You don't yet. Say so.)
49. "Why Postgres for vectors at all?"
50. "What breaks first at 10x the data?"
