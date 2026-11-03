# Benchmarks

Measured on 2026-11-02 on one Mac: 10 CPUs, with Docker limited to about 7.7GB of RAM. Load generator,
API and Postgres shared the machine. **One run of each. Treat these as a sizing guide, not a promise.**

None of this measures retrieval quality or a live model. Model streaming wasn't in the load test,
because no key was used.

## HNSW index, 1M vectors

`python scripts/bench_index.py --vectors 1000000`. 384 dimensions, synthetic clustered vectors, 50
queries per row, m=16, ef_construction=64, ef_search=100, recall against exact search.

| Query | p50 | p95 | recall@10 |
|---|---|---|---|
| No filter | 4.37 ms | 6.73 ms | 0.758 |
| 1% filter, iterative scans off | 3.98 ms | 6.53 ms | 0.122 |
| 1% filter, `relaxed_order` | 9.66 ms | 14.89 ms | 0.910 |

Load took 7.0s, the parallel index build 88.2s, and table plus index 1985MB.

- Iterative scans are why a small collection in a big table works: recall rises from 0.122 to 0.910.
- Unfiltered recall of 0.758 is the weakest number here. It's lower than the filtered one, which I
  haven't investigated. Synthetic data may be harder or easier than real text. `HNSW_EF_SEARCH` is the
  setting to raise, and it wasn't swept.

Two earlier runs showed recall of 1.0 and were wrong. `SET LOCAL enable_indexscan = off` from the exact
pass outlived its savepoint, and after that fix, a prepared plan from the exact pass still answered the
approximate pass, because both used the same SQL. The script now runs with no prepared statements and
checks that the plan uses the index.

## Service load

`scripts/load_test.py`, closed loop, 30s per level, against the real image: 2 API containers, on the
216-file corpus (195 indexed, 531 chunks), using golden questions, with no model key. So this covers
embedding, search, reranking and context, and stops before generation.

Ingesting that corpus through the API with 2 workers took 104.7s, ending as 195 succeeded, 9 duplicates
and 12 quarantined, each with a reason.

| Concurrent clients | Requests/s | p50 | p95 | Errors |
|---|---|---|---|---|
| 1 | 1.8 | 402 ms | 1,040 ms | 0 |
| 5 | 3.6 | 1,317 ms | 2,586 ms | 0 |
| 10 | 3.9 | 2,617 ms | 3,993 ms | 0 |
| 20 | 4.3 | 5,072 ms | 6,451 ms | 0 |
| 40 | 8.2 | 5,047 ms | 6,660 ms | 0 |

Settings: `RERANK_CONCURRENCY=2`, `RERANK_THREADS=2`, `RERANK_WAIT_S=5`. Peak memory was about 1.1GB per
replica.

- Sustained capacity with reranking is about **4 requests/s for 2 replicas** on this machine.
- The 40-client row is faster because requests are being shed: 172 of the two replicas' queries in the
  whole run skipped reranking after waiting 5s, and those are flagged in `degraded`.
- Reranking is nearly all the time. On one replica, rerank summed to 1,244s over 354 queries, and
  each other stage to under 2s. The wait for a slot is included in that figure.

### Before the fix

With no limit on concurrent inference, throughput peaked at 2.8 requests/s, one container reached
4.2GB, and the other was OOM-killed at 40 clients. Unit and integration tests had all passed.
