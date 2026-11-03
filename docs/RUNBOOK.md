# ReconRAG runbook

## Deploy

```bash
make setup                                        # Generates secrets in .env
docker compose up -d --scale api=2 --scale worker=2
```

Compose runs `reconrag migrate` first, and API and workers wait for it. With more than one API
replica, remove the fixed host port and put a load balancer in front. All replicas need the same
`.env`, and all containers need the same `blobs` volume.

Before production:

- Put TLS and a proxy in front. The API speaks plain HTTP.
- Block `/metrics` at the proxy. It has no auth.
- **Encrypt the volumes.** Original uploads sit unencrypted in `blobs`, including PHI files. Only
  chunks and the vault are pseudonymised or encrypted. Filenames are stored as uploaded.
- Back up Postgres and `blobs` together. `PHI_HMAC_KEY` and `PHI_ENCRYPTION_KEY` must be backed up
  separately: without them, PHI tokens can never be re-identified.

## Limits to know

- Every API key can reach every collection. There are no tenants, so run one deployment per trust
  boundary.
- `DELETE` on a document retires it from search but keeps its data. Only deleting the collection erases.
- With Redis down, rate limits are per process, so the real limit is multiplied by the replica count.

## Rollback

Deploy the previous image. Migrations 0001 and 0002 both have downgrades:

```bash
docker compose run --rm migrate alembic downgrade 0001
```

0002 only adds columns, so the previous image runs fine with it applied. Skip the downgrade unless
you need it gone.

## Health

| Check | Means |
|---|---|
| `GET /healthz` | Process is up. No dependencies |
| `GET /readyz` | Postgres reachable. Also reports Redis and available models |
| `GET /metrics` on the API, and port 9100 on a worker | Prometheus metrics |

Worker metrics need `WORKER_METRICS_PORT`, which compose sets to 9100 inside the container network.

## Following one request

Every response has `X-Request-ID`, which is also in error bodies. Each log line carries `request_id`.
Jobs store the id of the upload that created them, so the same id appears in worker logs, and those
lines also have `job_id` and `entry_point` (`api` or `cli`). `reconrag ingest` prints its run id.

```bash
docker compose logs api worker | grep <request_id>
```

## Alerts

Proposed thresholds. **None has been fired, and none is based on real traffic yet.** Tune them after a
week of real numbers. There is no Prometheus in compose, so these need your monitoring stack.

| Alert | Rule | Severity | First action |
|---|---|---|---|
| High error rate | 5xx above 1% for 5 min on `reconrag_http_requests_total` | Page | See "API returns 503 or 500" |
| Slow queries | p95 of `reconrag_http_seconds` above 10s for 10 min on the query route | Page | Check `reconrag_stage_seconds` for the slow stage |
| Queue stuck | `reconrag_queue_oldest_age_seconds` above 600 for 10 min | Page | See "Jobs aren't finishing" |
| Workers gone | Worker metrics target down for 5 min | Page | The queue gauges freeze with no worker, so this alert has to be separate |
| Jobs dying | Rate of `reconrag_jobs_total{status="dead"}` above 0 for 1 h | Ticket | See "A job is dead" |
| Provider cut off | `reconrag_breaker_open` is 1 for 5 min | Ticket | Check the provider's status and key |
| Spend | `reconrag_llm_cost_usd_total` beyond budget | Ticket | Lower rate limits or pause keys |

## Common problems

**API returns 503 `overloaded`.** The DB pool ran out. Check `reconrag_db_pool_waiting`. Add replicas,
or raise `DB_POOL_MAX` if Postgres has connections to spare. Each replica holds up to that many.

**API returns 500.** Find the `request_id` from the response in the logs. The log has the stack trace.

**Jobs aren't finishing.** Check that workers are up, then look for what they're doing:

```sql
SELECT status, stage, count(*), min(created_at) FROM ingest_jobs
WHERE status IN ('queued', 'running') GROUP BY 1, 2;
```

A worker that died leaves a `running` job, and it returns after `WORKER_LEASE_S`. That isn't a fault.

**A job is dead.** Read the job's `error_code`, then the worker logs for its id. Fix the cause, then:

```bash
reconrag retry-job <job_id>
```

**A file is quarantined.** That's a decision, not a failure. `error_code` says why, such as
`encrypted_file` or `zip_bomb`. Don't retry it. Fix the file and upload it again.

**Answers are refused too often, or not enough.** `reconrag_refusals_total` shows why. The gate
thresholds are uncalibrated guesses until the evals are adapted to v2, so expect to tune
`GATE_MIN_RERANK` and `GATE_MIN_DENSE`.

**Queries are slow, or `degraded` lists `reranker_unavailable`.** The cross-encoder is the query
bottleneck, and each process runs `RERANK_CONCURRENCY` at a time. `reconrag_rerank_shed_total` counts
requests that waited `RERANK_WAIT_S` and skipped reranking. Add replicas. Raising the concurrency
doesn't help, and unbounded concurrency got a container OOM-killed. See `docs/BENCHMARKS.md`.

**Answers are slow on the first request.** Models load at startup. Check that startup finished.

**`could not resize shared memory segment`.** A parallel HNSW build outgrew `/dev/shm`. Raise
`shm_size` on Postgres.

**A parser keeps dying.** `parser_crashed` or `parse_timeout` retries, then the job goes dead. The
worker survives, because parsing runs in a child process. The child has no memory limit of its own,
only the container's `mem_limit`, which can kill the whole worker. Its job then retries on the lease.

## Rotating secrets

- API keys: add the new key to `API_KEYS`, restart, move clients over, remove the old one.
- `PHI_ENCRYPTION_KEY`: there's no rotation command yet. Don't change it on a deployment with data.
- `PHI_HMAC_KEY`: changing it breaks token matching for all existing data. Treat it as permanent.
