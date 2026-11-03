# ADR 0003: The ingest queue is a Postgres table

**Status:** Accepted. **Date:** 2026-11-01.

## Context

Ingest is slow, must survive a killed worker, and must never create duplicate chunks.

## Decision

Jobs are rows, claimed with `FOR UPDATE SKIP LOCKED`. A claim is a lease, renewed by a heartbeat.
`attempts` is a fencing token: a worker that lost its lease can't finish the job. Publishing the
chunks and finishing the job happen in one transaction. A per-document advisory lock serialises two
versions of one document.

## Alternatives

| Option | Why not |
|---|---|
| Kafka or SQS | The broker's ack and the database commit can't be one transaction, so duplicates need handling anyway. Also new infrastructure |
| Redis queue | Redis is optional here, and its durability is weaker than Postgres |

## Consequences

- Delivery is at-least-once, and the transaction makes the effect exactly-once.
- Peak throughput is lower than a real broker, and workers poll. Parsing and embedding are far slower
  than a claim, so this hasn't been the limit in any run so far.
- `SIGKILL` during embedding and `SIGTERM` during a job are both tested with real processes.
