# ADR 0002: A synchronous core, with plain `def` routes

**Status:** Accepted. **Date:** 2026-11-01. Replaces the SPEC's "async providers".

## Context

The API, the worker and the CLI all call the same storage, embedding and model code. v1's bug was an
`async def` route calling blocking SDKs, which stalled the event loop for every other request.

## Decision

Everything is synchronous: psycopg 3 with a connection pool, and plain `def` FastAPI routes, which
FastAPI runs in its threadpool.

## Alternatives

| Option | Why not |
|---|---|
| Async everywhere | Better for many idle connections, but the worker and CLI need sync or `asyncio.run` wrappers, and one blocking call anywhere repeats v1's bug |
| Both, sync and async | Two code paths to test for the same behaviour |

## Consequences

- One code path, and no way to stall an event loop from a route.
- Concurrency per replica is capped by the threadpool and the DB pool. A long LLM stream holds a thread
  for its whole length, which is the real limit of this design. Scale by adding replicas.
- When the pool runs out, requests get a 503 with `Retry-After`, not a hang.
- If load tests with live model streams fail, move only the query routes to async. Nothing else needs it.
