"""Prometheus metrics. Labels stay low-cardinality: never a collection name or user id."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_LATENCY = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

HTTP_REQUESTS = Counter("reconrag_http_requests_total", "Requests", ["route", "status"])
HTTP_LATENCY = Histogram("reconrag_http_seconds", "Request latency", ["route"], buckets=_LATENCY)
STAGE_LATENCY = Histogram("reconrag_stage_seconds", "Stage latency", ["stage"], buckets=_LATENCY)
TTFT = Histogram("reconrag_ttft_seconds", "Time to first token", ["provider"], buckets=_LATENCY)
LLM_CALLS = Counter("reconrag_llm_calls_total", "Provider calls", ["provider", "outcome"])
LLM_TOKENS = Counter("reconrag_llm_tokens_total", "Tokens", ["provider", "direction"])
LLM_COST = Counter("reconrag_llm_cost_usd_total", "Estimated spend", ["provider"])
BREAKER_OPEN = Gauge("reconrag_breaker_open", "1 while a provider is cut off", ["provider"])
CACHE = Counter("reconrag_cache_total", "Query embedding cache", ["outcome"])
REFUSALS = Counter("reconrag_refusals_total", "Refused questions", ["reason"])
JOBS = Counter("reconrag_jobs_total", "Finished ingest jobs", ["status"])
JOB_RETRIES = Counter("reconrag_job_retries_total", "Retried ingest jobs", ["reason"])
INGEST_SECONDS = Histogram(
    "reconrag_ingest_seconds", "Ingest stage latency", ["stage"], buckets=_LATENCY
)
QUEUE_DEPTH = Gauge("reconrag_queue_depth", "Jobs queued or running")
QUEUE_AGE = Gauge("reconrag_queue_oldest_age_seconds", "Age of the oldest job that could run now")
DB_POOL_SIZE = Gauge("reconrag_db_pool_size", "Connections held by this process")
DB_POOL_AVAILABLE = Gauge("reconrag_db_pool_available", "Idle connections in this process")
DB_POOL_WAITING = Gauge("reconrag_db_pool_waiting", "Requests waiting for a connection")
RERANK_SHED = Counter("reconrag_rerank_shed_total", "Queries that skipped rerank under load")
REDIS_FAILURES = Counter("reconrag_redis_failures_total", "Redis errors handled", ["operation"])
