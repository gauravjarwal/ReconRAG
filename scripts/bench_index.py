"""Index benchmark on synthetic clustered vectors, in a throwaway database.

Measures build time, query latency and recall against exact search, for a selective filter with
iterative scans on and off. Synthetic data isn't real data: treat the output as one data point.

Usage: python scripts/bench_index.py --vectors 1000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

ADMIN = os.environ.get("BENCH_ADMIN_URL", "postgresql://reconrag:reconrag@localhost:5433/postgres")
DIM = 384
QUERIES = 200
K = 10
SMALL_TENANT_SHARE = 0.01
BATCH = 50_000


def vectors(rng: np.random.Generator, count: int, centres: np.ndarray) -> np.ndarray:
    picks = rng.integers(0, len(centres), count)
    points = centres[picks] + rng.normal(0, 0.35, (count, DIM)) / np.sqrt(DIM) * 3
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def load(conn: psycopg.Connection, rng: np.random.Generator, total: int, centres: np.ndarray) -> float:
    conn.execute(f"CREATE TABLE items (id bigint, tenant int, embedding halfvec({DIM}))")
    started = time.perf_counter()
    done = 0
    with conn.cursor() as cursor:
        while done < total:
            size = min(BATCH, total - done)
            batch = vectors(rng, size, centres)
            tenants = (rng.random(size) >= SMALL_TENANT_SHARE).astype(int)
            with cursor.copy("COPY items FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int8", "int4", "halfvec"])
                for index in range(size):
                    copy.write_row((done + index, int(tenants[index]), HalfVector(batch[index])))
            conn.commit()
            done += size
    return time.perf_counter() - started


def percentile(values: list[float], share: float) -> float:
    return round(float(np.percentile(values, share)), 2)


def run_queries(conn: psycopg.Connection, queries: np.ndarray, exact: list[set[int]], iterative: str, tenant: int | None) -> dict:
    latencies: list[float] = []
    recalls: list[float] = []
    where = "" if tenant is None else f"WHERE tenant = {tenant}"
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        conn.execute(f"SET LOCAL hnsw.iterative_scan = {iterative}")
        conn.execute("SET LOCAL enable_seqscan = off")
        plan = conn.execute(
            f"EXPLAIN SELECT id FROM items {where} ORDER BY embedding <=> %s LIMIT {K}",
            (HalfVector(queries[0]),),
        ).fetchall()
        if "items_embedding_idx" not in " ".join(row[0] for row in plan):
            raise RuntimeError(f"Not measuring the index: {plan}")
        for query, truth in zip(queries, exact, strict=True):
            started = time.perf_counter()
            rows = conn.execute(
                f"SELECT id FROM items {where} ORDER BY embedding <=> %s LIMIT {K}",
                (HalfVector(query),),
            ).fetchall()
            latencies.append((time.perf_counter() - started) * 1000)
            recalls.append(len({row[0] for row in rows} & truth) / K)
    return {
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "recall@10": round(float(np.mean(recalls)), 3),
    }


def exact_answers(conn: psycopg.Connection, queries: np.ndarray, tenant: int | None) -> list[set[int]]:
    where = "" if tenant is None else f"WHERE tenant = {tenant}"
    answers = []
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL statement_timeout = 0")
        for query in queries:
            rows = conn.execute(
                f"SELECT id FROM items {where} ORDER BY embedding <=> %s LIMIT {K}",
                (HalfVector(query),),
            ).fetchall()
            answers.append({row[0] for row in rows})
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=int, default=100_000)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(200, DIM))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    name = f"bench_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    url = ADMIN.rsplit("/", 1)[0] + f"/{name}"
    try:
        # No prepared statements: exact and approximate search share SQL text, so a plan cached
        # for the exact pass would silently answer the approximate pass too.
        with psycopg.connect(url, prepare_threshold=None) as conn:
            conn.execute("CREATE EXTENSION vector")
            conn.commit()
            register_vector(conn)
            load_s = load(conn, rng, args.vectors, centres)
            conn.execute("SET maintenance_work_mem = '2GB'")
            conn.execute("SET max_parallel_maintenance_workers = 7")
            started = time.perf_counter()
            conn.execute(
                "CREATE INDEX ON items USING hnsw (embedding halfvec_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
            conn.commit()
            build_s = time.perf_counter() - started
            size = conn.execute("SELECT pg_size_pretty(pg_total_relation_size('items'))").fetchone()
            conn.execute("ANALYZE items")
            conn.commit()
            # Autocommit makes each `conn.transaction()` real. Inside an open transaction it is a
            # savepoint, and SET LOCAL survives a savepoint: exact search would leak into the rest.
            conn.autocommit = True
            queries = vectors(rng, QUERIES, centres)
            everyone = exact_answers(conn, queries[:50], None)
            small = exact_answers(conn, queries[:50], 0)
            report = {
                "vectors": args.vectors,
                "dimensions": DIM,
                "load_s": round(load_s, 1),
                "build_s": round(build_s, 1),
                "table_and_index": size[0] if size else None,
                "no_filter": run_queries(conn, queries[:50], everyone, "off", None),
                "filter_1pct_iterative_off": run_queries(conn, queries[:50], small, "off", 0),
                "filter_1pct_relaxed_order": run_queries(conn, queries[:50], small, "relaxed_order", 0),
            }
        print(json.dumps(report, indent=2))
    finally:
        with psycopg.connect(ADMIN, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


if __name__ == "__main__":
    main()
