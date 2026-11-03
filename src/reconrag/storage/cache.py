"""Redis, treated as optional. Slow or down Redis must never fail a request."""

from __future__ import annotations

import logging
import threading
import time

import redis

from reconrag.config import Settings
from reconrag.observability.metrics import REDIS_FAILURES

logger = logging.getLogger(__name__)

# Atomic, so concurrent requests can't both see a count below the limit.
_RATE_LIMIT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return count
"""


class Cache:
    def __init__(self, settings: Settings) -> None:
        self._client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.redis_timeout_s,
            socket_connect_timeout=settings.redis_timeout_s,
        )
        self._script = self._client.register_script(_RATE_LIMIT)
        self._local: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def get(self, key: str) -> bytes | None:
        try:
            value = self._client.get(key)
        except redis.RedisError:
            REDIS_FAILURES.labels("get").inc()
            return None
        return value if isinstance(value, bytes) else None

    def set(self, key: str, value: bytes, ttl_s: int) -> None:
        try:
            self._client.set(key, value, ex=ttl_s)
        except redis.RedisError:
            REDIS_FAILURES.labels("set").inc()

    def _count_locally(self, key: str, window: int) -> int:
        """Per-process, so with Redis down the real limit is this times the replica count."""
        with self._lock:
            started, count = self._local.get(key, (window, 0))
            count = count + 1 if started == window else 1
            self._local[key] = (window, count)
            if len(self._local) > 10_000:
                self._local = {k: v for k, v in self._local.items() if v[0] == window}
            return count

    def over_limit(self, key: str, limit: int) -> int | None:
        """Fixed one-minute window. Returns seconds to wait if over the limit, else None."""
        now = int(time.time())
        window = now // 60
        try:
            count = int(self._script(keys=[f"rl:{key}:{window}"], args=[61]))  # type: ignore[arg-type]
        except redis.RedisError:
            REDIS_FAILURES.labels("rate_limit").inc()
            count = self._count_locally(key, window)
        return 60 - now % 60 if count > limit else None

    def close(self) -> None:
        self._client.close()
