"""Circuit breaker. After repeated failures, skip a provider instead of waiting for its timeout."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class CircuitBreaker:
    """Closed, then open after `threshold` straight failures, then one trial after `cooldown_s`."""

    def __init__(
        self,
        threshold: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_running = False
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._trial_running or self._clock() - self._opened_at < self._cooldown_s:
                return False
            self._trial_running = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._trial_running = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._trial_running or self._failures >= self._threshold:
                self._opened_at = self._clock()
            self._trial_running = False
