"""Per-stage timing for one request, returned to the caller and recorded as metrics."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from reconrag.observability.metrics import STAGE_LATENCY


@dataclass
class Trace:
    stages_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.stages_ms[name] = round(self.stages_ms.get(name, 0.0) + elapsed * 1000, 2)
            STAGE_LATENCY.labels(name).observe(elapsed)
