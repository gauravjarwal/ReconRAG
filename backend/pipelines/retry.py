from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

import requests
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transient failures worth retrying. 4xx errors (bad key, bad request) are not.
GEMINI_RETRYABLE: tuple[type[BaseException], ...] = (
    google_exceptions.TooManyRequests,
    google_exceptions.ServiceUnavailable,
    google_exceptions.InternalServerError,
    google_exceptions.BadGateway,
    google_exceptions.GatewayTimeout,
    google_exceptions.DeadlineExceeded,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class UpstreamServiceError(Exception):
    """An external model API failed after retries."""


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = GEMINI_RETRYABLE,
    operation: str = "upstream call",
) -> T:
    """Call fn, retrying transient errors with exponential backoff and full jitter."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            logger.warning("%s failed (%s), retry %d/%d in %.2fs", operation, type(exc).__name__, attempt, attempts - 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")
