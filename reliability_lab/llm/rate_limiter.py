"""Thread-safe rate limiter for controlling LLM request pacing."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Sliding-window request rate limiter.

    Paces outbound requests to stay within a configured requests-per-minute (RPM)
    threshold.
    """

    def __init__(self, requests_per_minute: int = 60):
        if requests_per_minute <= 0:
            raise ValueError(
                f"requests_per_minute must be strictly positive, got {requests_per_minute}"
            )
        self.rpm = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """Acquire permission to execute a request, pacing to remain under RPM limit."""
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                now = time.monotonic()
                window_start = now - 60.0

                # Discard timestamps outside the 60s sliding window
                while self._timestamps and self._timestamps[0] <= window_start:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return True

                oldest = self._timestamps[0]
                sleep_needed = oldest + 60.0 - now

            if sleep_needed <= 0:
                continue

            if time.monotonic() + sleep_needed > deadline:
                return False

            time.sleep(sleep_needed)
