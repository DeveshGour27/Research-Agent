"""In-process, per-user rate limiter for Phase 7.6.

Design:
    - Token-bucket–style sliding window stored in bounded in-memory state.
    - Thread-safe via threading.Lock.
    - Expired entries are cleaned on every check to prevent unbounded growth.
    - Injectable / testable — not hard-coded into route logic.
    - Does NOT require Redis or any external dependency.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class RateLimiter:
    """Per-key sliding-window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        # key -> list of request timestamps (epoch float)
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check whether *key* is allowed to make another request.

        Returns ``(allowed, retry_after_seconds)``.
        ``retry_after_seconds`` is 0 when allowed, else the number of seconds
        until the oldest request in the window expires.
        """
        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            # Purge expired timestamps for this key
            timestamps = self._requests[key]
            self._requests[key] = [t for t in timestamps if t > cutoff]
            timestamps = self._requests[key]

            if len(timestamps) >= self._max_requests:
                retry_after = int(timestamps[0] - cutoff) + 1
                return False, max(retry_after, 1)

            timestamps.append(now)
            return True, 0

    def cleanup(self) -> None:
        """Remove all keys whose windows have fully expired."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            empty_keys = [
                k for k, ts in self._requests.items()
                if not ts or all(t <= cutoff for t in ts)
            ]
            for k in empty_keys:
                del self._requests[k]

    @property
    def active_keys(self) -> int:
        """Number of keys with active (non-expired) windows."""
        with self._lock:
            return len(self._requests)
