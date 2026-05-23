"""Thread-and-asyncio-safe token-bucket rate limiter.

Used by the EDGAR client to stay under SEC's documented fair-access policy
(`https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data`):
"Current max request rate: 10 requests/second". We default to a 20% safety
margin (8 req/s) so transient clock skew or a slightly bursty client
doesn't trip the 10 r/s ceiling.

The bucket is single-implementation, dual-interface: `wait()` for sync
callers, `wait_async()` for async callers. State is shared via a single
`threading.Lock`, so one limiter can govern both a sync and an async
client in the same process — important because SEC throttles by IP,
not by client object.

Implementation: deadline-style. We track `_next_available`, the
earliest time the next acquire is allowed to fire. Each acquire snaps
forward by `1/rate`; an idle period longer than the burst window
(`capacity/rate`) is collapsed so callers can burst up to `capacity`
after a quiet stretch. This makes FIFO behaviour under contention
deterministic — successive waiters get reservations 1/rate apart,
instead of all racing for the same future token.

Why hand-roll instead of `aiolimiter` / `ratelimit`: this is ~50 lines
with no third-party dependency, and it's deterministically testable
via a injected `time_fn` (no monkey-patching `time.monotonic`).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

SEC_RATE_LIMIT_PER_SEC: float = 10.0
SEC_SAFE_RATE_PER_SEC: float = 8.0


@dataclass
class TokenBucket:
    """Deadline-style token bucket.

    `rate` requests per second are permitted on average; up to
    `capacity` may fire in a single burst after an idle window of at
    least `capacity/rate` seconds.
    """

    rate: float
    capacity: float | None = None
    time_fn: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")
        if self.capacity is None:
            self.capacity = self.rate
        # `_next_available` is the earliest time the next acquire may fire.
        # Initialise it `burst_window` seconds in the past so the very first
        # `capacity` acquires drain without waiting.
        burst_window = self.capacity / self.rate
        self._next_available: float = self.time_fn() - burst_window
        self._lock = threading.Lock()

    def _acquire(self) -> float:
        """Reserve one slot; return seconds the caller must wait."""
        with self._lock:
            now = self.time_fn()
            assert self.capacity is not None  # set in __post_init__
            burst_window = self.capacity / self.rate
            # If we've been idle longer than the burst window, snap the
            # pointer forward so the bucket is "full" of accumulated
            # tokens — the next acquire bursts immediately.
            if now - self._next_available > burst_window:
                self._next_available = now - burst_window
            available_at = self._next_available + (1.0 / self.rate)
            self._next_available = available_at
            return max(0.0, available_at - now)

    def wait(self) -> None:
        """Block (sync) until a token is available."""
        delay = self._acquire()
        if delay > 0:
            time.sleep(delay)

    async def wait_async(self) -> None:
        """Suspend (async) until a token is available."""
        delay = self._acquire()
        if delay > 0:
            await asyncio.sleep(delay)


def sec_rate_limiter() -> TokenBucket:
    """Default SEC limiter: 8 req/s sustained, burst capacity == rate."""
    return TokenBucket(rate=SEC_SAFE_RATE_PER_SEC)
