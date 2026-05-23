"""Unit tests for the token-bucket rate limiter (Issue #5)."""

from __future__ import annotations

import asyncio

import pytest

from auto_research.ingest.rate_limit import (
    SEC_RATE_LIMIT_PER_SEC,
    SEC_SAFE_RATE_PER_SEC,
    TokenBucket,
    sec_rate_limiter,
)


class FakeClock:
    """Monotonic clock you can step manually. Avoids real `time.sleep`."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_full_bucket_serves_burst_immediately() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=10.0, capacity=10.0, time_fn=clock)
    # All 10 capacity tokens drain at t=0 with no waiting.
    waits = [bucket._acquire() for _ in range(10)]
    assert waits == [0.0] * 10


def test_acquire_after_drain_returns_replenishment_delay() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=10.0, capacity=1.0, time_fn=clock)
    assert bucket._acquire() == 0.0  # drains the single token
    next_wait = bucket._acquire()
    assert next_wait == pytest.approx(0.1, rel=1e-6)  # 1 / 10 r/s


def test_acquire_replenishes_proportional_to_elapsed_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=10.0, capacity=10.0, time_fn=clock)
    for _ in range(10):
        bucket._acquire()
    # No tokens left; advance half a second → 5 tokens regenerated.
    clock.advance(0.5)
    waits = [bucket._acquire() for _ in range(5)]
    assert waits == [0.0] * 5
    # Sixth must wait 0.1s for the next token.
    assert bucket._acquire() == pytest.approx(0.1, rel=1e-6)


def test_concurrent_acquires_are_serialised_via_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two acquirers under contention get distinct reservation slots, not the same one.

    The bucket's `_last` advances by `wait_seconds` each time a token
    has to be reserved against future tokens, so the second waiter
    must wait *behind* the first — never the same time.
    """
    clock = FakeClock()
    bucket = TokenBucket(rate=10.0, capacity=1.0, time_fn=clock)
    assert bucket._acquire() == 0.0  # drain
    first = bucket._acquire()
    second = bucket._acquire()
    assert first == pytest.approx(0.1, rel=1e-6)
    assert second == pytest.approx(0.2, rel=1e-6)


def test_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="rate must be positive"):
        TokenBucket(rate=0.0)


def test_sec_rate_limiter_is_under_documented_ceiling() -> None:
    bucket = sec_rate_limiter()
    assert bucket.rate < SEC_RATE_LIMIT_PER_SEC
    assert bucket.rate == SEC_SAFE_RATE_PER_SEC


def test_wait_blocks_for_acquired_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    sleeps: list[float] = []

    def fake_sleep(t: float) -> None:
        sleeps.append(t)
        clock.advance(t)

    monkeypatch.setattr("time.sleep", fake_sleep)
    bucket = TokenBucket(rate=10.0, capacity=1.0, time_fn=clock)
    bucket.wait()  # consumes the one token, no sleep
    bucket.wait()  # has to wait 0.1s
    assert sleeps == [pytest.approx(0.1, rel=1e-6)]


def test_wait_async_suspends_for_acquired_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(t: float) -> None:
        sleeps.append(t)
        clock.advance(t)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def runner() -> None:
        bucket = TokenBucket(rate=10.0, capacity=1.0, time_fn=clock)
        await bucket.wait_async()
        await bucket.wait_async()

    asyncio.run(runner())
    assert sleeps == [pytest.approx(0.1, rel=1e-6)]
