"""Tests for the tokens-per-minute token-bucket limiter.

The bucket takes an injectable clock + sleep, so every timing assertion is
deterministic: ``_Clock.sleep`` just advances the fake clock the way real time
would, and ``advance`` fast-forwards without a call.
"""

from __future__ import annotations

import pytest

from app.services.rate_limit import TokenBucketLimiter


class _Clock:
    """Manually-advanced fake monotonic clock; sleeping moves it forward."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _bucket(clock: _Clock, rate: float = 10.0, capacity: float = 100.0) -> TokenBucketLimiter:
    return TokenBucketLimiter(rate, capacity, now=clock.now, sleep=clock.sleep)


class TestTokenBucketLimiter:
    def test_rejects_nonpositive_config(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(0, 100)
        with pytest.raises(ValueError):
            TokenBucketLimiter(10, 0)

    def test_starts_full_no_wait_within_capacity(self):
        clock = _Clock()
        b = _bucket(clock)
        # Exactly one capacity's worth, and the bucket starts full -> no sleep.
        assert b.acquire(100) == 0.0
        assert clock.slept == []

    def test_overdraw_waits_for_the_deficit(self):
        clock = _Clock()
        b = _bucket(clock, rate=10.0, capacity=100.0)
        b.acquire(100)  # drain to zero
        wait = b.acquire(50)  # 50-token deficit / 10 per s = 5s
        assert wait == pytest.approx(5.0)
        assert clock.slept == [pytest.approx(5.0)]

    def test_refills_over_time(self):
        clock = _Clock()
        b = _bucket(clock, rate=10.0, capacity=100.0)
        b.acquire(100)  # empty
        clock.advance(5.0)  # +50 tokens
        assert b.acquire(50) == 0.0  # covered by the refill, no wait
        assert clock.slept == []

    def test_refill_is_capped_at_capacity(self):
        clock = _Clock()
        b = _bucket(clock, rate=10.0, capacity=100.0)
        b.acquire(100)  # empty
        clock.advance(1000.0)  # 10_000 tokens would accrue, but cap is 100
        assert b.acquire(100) == 0.0
        assert b.acquire(1) > 0.0  # proves the refill stopped at 100, not more

    def test_single_request_larger_than_capacity_is_clamped(self):
        clock = _Clock()
        b = _bucket(clock, rate=10.0, capacity=100.0)
        # Clamped to capacity; the full bucket covers it, so no unbounded park.
        assert b.acquire(1_000_000) == 0.0
        assert b.acquire(100) == pytest.approx(10.0)  # next call waits one window

    def test_reconcile_debits_an_underestimate(self):
        clock = _Clock()
        b = _bucket(clock, rate=10.0, capacity=100.0)
        b.acquire(50)  # 50 left
        b.reconcile(30)  # actual-estimate = +30 -> 20 left
        assert b.acquire(20) == 0.0
        assert b.acquire(1) > 0.0

    def test_reconcile_credit_is_capped_at_capacity(self):
        clock = _Clock()
        b = _bucket(clock, rate=10.0, capacity=100.0)
        b.acquire(50)  # 50 left
        b.reconcile(-1000)  # huge credit, but the bucket can't exceed capacity
        assert b.acquire(100) == 0.0  # back to full, not more
        assert b.acquire(1) > 0.0
