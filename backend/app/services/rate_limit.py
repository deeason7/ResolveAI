"""Token-bucket limiter for tokens-per-minute (TPM) budgets.

Cloud LLM free tiers meter *tokens* per minute, not requests. Draining an
enqueue fires completions back-to-back and blows that budget; the provider
429s, and ``instructor`` swallows the 429 into a retry exception before the
client's reactive ``Retry-After`` backoff can act on it — so the call
fail-closes to the deterministic fallback. The cure is proactive: meter tokens
*before* the call so we never trip the wall.

``TokenBucketLimiter`` is a plain token bucket whose "tokens" are LLM tokens: it
refills at ``rate_per_sec`` up to ``capacity`` and each call withdraws its
(estimated, later reconciled) cost, sleeping when the balance would go negative.
Provider-agnostic and stdlib-only.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """Thread-safe token bucket metering a tokens-per-minute budget.

    The clock and sleep are injectable so the pacing is unit-testable without
    wall-clock flakiness. Production uses ``time.monotonic`` (immune to NTP
    steps — a wall-clock jump backwards must never hand out free tokens) and
    ``time.sleep``.
    """

    def __init__(
        self,
        rate_per_sec: float,
        capacity: float,
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_sec <= 0 or capacity <= 0:
            raise ValueError("rate_per_sec and capacity must both be positive")
        self._rate = rate_per_sec
        self._capacity = capacity
        self._now = now
        self._sleep = sleep
        self._tokens = capacity  # start full: a quiet bucket pays no cold-start toll
        self._updated = now()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        """Credit the tokens accrued since the last touch. Caller holds the lock."""
        now = self._now()
        elapsed = now - self._updated
        if elapsed > 0:
            self._updated = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    def acquire(self, tokens: float) -> float:
        """Withdraw ``tokens``, sleeping until the budget can cover them.

        Returns the seconds slept (0.0 when the bucket had room immediately). A
        request larger than the whole bucket is clamped to ``capacity`` so a
        single fat prompt can never park a caller longer than one refill window.
        """
        want = min(float(tokens), self._capacity)
        with self._lock:
            self._refill_locked()
            self._tokens -= want
            # A negative balance is debt: the next caller sees it through
            # _refill_locked and waits proportionally, so concurrent callers
            # queue up in order instead of all firing at once.
            wait = -self._tokens / self._rate if self._tokens < 0 else 0.0
        if wait > 0:
            logger.debug("token bucket empty; pacing %.2fs for ~%d tokens", wait, int(want))
            self._sleep(wait)
        return wait

    def reconcile(self, delta: float) -> None:
        """Correct the balance by ``actual - estimate`` once real usage is known.

        Positive ``delta`` (we under-estimated) debits the difference so the
        next call waits a touch longer; negative credits it back. Clamped to
        ``capacity`` so a run of cheap calls can't bank unbounded burst credit.
        """
        with self._lock:
            self._refill_locked()
            self._tokens = min(self._capacity, self._tokens - delta)
