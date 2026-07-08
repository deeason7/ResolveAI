"""LLM client with automatic fallback: Ollama (local) -> Groq (cloud).

Both providers speak the OpenAI-compatible API, so a single ``instructor``-
patched code path serves both. This module owns *transport* concerns only:
provider fallback, structured parsing with bounded retries, request timing, and
token accounting. It is deliberately schema-agnostic -- callers pass whatever
Pydantic ``response_model`` they need, so the classification contract stays in
the classifier, not here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Generic, TypeVar

import httpx
import instructor
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel

from app.config import Settings
from app.config import settings as default_settings
from app.services.rate_limit import TokenBucketLimiter

logger = logging.getLogger(__name__)

try:  # instructor raises this once its validation retries are exhausted
    from instructor.core import InstructorRetryException
except Exception:  # pragma: no cover - import path moved across versions
    try:
        from instructor.exceptions import InstructorRetryException
    except Exception:
        InstructorRetryException = Exception  # type: ignore[assignment,misc]

# Bound to BaseModel so every caller hands us a validatable schema.
T = TypeVar("T", bound=BaseModel)

# Hard ceiling on a single rate-limit backoff. A malformed or hostile
# Retry-After header can't park a worker thread for minutes; past this we'd
# rather give up to the fallback chain.
_MAX_BACKOFF_S = 60.0


def _retry_after_seconds(exc: RateLimitError, default: float) -> float:
    """Seconds to wait before retrying a 429, honoring the provider's hint.

    Groq and OpenAI return a ``Retry-After`` header (seconds) on a rate-limit
    429 — the server knows when our window resets, so we obey it rather than
    guessing with blind exponential backoff. Missing or unparseable (e.g. the
    HTTP-date form, which we don't bother decoding) falls back to ``default``.
    Always clamped to ``[0, _MAX_BACKOFF_S]``.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after")
    wait = default
    if raw is not None:
        try:
            wait = float(raw)
        except (TypeError, ValueError):
            wait = default
    return max(0.0, min(wait, _MAX_BACKOFF_S))


def _as_rate_limit_error(exc: BaseException) -> RateLimitError | None:
    """Recover the ``RateLimitError`` behind a 429, unwrapping instructor if needed.

    A bare ``RateLimitError`` is returned as-is. But instructor runs its own
    retry loop *inside* ``create_with_completion``; when a burst 429 outlives
    that budget it re-raises ``InstructorRetryException`` with the original 429
    chained underneath. Left unexamined that reads as a generic provider failure
    and we fail-close to the fallback chain -- so we walk the ``__cause__`` /
    ``__context__`` chain to tell "wait and retry this provider" (a rate limit)
    apart from "this provider is broken" (anything else). Returns ``None`` when
    no 429 is in the chain, i.e. the caller should move on to the next provider.
    """
    current: BaseException | None = exc
    # Cause chains are short; the bound is just a guard against a cyclic chain.
    for _ in range(10):
        if isinstance(current, RateLimitError):
            return current
        nxt = current.__cause__ or current.__context__
        if nxt is None:
            break
        current = nxt
    return None


# --- Proactive TPM pacing (cloud free tiers meter tokens/min, not requests) ---
# A pre-call token estimate: ~4 chars/token is the usual English ballpark, plus
# a flat completion allowance we can't know until the response. reconcile()
# trues this up against real usage after each call, so the estimate only has to
# be roughly right to pace the *first* burst — the part backoff can't catch.
_CHARS_PER_TOKEN = 4.0
_COMPLETION_TOKEN_ALLOWANCE = 400
# Shave the advertised limit: estimate error plus cross-process slop (the API
# and both workers each hold their own in-process bucket but share one Groq
# budget) shouldn't nudge us back over the wall.
_TPM_SAFETY = 0.9


def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Cheap pre-call token estimate: chars/4 over content + a completion allowance."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return int(chars / _CHARS_PER_TOKEN) + _COMPLETION_TOKEN_ALLOWANCE


class Provider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"


class LLMUnavailableError(RuntimeError):
    """Raised when every configured provider failed for a single request."""


@dataclass(frozen=True)
class _ProviderCfg:
    provider: Provider
    base_url: str
    api_key: str
    model: str


@dataclass
class LLMResponse(Generic[T]):
    """One successful structured completion plus the metadata tracking needs."""

    data: T
    provider: Provider
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    is_fallback: bool
    raw_json: str


class LLMClient:
    """Ollama-primary, Groq-fallback structured-completion client.

    The fallback chain is built once at construction. Each provider gets its
    own ``instructor``-patched OpenAI client (distinct base URL / key / model),
    and :meth:`structured` walks the chain until one succeeds.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        limiter: TokenBucketLimiter | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.timeout_s = self.settings.llm_timeout_s
        self._chain: list[_ProviderCfg] = self._build_chain(self.settings)
        # Proactive TPM pacing for the cloud provider (None => disabled). An
        # explicit limiter overrides the settings-built one so tests can watch it.
        self._limiter: TokenBucketLimiter | None = (
            limiter if limiter is not None else self._build_limiter(self.settings)
        )
        self._clients: dict[Provider, instructor.Instructor] = {}
        for cfg in self._chain:
            # max_retries=0: let instructor own the (validation) retry budget;
            # we do NOT want the OpenAI SDK silently retrying network failures
            # and masking a provider outage we would rather fall back on.
            raw = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                timeout=self.timeout_s,
                max_retries=0,
            )
            self._clients[cfg.provider] = instructor.from_openai(raw, mode=instructor.Mode.JSON)

    @staticmethod
    def _build_chain(settings: Settings) -> list[_ProviderCfg]:
        """Ollama is primary; Groq is appended when a key is set.

        With ``llm_skip_local`` on, Ollama is dropped so requests go straight to
        Groq. On hardware that can't serve the local SLM, trying it first only
        buys a guaranteed failure and a wasted timeout on every call — so Groq
        becomes the de-facto primary (``is_fallback`` is then False for it).
        """
        chain: list[_ProviderCfg] = []
        if not settings.llm_skip_local:
            # Ollama's OpenAI-compatible surface lives under /v1.
            ollama_base = settings.ollama_base_url.rstrip("/") + "/v1"
            chain.append(
                _ProviderCfg(
                    provider=Provider.OLLAMA,
                    base_url=ollama_base,
                    api_key="ollama",  # ignored by Ollama; OpenAI SDK needs non-empty
                    model=settings.ollama_model,
                )
            )
        if settings.groq_api_key:
            chain.append(
                _ProviderCfg(
                    provider=Provider.GROQ,
                    base_url=settings.groq_base_url.rstrip("/"),
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                )
            )
        return chain

    @staticmethod
    def _build_limiter(settings: Settings) -> TokenBucketLimiter | None:
        """A TPM bucket for the cloud provider, or None when pacing is off.

        ``groq_tpm_limit <= 0`` (the local-first default) means no bucket —
        Ollama has no such cap and existing behavior stays byte-identical. The
        managed-tier deploy sets it to the provider's tokens/min. Capacity is one
        minute's worth, shaved by ``_TPM_SAFETY`` for headroom.
        """
        tpm = getattr(settings, "groq_tpm_limit", 0)
        if tpm and tpm > 0:
            effective = tpm * _TPM_SAFETY
            return TokenBucketLimiter(rate_per_sec=effective / 60.0, capacity=effective)
        return None

    def structured(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        *,
        max_retries: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Return a validated ``response_model`` from the first healthy provider.

        ``temperature`` defaults to 0.0 — classification and judging want
        determinism. Callers that need a warmer model (the tone judge runs at
        0.1 per spec) override it per call.

        Raises:
            LLMUnavailableError: if every provider in the chain failed.
        """
        retries = self.settings.classification_max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for idx, cfg in enumerate(self._chain):
            try:
                return self._attempt(
                    cfg,
                    response_model,
                    messages,
                    retries,
                    is_fallback=idx > 0,
                    temperature=temperature,
                )
            except (
                APIConnectionError,
                APITimeoutError,
                APIError,
                InstructorRetryException,
                httpx.HTTPError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "provider %s failed (%s): %s",
                    cfg.provider.value,
                    type(exc).__name__,
                    exc,
                )
                continue
        raise LLMUnavailableError(f"all providers failed; last error: {last_error}") from last_error

    def _attempt(
        self,
        cfg: _ProviderCfg,
        response_model: type[T],
        messages: list[dict[str, str]],
        retries: int,
        *,
        is_fallback: bool,
        temperature: float,
    ) -> LLMResponse[T]:
        client = self._clients[cfg.provider]
        # Proactive TPM pacing: only the cloud provider is rate-capped, and only
        # when a bucket is configured. Estimate the cost up front, then true it
        # up against real usage once the call returns.
        paced = self._limiter is not None and cfg.provider is Provider.GROQ
        estimated_tokens = _estimate_prompt_tokens(messages) if paced else 0
        # Same-provider retry budget for 429s only. We sit inside _attempt (one
        # provider) rather than structured() (the cross-provider chain) because
        # a rate limit means "this provider is fine, just wait" — not "this
        # provider is down, move on". `started` is taken per iteration and only
        # read after success, so the reported latency is the winning call's
        # wall time, excluding the backoff sleeps (which aren't model latency).
        attempt = 0
        while True:
            if paced:
                # Block until the budget can cover this call — the sleep that
                # keeps a burst from ever reaching the 429. Safe to block here:
                # every structured() caller runs under asyncio.to_thread.
                self._limiter.acquire(estimated_tokens)
            started = time.perf_counter()
            try:
                obj, completion = client.chat.completions.create_with_completion(
                    model=cfg.model,
                    messages=messages,
                    response_model=response_model,
                    max_retries=retries,
                    temperature=temperature,
                )
                break
            except (RateLimitError, InstructorRetryException) as exc:
                # A 429 can arrive bare, or wrapped by instructor's own retry
                # loop once its budget is spent. Unwrap to find out which -- a
                # non-429 (e.g. exhausted validation retries) means this provider
                # genuinely failed, so re-raise and let structured() move on.
                rate_error = _as_rate_limit_error(exc)
                if rate_error is None:
                    raise
                if attempt >= self.settings.llm_rate_limit_retries:
                    # Backoff budget spent — re-raise so structured() can fall
                    # back to the next provider (or surface LLMUnavailableError).
                    raise
                wait_s = _retry_after_seconds(rate_error, self.settings.llm_rate_limit_backoff_s)
                logger.warning(
                    "provider %s rate-limited (429%s); backing off %.1fs then retrying (%d/%d)",
                    cfg.provider.value,
                    "" if rate_error is exc else " wrapped",
                    wait_s,
                    attempt + 1,
                    self.settings.llm_rate_limit_retries,
                )
                time.sleep(wait_s)
                attempt += 1
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        if paced:
            # Correct the bucket by (actual - estimate) so long-run throughput
            # tracks the real limit rather than our up-front guess.
            self._limiter.reconcile((prompt_tokens + completion_tokens) - estimated_tokens)

        try:
            raw_json = completion.choices[0].message.content or ""
        except (AttributeError, IndexError):
            raw_json = ""

        return LLMResponse(
            data=obj,
            provider=cfg.provider,
            model=cfg.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            is_fallback=is_fallback,
            raw_json=raw_json,
        )

    def health_check(self) -> dict[str, bool]:
        """Best-effort liveness probe per provider (hits the /models endpoint)."""
        status: dict[str, bool] = {}
        for cfg in self._chain:
            url = cfg.base_url.rstrip("/") + "/models"
            try:
                resp = httpx.get(
                    url,
                    headers={"Authorization": f"Bearer {cfg.api_key}"},
                    timeout=5.0,
                )
                status[cfg.provider.value] = resp.status_code == 200
            except httpx.HTTPError:
                status[cfg.provider.value] = False
        return status


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Process-wide singleton client wired to the configured providers.

    Building a provider client also builds its HTTP connection pool, so we do it
    once per process and reuse it — same convention as ``embedder._get_model``
    and ``vector_store.get_default_store``. Constructing a fresh ``LLMClient``
    per classification would leak pools under load.
    """
    return LLMClient()


def reset_llm_client() -> None:
    """Drop the singleton so the next call rebuilds it. Test hook only."""
    get_llm_client.cache_clear()
