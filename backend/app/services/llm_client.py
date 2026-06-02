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
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from pydantic import BaseModel

from app.config import Settings
from app.config import settings as default_settings

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

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.timeout_s = self.settings.llm_timeout_s
        self._chain: list[_ProviderCfg] = self._build_chain(self.settings)
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

    def structured(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        *,
        max_retries: int | None = None,
    ) -> LLMResponse[T]:
        """Return a validated ``response_model`` from the first healthy provider.

        Raises:
            LLMUnavailableError: if every provider in the chain failed.
        """
        retries = self.settings.classification_max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for idx, cfg in enumerate(self._chain):
            try:
                return self._attempt(cfg, response_model, messages, retries, is_fallback=idx > 0)
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
    ) -> LLMResponse[T]:
        client = self._clients[cfg.provider]
        started = time.perf_counter()
        obj, completion = client.chat.completions.create_with_completion(
            model=cfg.model,
            messages=messages,
            response_model=response_model,
            max_retries=retries,
            temperature=0.0,  # deterministic: classification is not creative
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = getattr(completion, "usage", None)
        try:
            raw_json = completion.choices[0].message.content or ""
        except (AttributeError, IndexError):
            raw_json = ""

        return LLMResponse(
            data=obj,
            provider=cfg.provider,
            model=cfg.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
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
