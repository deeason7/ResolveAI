"""Tests for the Ollama->Groq fallback LLM client.

Providers are faked at the instructor-client boundary: we build a real
``LLMClient`` (offline — constructing OpenAI clients makes no network call) and
then swap ``_clients`` so ``structured`` exercises the real chain-walking and
token-extraction logic against controllable fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError
from pydantic import BaseModel

from app.services.llm_client import (
    _MAX_BACKOFF_S,
    LLMClient,
    LLMResponse,
    LLMUnavailableError,
    Provider,
    _estimate_prompt_tokens,
    _retry_after_seconds,
    get_llm_client,
    reset_llm_client,
)


def _settings(
    groq_key: str = "",
    skip_local: bool = False,
    rate_limit_retries: int = 2,
    rate_limit_backoff_s: float = 0.0,
    tpm_limit: int = 0,
) -> SimpleNamespace:
    # rate_limit_backoff_s defaults to 0.0 so tests that hit the no-header path
    # don't actually sleep. tpm_limit defaults to 0 so the token bucket is off
    # and the transport behavior stays identical to the pre-pacing suite.
    return SimpleNamespace(
        llm_timeout_s=5.0,
        classification_max_retries=1,
        ollama_base_url="http://ollama:11434",
        ollama_model="resolveai-sentiment",
        groq_api_key=groq_key,
        groq_base_url="https://api.groq.com/openai/v1",
        groq_model="llama-3.3-70b-versatile",
        llm_skip_local=skip_local,
        llm_rate_limit_retries=rate_limit_retries,
        llm_rate_limit_backoff_s=rate_limit_backoff_s,
        groq_tpm_limit=tpm_limit,
    )


def _rate_limit_error(retry_after: str | None = "0") -> RateLimitError:
    """Build a real openai.RateLimitError carrying a chosen Retry-After header."""
    headers = {} if retry_after is None else {"retry-after": retry_after}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "https://api.groq.com")
    )
    return RateLimitError("rate limited", response=response, body=None)


class _FlakyCompletions:
    """Raise ``exc`` for the first ``fail_times`` calls, then return ``result``."""

    def __init__(self, exc, result, fail_times):
        self._exc = exc
        self._result = result
        self._fail_times = fail_times
        self.calls: list[dict] = []

    def create_with_completion(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self._fail_times:
            raise self._exc
        return self._result


class _FlakyClient:
    def __init__(self, exc, result, fail_times):
        self.chat = SimpleNamespace(completions=_FlakyCompletions(exc, result, fail_times))


class _Dummy(BaseModel):
    x: int


def _completion(prompt: int = 10, completion: int = 5, content: str = '{"x": 1}'):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


class _FakeCompletions:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.calls: list[dict] = []

    def create_with_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeClient:
    def __init__(self, result=None, exc=None):
        self.chat = SimpleNamespace(completions=_FakeCompletions(result, exc))


def _client_with(ollama=None, groq=None, groq_key: str = "gk") -> LLMClient:
    c = LLMClient(settings=_settings(groq_key=groq_key))
    if ollama is not None:
        c._clients[Provider.OLLAMA] = ollama
    if groq is not None and Provider.GROQ in c._clients:
        c._clients[Provider.GROQ] = groq
    return c


class TestBuildChain:
    def test_ollama_only_without_groq_key(self):
        c = LLMClient(settings=_settings(groq_key=""))
        assert [cfg.provider for cfg in c._chain] == [Provider.OLLAMA]

    def test_includes_groq_when_key_present(self):
        c = LLMClient(settings=_settings(groq_key="gk"))
        assert [cfg.provider for cfg in c._chain] == [Provider.OLLAMA, Provider.GROQ]

    def test_ollama_base_url_gets_v1_suffix(self):
        c = LLMClient(settings=_settings())
        assert c._chain[0].base_url.endswith("/v1")

    def test_skip_local_drops_ollama(self):
        c = LLMClient(settings=_settings(groq_key="gk", skip_local=True))
        assert [cfg.provider for cfg in c._chain] == [Provider.GROQ]

    def test_skip_local_makes_groq_primary_not_fallback(self):
        # With Ollama skipped, Groq is at index 0, so a success is NOT a fallback.
        good = _FakeClient(result=(_Dummy(x=9), _completion(4, 2)))
        c = LLMClient(settings=_settings(groq_key="gk", skip_local=True))
        c._clients[Provider.GROQ] = good
        resp = c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert resp.provider == Provider.GROQ
        assert resp.is_fallback is False


class TestStructured:
    def test_primary_success_captures_tokens(self):
        fake = _FakeClient(result=(_Dummy(x=1), _completion(10, 5)))
        c = _client_with(ollama=fake, groq_key="")
        resp = c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.data.x == 1
        assert resp.provider == Provider.OLLAMA
        assert resp.is_fallback is False
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.latency_ms >= 0

    def test_falls_back_to_groq_on_primary_failure(self):
        bad = _FakeClient(exc=httpx.ConnectError("ollama down"))
        good = _FakeClient(result=(_Dummy(x=2), _completion(7, 3)))
        c = _client_with(ollama=bad, groq=good, groq_key="gk")
        resp = c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert resp.data.x == 2
        assert resp.provider == Provider.GROQ
        assert resp.is_fallback is True

    def test_all_providers_failing_raises(self):
        bad = _FakeClient(exc=httpx.ConnectError("down"))
        c = _client_with(ollama=bad, groq_key="")
        with pytest.raises(LLMUnavailableError):
            c.structured(_Dummy, [{"role": "user", "content": "hi"}])

    def test_temperature_defaults_to_deterministic(self):
        fake = _FakeClient(result=(_Dummy(x=1), _completion()))
        c = _client_with(ollama=fake, groq_key="")
        c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert fake.chat.completions.calls[0]["temperature"] == 0.0

    def test_temperature_override_reaches_provider(self):
        # The tone judge runs at 0.1; the override must reach the actual call.
        fake = _FakeClient(result=(_Dummy(x=1), _completion()))
        c = _client_with(ollama=fake, groq_key="")
        c.structured(_Dummy, [{"role": "user", "content": "hi"}], temperature=0.1)
        assert fake.chat.completions.calls[0]["temperature"] == 0.1


class TestRateLimitBackoff:
    def test_retries_same_provider_then_succeeds(self):
        # Two 429s then a success: the client stays on the SAME provider and
        # returns the eventual good result rather than fail-closing.
        flaky = _FlakyClient(
            exc=_rate_limit_error("0"),
            result=(_Dummy(x=7), _completion(3, 1)),
            fail_times=2,
        )
        c = _client_with(ollama=flaky, groq_key="")
        resp = c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert resp.data.x == 7
        assert resp.provider == Provider.OLLAMA
        assert resp.is_fallback is False
        # 2 rate-limited attempts + 1 success, all on the one provider.
        assert len(flaky.chat.completions.calls) == 3

    def test_honors_retry_after_header_duration(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr("app.services.llm_client.time.sleep", lambda s: slept.append(s))
        flaky = _FlakyClient(
            exc=_rate_limit_error("2.5"),
            result=(_Dummy(x=1), _completion()),
            fail_times=1,
        )
        c = _client_with(ollama=flaky, groq_key="")
        c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert slept == [2.5]

    def test_exhausting_budget_falls_back_to_next_provider(self):
        # Primary 429s past its budget (1 retry → 2 attempts), so the chain
        # walks on to Groq instead of looping forever.
        always = _FakeClient(exc=_rate_limit_error("0"))
        good = _FakeClient(result=(_Dummy(x=9), _completion(4, 2)))
        c = LLMClient(settings=_settings(groq_key="gk", rate_limit_retries=1))
        c._clients[Provider.OLLAMA] = always
        c._clients[Provider.GROQ] = good
        resp = c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert resp.provider == Provider.GROQ
        assert resp.is_fallback is True
        assert len(always.chat.completions.calls) == 2  # 1 initial + 1 retry

    def test_exhausting_budget_on_only_provider_raises(self):
        always = _FakeClient(exc=_rate_limit_error("0"))
        c = LLMClient(settings=_settings(groq_key="", rate_limit_retries=1))
        c._clients[Provider.OLLAMA] = always
        with pytest.raises(LLMUnavailableError):
            c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert len(always.chat.completions.calls) == 2


class TestRetryAfter:
    def test_honors_header_value(self):
        assert _retry_after_seconds(_rate_limit_error("3"), default=10.0) == 3.0

    def test_missing_header_uses_default(self):
        assert _retry_after_seconds(_rate_limit_error(None), default=10.0) == 10.0

    def test_unparseable_header_uses_default(self):
        # The HTTP-date form is valid HTTP but we don't decode it -> default.
        err = _rate_limit_error("Wed, 21 Oct 2025 07:28:00 GMT")
        assert _retry_after_seconds(err, default=7.0) == 7.0

    def test_caps_at_max_backoff(self):
        assert _retry_after_seconds(_rate_limit_error("99999"), default=10.0) == _MAX_BACKOFF_S

    def test_negative_clamped_to_zero(self):
        assert _retry_after_seconds(_rate_limit_error("-5"), default=10.0) == 0.0


class TestSingleton:
    def test_get_llm_client_is_cached_until_reset(self):
        reset_llm_client()
        a = get_llm_client()
        assert get_llm_client() is a
        reset_llm_client()
        assert get_llm_client() is not a
        reset_llm_client()


class _RecordingLimiter:
    """A test spy: captures acquire/reconcile calls and never sleeps."""

    def __init__(self) -> None:
        self.acquired: list[float] = []
        self.reconciled: list[float] = []

    def acquire(self, tokens: float) -> float:
        self.acquired.append(tokens)
        return 0.0

    def reconcile(self, delta: float) -> None:
        self.reconciled.append(delta)


class TestTpmPacing:
    def test_no_limiter_when_tpm_disabled(self):
        c = LLMClient(settings=_settings(groq_key="gk", tpm_limit=0))
        assert c._limiter is None

    def test_limiter_built_when_tpm_enabled(self):
        c = LLMClient(settings=_settings(groq_key="gk", tpm_limit=12000))
        assert c._limiter is not None

    def test_groq_call_is_paced_and_reconciled(self):
        lim = _RecordingLimiter()
        good = _FakeClient(result=(_Dummy(x=1), _completion(prompt=10, completion=5)))
        c = LLMClient(settings=_settings(groq_key="gk", skip_local=True), limiter=lim)
        c._clients[Provider.GROQ] = good
        messages = [{"role": "user", "content": "x" * 40}]
        resp = c.structured(_Dummy, messages)
        est = _estimate_prompt_tokens(messages)  # 40/4 + 400 = 410
        assert resp.provider == Provider.GROQ
        assert lim.acquired == [est]  # paced BEFORE the call
        assert lim.reconciled == [(10 + 5) - est]  # trued up with real usage

    def test_local_provider_is_not_paced(self):
        # Ollama has no TPM cap, so the bucket must stay untouched on that path.
        lim = _RecordingLimiter()
        fake = _FakeClient(result=(_Dummy(x=1), _completion()))
        c = LLMClient(settings=_settings(groq_key="", skip_local=False), limiter=lim)
        c._clients[Provider.OLLAMA] = fake
        c.structured(_Dummy, [{"role": "user", "content": "hi"}])
        assert lim.acquired == []
        assert lim.reconciled == []

    def test_estimate_grows_with_message_length(self):
        short = _estimate_prompt_tokens([{"role": "user", "content": "hi"}])
        long = _estimate_prompt_tokens([{"role": "user", "content": "hi" * 100}])
        assert long > short
