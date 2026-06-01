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
from pydantic import BaseModel

from app.services.llm_client import (
    LLMClient,
    LLMResponse,
    LLMUnavailableError,
    Provider,
    get_llm_client,
    reset_llm_client,
)


def _settings(groq_key: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        llm_timeout_s=5.0,
        classification_max_retries=1,
        ollama_base_url="http://ollama:11434",
        ollama_model="resolveai-sentiment",
        groq_api_key=groq_key,
        groq_base_url="https://api.groq.com/openai/v1",
        groq_model="llama-3.3-70b-versatile",
    )


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


class TestSingleton:
    def test_get_llm_client_is_cached_until_reset(self):
        reset_llm_client()
        a = get_llm_client()
        assert get_llm_client() is a
        reset_llm_client()
        assert get_llm_client() is not a
        reset_llm_client()
