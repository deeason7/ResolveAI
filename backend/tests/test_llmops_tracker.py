"""Tests for LLMOps cost accounting and log-row construction."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from app.models.llm_log import LLMLog
from app.services.llmops_tracker import LLMOpsTracker, estimate_cost_usd


class TestEstimateCost:
    def test_known_model_full_mtok(self):
        # 1M input + 1M output at (0.59, 0.79) per Mtok.
        assert estimate_cost_usd("llama-3.3-70b-versatile", 1_000_000, 1_000_000) == 1.38

    def test_gpt_oss_120b_priced(self):
        # openai/gpt-oss-120b — the current cloud model — at (0.15, 0.60) per Mtok.
        assert estimate_cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000) == 0.75

    def test_unknown_model_is_free(self):
        assert estimate_cost_usd("mystery-model", 1000, 1000) == 0.0

    def test_partial_tokens_rounded(self):
        expected = round(1500 / 1_000_000 * 0.59, 6)
        assert estimate_cost_usd("llama-3.3-70b-versatile", 1500, 0) == expected


class TestBuildLog:
    def test_ollama_is_free_even_if_model_priced(self):
        log = LLMOpsTracker().build_log(
            operation="classify",
            provider="ollama",
            model="llama-3.3-70b-versatile",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=10,
            was_fallback=False,
        )
        assert log.cost_usd == 0.0

    def test_groq_call_is_priced(self):
        log = LLMOpsTracker().build_log(
            operation="classify",
            provider="groq",
            model="llama-3.3-70b-versatile",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            latency_ms=10,
            was_fallback=True,
        )
        assert log.cost_usd == 0.59
        assert log.was_fallback is True

    def test_field_mapping(self):
        cid = uuid.uuid4()
        log = LLMOpsTracker().build_log(
            operation="classify",
            provider="groq",
            model="m",
            prompt_tokens=3,
            completion_tokens=4,
            latency_ms=5,
            was_fallback=False,
            complaint_id=cid,
        )
        assert isinstance(log, LLMLog)
        assert log.operation == "classify"
        assert log.model_used == "m"
        assert log.provider == "groq"
        assert log.prompt_tokens == 3
        assert log.completion_tokens == 4
        assert log.latency_ms == 5
        assert log.complaint_id == cid


class TestRecord:
    def test_record_stages_without_committing(self):
        session = MagicMock()
        log = LLMOpsTracker().record(
            session,
            operation="classify",
            provider="ollama",
            model="resolveai-sentiment",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1,
            was_fallback=False,
        )
        session.add.assert_called_once_with(log)
        session.commit.assert_not_called()
