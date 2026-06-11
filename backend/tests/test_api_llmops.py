"""Tests for the LLMOps observability endpoints."""

import uuid
from datetime import datetime, timedelta

import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

REGISTER_URL = "/api/v1/auth/register"
COSTS_URL = "/api/v1/llmops/costs"
LATENCY_URL = "/api/v1/llmops/latency"
ROUTING_URL = "/api/v1/llmops/routing"
DRIFT_URL = "/api/v1/llmops/drift"
GUARDRAILS_URL = "/api/v1/llmops/guardrails"

VALID_USER = {
    "email": "llmops@test.com",
    "full_name": "LLMOps Analyst",
    "password": "securepassword123",
}


@pytest_asyncio.fixture()
async def token(client: AsyncClient) -> str:
    r = await client.post(REGISTER_URL, json=VALID_USER)
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _log(**overrides):
    """Unsaved LLMLog with sensible classify-call defaults."""
    from app.models.llm_log import LLMLog

    fields = {
        "operation": "classify",
        "model_used": "llama-3.3-70b-versatile",
        "provider": "groq",
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "latency_ms": 800,
        "cost_usd": 0.001,
        "was_fallback": False,
    }
    fields.update(overrides)
    return LLMLog(**fields)


async def _seed(*rows) -> None:
    from app.database import engine

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        for r in rows:
            s.add(r)
        await s.commit()


class TestCosts:
    async def test_requires_auth(self, client: AsyncClient):
        assert (await client.get(COSTS_URL)).status_code == 401

    async def test_empty_table_returns_zero_totals(self, client: AsyncClient, token: str):
        r = await client.get(COSTS_URL, headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert body["total_cost_usd"] == 0
        assert body["total_calls"] == 0
        assert body["points"] == []

    async def test_groups_by_day_and_provider(self, client: AsyncClient, token: str):
        await _seed(
            _log(cost_usd=0.002),
            _log(cost_usd=0.003),
            _log(provider="ollama", cost_usd=None, prompt_tokens=500),
        )
        r = await client.get(COSTS_URL, headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert body["total_calls"] == 3
        assert body["total_cost_usd"] == 0.005  # None cost coalesces to 0
        by_provider = {p["provider"]: p for p in body["points"]}
        assert by_provider["groq"]["calls"] == 2
        assert by_provider["groq"]["prompt_tokens"] == 2000
        assert by_provider["ollama"]["cost_usd"] == 0

    async def test_window_excludes_old_rows(self, client: AsyncClient, token: str):
        await _seed(
            _log(),
            _log(created_at=datetime.utcnow() - timedelta(days=40)),
        )
        r = await client.get(COSTS_URL, params={"days": 7}, headers=_auth(token))
        assert r.json()["total_calls"] == 1


class TestLatency:
    async def test_requires_auth(self, client: AsyncClient):
        assert (await client.get(LATENCY_URL)).status_code == 401

    async def test_percentiles_per_operation(self, client: AsyncClient, token: str):
        await _seed(*[_log(latency_ms=ms) for ms in range(100, 1100, 100)])  # 100..1000
        await _seed(_log(operation="resolve", latency_ms=5000))
        r = await client.get(LATENCY_URL, headers=_auth(token))
        assert r.status_code == 200
        stats = {i["operation"]: i for i in r.json()["items"]}
        classify = stats["classify"]
        assert classify["calls"] == 10
        assert 500 <= classify["p50_ms"] <= 600
        assert classify["p95_ms"] >= 900
        assert classify["max_ms"] == 1000
        # single-sample operation: p50 == p95 == the value
        assert stats["resolve"]["p50_ms"] == stats["resolve"]["p95_ms"] == 5000

    async def test_null_latency_rows_excluded(self, client: AsyncClient, token: str):
        await _seed(_log(latency_ms=None), _log(latency_ms=300))
        r = await client.get(LATENCY_URL, headers=_auth(token))
        assert r.json()["items"][0]["calls"] == 1


class TestRouting:
    async def test_requires_auth(self, client: AsyncClient):
        assert (await client.get(ROUTING_URL)).status_code == 401

    async def test_splits_provider_and_fallback(self, client: AsyncClient, token: str):
        await _seed(
            _log(provider="ollama", cost_usd=None),
            _log(provider="groq", was_fallback=True, cost_usd=0.002),
            _log(provider="groq", was_fallback=False, cost_usd=0.001),
            _log(provider="none", model_used="none", cost_usd=0.0, latency_ms=0),
        )
        r = await client.get(ROUTING_URL, headers=_auth(token))
        assert r.status_code == 200
        items = {(i["provider"], i["was_fallback"]): i["calls"] for i in r.json()["items"]}
        assert items[("ollama", False)] == 1
        assert items[("groq", True)] == 1
        assert items[("groq", False)] == 1
        assert items[("none", False)] == 1  # deterministic fail-closed path


class TestDrift:
    async def test_requires_auth(self, client: AsyncClient):
        assert (await client.get(DRIFT_URL)).status_code == 401

    async def test_joins_classify_calls_to_current_labels(self, client: AsyncClient, token: str):
        from app.models.complaint import Complaint

        negative = Complaint(narrative="n" * 30, sentiment="negative")
        extreme = Complaint(narrative="e" * 30, sentiment="extreme_negative")
        unlabeled = Complaint(narrative="u" * 30)  # classify logged, label never landed
        await _seed(negative, extreme, unlabeled)
        await _seed(
            _log(complaint_id=negative.id),
            _log(complaint_id=extreme.id),
            _log(complaint_id=extreme.id),  # re-classified: counts per event
            _log(complaint_id=unlabeled.id),
            _log(complaint_id=negative.id, operation="resolve"),  # non-classify excluded
        )
        r = await client.get(DRIFT_URL, headers=_auth(token))
        assert r.status_code == 200
        counts = {p["sentiment"]: p["count"] for p in r.json()["points"]}
        assert counts == {"negative": 1, "extreme_negative": 2}


def _resolution(violations, **overrides):
    from app.models.resolution import GuardrailStatus, Resolution

    fields = {
        "complaint_id": uuid.uuid4(),
        "draft_text": "Dear customer, we hear you." * 5,
        "guardrail_status": GuardrailStatus.failed,
        "guardrail_violations": violations,
    }
    fields.update(overrides)
    return Resolution(**fields)


class TestGuardrailLog:
    async def test_requires_auth(self, client: AsyncClient):
        assert (await client.get(GUARDRAILS_URL)).status_code == 401

    async def test_flattens_violations_newest_first(self, client: AsyncClient, token: str):
        old = _resolution(
            [{"layer": "structural", "code": "too_short", "message": "Draft too short"}],
            created_at=datetime.utcnow() - timedelta(days=1),
        )
        new = _resolution(
            [
                {"layer": "tone", "code": "empathy_low", "message": "Score 4/10"},
                {
                    "layer": "content_safety",
                    "code": "forbidden_promise",
                    "message": "Promised refund",
                },
            ]
        )
        passed = _resolution(None)  # no violations → not in the log
        await _seed(old, new, passed)

        r = await client.get(GUARDRAILS_URL, headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert body["total_violations"] == 3
        assert [i["code"] for i in body["items"]] == [
            "empathy_low",
            "forbidden_promise",
            "too_short",
        ]

    async def test_layer_filter_and_limit(self, client: AsyncClient, token: str):
        await _seed(
            _resolution(
                [
                    {"layer": "tone", "code": "a", "message": "m"},
                    {"layer": "structural", "code": "b", "message": "m"},
                    {"layer": "tone", "code": "c", "message": "m"},
                ]
            )
        )
        r = await client.get(GUARDRAILS_URL, params={"layer": "tone"}, headers=_auth(token))
        body = r.json()
        assert body["total_violations"] == 2
        assert all(i["layer"] == "tone" for i in body["items"])

        r = await client.get(
            GUARDRAILS_URL, params={"layer": "tone", "limit": 1}, headers=_auth(token)
        )
        body = r.json()
        assert len(body["items"]) == 1
        assert body["total_violations"] == 2  # total counts beyond the page
