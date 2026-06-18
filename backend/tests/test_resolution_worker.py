"""Tests for the Redis-stream resolution worker.

The worker is driven against real fakeredis streams and a real (SQLite)
session factory, with the agent replaced by a stub factory — so these cover
the worker's own responsibilities (status choreography, transactional
persistence, LLMOps logging, poison/stale handling) without running tools,
guardrails, or LLM calls. The agent's internals are covered in
test_agent_tools.py and test_guardrails.py.
"""

from __future__ import annotations

import uuid

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint, ComplaintStatus
from app.models.llm_log import LLMLog
from app.models.resolution import GuardrailStatus, Resolution
from app.schemas.agent import DraftedResponse
from app.schemas.guardrails import GuardrailViolation, JudgeCallMetadata
from app.services.agent.orchestrator import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    AgentResult,
)
from app.services.agent.tools import DraftOutcome
from app.workers.resolution_worker import (
    OPERATION_DRAFT,
    OPERATION_TONE,
    ResolutionWorker,
    enqueue_resolution,
)


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401 — registers tables on SQLModel.metadata
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture()
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _draft(text: str = "Thank you for reaching out. " + "We will review your dispute. " * 10):
    return DraftedResponse(
        response_text=text,
        tone="empathetic",
        cited_regulations=["Fair Credit Reporting Act (FCRA)"],
        recommended_actions=["Submit a written dispute"],
        confidence=0.8,
    )


def _draft_call() -> DraftOutcome:
    return DraftOutcome(
        drafted=_draft(),
        provider="groq",
        model="llama-3.3-70b-versatile",
        prompt_tokens=900,
        completion_tokens=250,
        latency_ms=1500,
        is_fallback=False,
    )


def _judge_call() -> JudgeCallMetadata:
    return JudgeCallMetadata(
        provider="groq",
        model="llama-3.3-70b-versatile",
        prompt_tokens=400,
        completion_tokens=60,
        latency_ms=800,
    )


def _result(
    *,
    status: str = STATUS_PASSED,
    with_draft: bool = True,
    drafts: int = 1,
    judges: int = 1,
    violations: list[GuardrailViolation] | None = None,
    feedback: str = "",
) -> AgentResult:
    return AgentResult(
        drafted=_draft() if with_draft else None,
        status=status,
        reasoning_steps=["context gathered", "attempt 1: drafted via groq"],
        llm_calls=[_draft_call() for _ in range(drafts)],
        judge_calls=[_judge_call() for _ in range(judges)],
        attempts=drafts,
        guardrail_feedback=feedback,
        guardrail_violations=violations or [],
    )


class _StubAgent:
    def __init__(self, result: AgentResult):
        self._result = result

    async def run(self) -> AgentResult:
        return self._result


def _stub_factory(result: AgentResult):
    """Factory recording each construction so tests can assert what it saw."""
    calls: list[dict] = []

    def factory(complaint, classification, *, initial_feedback=None, previous_draft_text=None):
        calls.append(
            {
                "status_at_build": complaint.status,
                "classification": classification,
                "initial_feedback": initial_feedback,
                "previous_draft_text": previous_draft_text,
            }
        )
        return _StubAgent(result)

    factory.calls = calls
    return factory


async def _seed(factory, **over) -> uuid.UUID:
    fields = {
        "narrative": "They charged me twice for the same payment.",
        "product": "Mortgage",
        "issue": "Trouble during payment",
        "company": "Wells Fargo",
        "sentiment": "extreme_negative",
        "intent": "dispute_resolution",
        "urgency": 5,
        "status": ComplaintStatus.escalated,
    }
    fields.update(over)
    async with factory() as s:
        c = Complaint(**fields)
        s.add(c)
        await s.commit()
        return c.id


def _worker(redis_client, factory, result: AgentResult, agent_factory=None):
    agent_factory = agent_factory or _stub_factory(result)
    return ResolutionWorker(
        redis_client=redis_client,
        session_factory=factory,
        agent_factory=agent_factory,
    )


async def _get_resolutions(factory, cid) -> list[Resolution]:
    async with factory() as s:
        stmt = select(Resolution).where(Resolution.complaint_id == cid)
        return (await s.exec(stmt.order_by(Resolution.version))).all()


class TestProcessMessage:
    async def test_passed_draft_persists_and_marks_draft_ready(self, redis_client, factory):
        cid = await _seed(factory)
        af = _stub_factory(_result())
        w = _worker(redis_client, factory, None, agent_factory=af)
        await w.process_message({"complaint_id": str(cid)})

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.draft_ready
        rows = await _get_resolutions(factory, cid)
        assert len(rows) == 1
        assert rows[0].version == 1
        assert rows[0].guardrail_status == GuardrailStatus.passed
        assert rows[0].guardrail_violations is None
        assert rows[0].draft_text.startswith("Thank you")
        assert "drafted via groq" in rows[0].reasoning_summary
        # The agent was built only after the in-flight status was committed.
        assert af.calls[0]["status_at_build"] == ComplaintStatus.agent_triggered

    async def test_llm_calls_and_judge_calls_logged_in_one_txn(self, redis_client, factory):
        cid = await _seed(factory)
        w = _worker(redis_client, factory, _result(drafts=2, judges=1))
        await w.process_message({"complaint_id": str(cid)})

        async with factory() as s:
            logs = (await s.exec(select(LLMLog))).all()
        assert len(logs) == 3
        ops = sorted(log.operation for log in logs)
        assert ops == [OPERATION_DRAFT, OPERATION_DRAFT, OPERATION_TONE]
        assert all(log.complaint_id == cid for log in logs)
        # Cloud drafting calls are metered: groq model has a pricing entry.
        draft_logs = [log for log in logs if log.operation == OPERATION_DRAFT]
        assert all(log.cost_usd > 0 for log in draft_logs)

    async def test_failed_draft_persists_violations_and_needs_review(self, redis_client, factory):
        cid = await _seed(factory)
        violations = [
            GuardrailViolation(layer="structural", code="too_short", message="Too short."),
            GuardrailViolation(
                layer="regulatory_accuracy", code="ungrounded_citation", message="Fabricated."
            ),
        ]
        w = _worker(
            redis_client,
            factory,
            _result(status=STATUS_FAILED, violations=violations, feedback="fix it"),
        )
        await w.process_message({"complaint_id": str(cid)})

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.needs_review
        rows = await _get_resolutions(factory, cid)
        assert rows[0].guardrail_status == GuardrailStatus.failed
        assert rows[0].guardrail_notes == "fix it"
        # The jsonb round-trip preserves the structured shape, not prose.
        assert rows[0].guardrail_violations == [v.model_dump() for v in violations]

    async def test_unavailable_with_no_draft_writes_no_resolution(self, redis_client, factory):
        cid = await _seed(factory)
        w = _worker(
            redis_client,
            factory,
            _result(status=STATUS_UNAVAILABLE, with_draft=False, drafts=0, judges=0),
        )
        await w.process_message({"complaint_id": str(cid)})

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.needs_review
        assert await _get_resolutions(factory, cid) == []

    async def test_unavailable_with_partial_draft_is_escalated(self, redis_client, factory):
        # Attempt 1 drafted, attempt 2's LLM call died: keep the draft, mark it
        # escalated (no guardrail verdict), and route the complaint to a human.
        cid = await _seed(factory)
        w = _worker(redis_client, factory, _result(status=STATUS_UNAVAILABLE, judges=0))
        await w.process_message({"complaint_id": str(cid)})

        rows = await _get_resolutions(factory, cid)
        assert rows[0].guardrail_status == GuardrailStatus.escalated
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.needs_review

    async def test_feedback_message_builds_next_version_with_context(self, redis_client, factory):
        cid = await _seed(factory)
        w1 = _worker(redis_client, factory, _result())
        await w1.process_message({"complaint_id": str(cid)})  # version 1, draft_ready

        af = _stub_factory(_result())
        w2 = _worker(redis_client, factory, None, agent_factory=af)
        await w2.process_message({"complaint_id": str(cid), "feedback": "Cite the FCRA timeline"})

        rows = await _get_resolutions(factory, cid)
        assert [r.version for r in rows] == [1, 2]
        # The rejected draft + the reviewer's words reached the agent.
        assert af.calls[0]["initial_feedback"] == "Cite the FCRA timeline"
        assert af.calls[0]["previous_draft_text"] == rows[0].draft_text

    async def test_draft_ready_without_feedback_is_stale_and_skipped(self, redis_client, factory):
        cid = await _seed(factory, status=ComplaintStatus.draft_ready)
        af = _stub_factory(_result())
        w = _worker(redis_client, factory, None, agent_factory=af)
        await w.process_message({"complaint_id": str(cid)})

        assert af.calls == []  # agent never built
        assert await _get_resolutions(factory, cid) == []
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.draft_ready  # untouched

    async def test_resolved_complaint_is_never_redrafted(self, redis_client, factory):
        cid = await _seed(factory, status=ComplaintStatus.resolved)
        af = _stub_factory(_result())
        w = _worker(redis_client, factory, None, agent_factory=af)
        await w.process_message({"complaint_id": str(cid)})
        assert af.calls == []

    async def test_unclassified_complaint_is_poison(self, redis_client, factory):
        cid = await _seed(factory, sentiment=None, intent=None, urgency=None)
        af = _stub_factory(_result())
        w = _worker(redis_client, factory, None, agent_factory=af)
        # Returns normally (-> ack); the agent has nothing to work from.
        await w.process_message({"complaint_id": str(cid)})
        assert af.calls == []
        assert await _get_resolutions(factory, cid) == []

    async def test_out_of_vocabulary_classification_is_poison(self, redis_client, factory):
        cid = await _seed(factory, sentiment="furious")  # not in the closed vocab
        w = _worker(redis_client, factory, _result())
        await w.process_message({"complaint_id": str(cid)})
        assert await _get_resolutions(factory, cid) == []

    async def test_bad_uuid_is_poison(self, redis_client, factory):
        w = _worker(redis_client, factory, _result())
        await w.process_message({"complaint_id": "not-a-uuid"})

    async def test_missing_complaint_is_poison(self, redis_client, factory):
        w = _worker(redis_client, factory, _result())
        await w.process_message({"complaint_id": str(uuid.uuid4())})


class TestStreamIntegration:
    async def test_ensure_group_is_idempotent(self, redis_client, factory):
        w = _worker(redis_client, factory, _result())
        await w.ensure_group()
        await w.ensure_group()  # second call must not raise BUSYGROUP

    async def test_enqueue_then_handle_acks(self, redis_client, factory):
        cid = await _seed(factory)
        w = _worker(redis_client, factory, _result())
        await w.ensure_group()
        await enqueue_resolution(redis_client, cid, stream=w.stream)

        resp = await redis_client.xreadgroup(
            groupname=w.group,
            consumername=w.consumer,
            streams={w.stream: ">"},
            count=10,
        )
        _stream, messages = resp[0]
        message_id, fields = messages[0]
        assert fields["complaint_id"] == str(cid)

        await w._handle(message_id, fields)
        summary = await redis_client.xpending(w.stream, w.group)
        assert summary["pending"] == 0

    async def test_enqueue_carries_feedback_field(self, redis_client):
        await enqueue_resolution(redis_client, uuid.uuid4(), feedback="too curt", stream="s")
        entries = await redis_client.xrange("s")
        assert entries[0][1]["feedback"] == "too curt"

    async def test_transient_failure_stays_pending_for_retry(self, redis_client, factory):
        cid = await _seed(factory)

        def _boom_factory(complaint, classification, **kwargs):
            raise RuntimeError("infrastructure hiccup")

        w = _worker(redis_client, factory, None, agent_factory=_boom_factory)
        await w.ensure_group()
        await enqueue_resolution(redis_client, cid, stream=w.stream)

        resp = await redis_client.xreadgroup(
            groupname=w.group,
            consumername=w.consumer,
            streams={w.stream: ">"},
            count=10,
        )
        message_id, fields = resp[0][1][0]
        await w._handle(message_id, fields)  # must swallow, not raise

        summary = await redis_client.xpending(w.stream, w.group)
        assert summary["pending"] == 1  # left for a future claim


class TestPollCadence:
    """Mirror of the classification worker's cadence gate (see its
    TestPollCadence): the PEL sweep runs only every Nth cycle and the read
    honors the configured block window.
    """

    async def test_sweeps_only_every_nth_cycle_and_honors_block(
        self, redis_client, factory, monkeypatch
    ):
        from unittest.mock import AsyncMock

        from app.config import Settings
        from app.workers import resolution_worker as rw

        sweeps = AsyncMock(return_value=0)
        monkeypatch.setattr(rw, "reclaim_stale_messages", sweeps)
        redis_client.xreadgroup = AsyncMock(return_value=[])

        settings = Settings(worker_reclaim_every=2, worker_block_ms=30000)
        w = ResolutionWorker(
            redis_client=redis_client,
            session_factory=factory,
            agent_factory=_stub_factory(_result()),
            settings=settings,
        )

        for cycle in range(4):
            await w._tick(cycle)

        assert sweeps.await_count == 2  # only cycles 0 and 2
        assert redis_client.xreadgroup.await_args.kwargs["block"] == 30000
