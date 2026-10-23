"""Durable resolution jobs retain exact draft/review invariants without live providers."""

from __future__ import annotations

import uuid

import pytest
from sqlmodel import select

from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
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
from app.services.jobs import enqueue_job
from app.workers.resolution_worker import OPERATION_DRAFT, OPERATION_TONE, ResolutionWorker
from tests.test_classification_worker import factory as factory


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


async def _enqueue(factory, cid, *, payload=None, max_attempts=1):
    async with factory() as session:
        case = await session.get(Complaint, cid)
        job = await enqueue_job(
            session, case, JobKind.resolution, payload=payload, max_attempts=max_attempts
        )
        await session.commit()
        return job


@pytest.mark.parametrize(
    "status,guardrail,case_status",
    [
        (STATUS_PASSED, GuardrailStatus.passed, ComplaintStatus.draft_ready),
        (STATUS_FAILED, GuardrailStatus.failed, ComplaintStatus.needs_review),
        (STATUS_UNAVAILABLE, GuardrailStatus.escalated, ComplaintStatus.needs_review),
    ],
)
async def test_draft_and_telemetry_commit_with_completion(factory, status, guardrail, case_status):
    cid = await _seed(factory)
    job = await _enqueue(factory, cid)
    violation = GuardrailViolation(layer="structural", code="too_short", message="Too short.")
    agent = _stub_factory(
        _result(status=status, drafts=2, violations=[violation], feedback="Revise")
    )
    worker = ResolutionWorker(session_factory=factory, agent_factory=agent)
    assert await worker.run_once()
    assert not await worker.run_once()
    async with factory() as session:
        case = await session.get(Complaint, cid)
        draft = (await session.exec(select(Resolution))).one()
        logs = (await session.exec(select(LLMLog))).all()
        stored = await session.get(Job, job.id)
        assert case.status == case_status
        assert case.case_version == 2 and case.input_revision == 1
        assert draft.version == 1 and draft.input_revision == 1
        assert draft.guardrail_status == guardrail
        assert draft.guardrail_violations == [violation.model_dump()]
        assert draft.guardrail_notes == "Revise"
        assert draft.draft_text.startswith("Thank you")
        assert "drafted via groq" in draft.reasoning_summary
        assert sorted(log.operation for log in logs) == [
            OPERATION_DRAFT,
            OPERATION_DRAFT,
            OPERATION_TONE,
        ]
        assert all(log.cost_usd > 0 for log in logs)
        assert stored.status == JobStatus.succeeded
    assert agent.calls[0]["status_at_build"] == ComplaintStatus.escalated


async def test_no_draft_provider_failure_exhausts_to_manual_review(factory):
    cid = await _seed(factory)
    job = await _enqueue(factory, cid)
    worker = ResolutionWorker(
        session_factory=factory,
        agent_factory=_stub_factory(
            _result(
                status=STATUS_UNAVAILABLE,
                with_draft=False,
                drafts=0,
                judges=0,
            )
        ),
    )
    await worker.run_once()
    async with factory() as session:
        assert (await session.exec(select(Resolution))).all() == []
        case = await session.get(Complaint, cid)
        stored = await session.get(Job, job.id)
        assert case.status == ComplaintStatus.needs_review
        assert case.case_version == 1
        assert stored.status == JobStatus.failed
        assert stored.last_error_code == "providers_unavailable"


@pytest.mark.parametrize(
    "status",
    [ComplaintStatus.response_approved, ComplaintStatus.resolved, ComplaintStatus.draft_ready],
)
async def test_feedback_cannot_reopen_reviewed_case(factory, status):
    cid = await _seed(factory, status=status)
    job = await _enqueue(factory, cid, payload={"feedback": "Old rejection"})
    agent = _stub_factory(_result())
    await ResolutionWorker(session_factory=factory, agent_factory=agent).run_once()
    assert agent.calls == []
    async with factory() as session:
        assert (await session.get(Job, job.id)).status == JobStatus.superseded
        assert (await session.exec(select(Resolution))).all() == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"classification_outcome": ClassificationState.unavailable},
        {"classification_stale": True},
        {"sentiment": None, "intent": None, "urgency": None},
        {"sentiment": "furious"},
    ],
)
async def test_unusable_classification_never_runs_agent(factory, overrides):
    cid = await _seed(factory, **overrides)
    job = await _enqueue(factory, cid)
    agent = _stub_factory(_result())
    await ResolutionWorker(session_factory=factory, agent_factory=agent).run_once()
    assert agent.calls == []
    async with factory() as session:
        assert (await session.get(Job, job.id)).status in {JobStatus.failed, JobStatus.superseded}
        assert (await session.exec(select(Resolution))).all() == []


async def test_review_event_feedback_builds_next_revision(factory):
    from app.models.user import User
    from app.services.case_transitions import review_revision

    cid = await _seed(factory)
    await _enqueue(factory, cid)
    await ResolutionWorker(
        session_factory=factory, agent_factory=_stub_factory(_result())
    ).run_once()
    async with factory() as session:
        user = User(
            email="worker-review@example.test", full_name="Reviewer", hashed_password="unused"
        )
        session.add(user)
        await session.flush()
        first = (await session.exec(select(Resolution))).one()
        case, _, event = await review_revision(
            session, cid, first.id, 2, user.id, action="rejected", feedback="Cite the timeline"
        )
        await enqueue_job(
            session,
            case,
            JobKind.resolution,
            review_event_id=event.id,
            payload={"feedback": event.feedback},
        )
        await session.commit()
    agent = _stub_factory(_result())
    await ResolutionWorker(session_factory=factory, agent_factory=agent).run_once()
    async with factory() as session:
        drafts = (await session.exec(select(Resolution).order_by(Resolution.version))).all()
        assert [draft.version for draft in drafts] == [1, 2]
        assert agent.calls[0]["previous_draft_text"] == drafts[0].draft_text
        assert agent.calls[0]["initial_feedback"] == "Cite the timeline"


async def test_transaction_failure_does_not_save_or_complete_draft(factory):
    cid = await _seed(factory)
    job = await _enqueue(factory, cid)

    class BrokenTracker:
        def record(self, *_args, **_kwargs):
            raise RuntimeError("internal secret must not be recorded")

    await ResolutionWorker(
        session_factory=factory, agent_factory=_stub_factory(_result()), tracker=BrokenTracker()
    ).run_once()
    async with factory() as session:
        assert (await session.exec(select(Resolution))).all() == []
        assert (await session.exec(select(LLMLog))).all() == []
        assert (await session.get(Complaint, cid)).case_version == 1
        stored = await session.get(Job, job.id)
        assert stored.status == JobStatus.failed
        assert stored.last_error_code == "worker_error"
