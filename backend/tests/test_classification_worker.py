"""Classification output, telemetry and follow-up intents share one transaction."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.models.llm_log import LLMLog
from app.schemas.classification import ComplaintClassification
from app.services.classifier import ClassificationOutcome
from app.services.jobs import enqueue_job
from app.workers.classification_worker import (
    ClassificationWorker,
    compute_priority_score,
    is_high_priority,
)


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


def _outcome(*, sentiment="negative", urgency=3, succeeded=True, is_fallback=False):
    cls = ComplaintClassification(
        sentiment=sentiment,
        intent="dispute_resolution",
        urgency=urgency,
        key_entities=[],
        reasoning="model reasoning text here",
    )
    return ClassificationOutcome(
        classification=cls if succeeded else None,
        provider=("groq" if is_fallback else "ollama") if succeeded else "none",
        model="resolveai-sentiment" if succeeded else "none",
        prompt_tokens=10 if succeeded else None,
        completion_tokens=5 if succeeded else None,
        latency_ms=100,
        is_fallback=is_fallback,
        succeeded=succeeded,
    )


class _StubClassifier:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def classify(self, narrative, **_kwargs):
        self.calls.append(narrative)
        return self.outcome


async def _seed(factory, *, max_attempts=1, **overrides):
    async with factory() as session:
        case = Complaint(
            narrative="Synthetic fee dispute",
            company="Chase",
            product="Credit card",
            issue="Fees",
            **overrides,
        )
        session.add(case)
        job = await enqueue_job(session, case, JobKind.classification, max_attempts=max_attempts)
        await session.commit()
        return case.id, job.id


def test_priority_helpers():
    assert compute_priority_score(5, "extreme_negative") == 1.0
    assert compute_priority_score(1, "neutral") == 0.0
    assert is_high_priority(4, "neutral")
    assert is_high_priority(1, "extreme_negative")
    assert not is_high_priority(2, "negative")


@pytest.mark.parametrize("high,fallback", [(False, False), (True, False), (False, True)])
async def test_classification_commits_telemetry_and_downstream_intents(factory, high, fallback):
    cid, jid = await _seed(factory)
    classifier = _StubClassifier(_outcome(urgency=5 if high else 3, is_fallback=fallback))
    worker = ClassificationWorker(classifier, session_factory=factory)
    assert await worker.run_once()
    assert not await worker.run_once()
    assert len(classifier.calls) == 1
    async with factory() as session:
        case = await session.get(Complaint, cid)
        assert case.status == (ComplaintStatus.escalated if high else ComplaintStatus.classified)
        assert case.classification_outcome == ClassificationState.succeeded
        assert case.classification_source == ("groq" if fallback else "ollama")
        assert case.classification_was_fallback == fallback
        assert case.classification_stale is False
        assert case.case_version == case.input_revision == 2
        assert case.classified_at is not None
        logs = (await session.exec(select(LLMLog))).all()
        assert len(logs) == 1 and logs[0].operation == "classify"
        jobs = (await session.exec(select(Job))).all()
        assert (await session.get(Job, jid)).status == JobStatus.succeeded
        followups = [job for job in jobs if job.id != jid]
        assert {job.kind for job in followups} == (
            {JobKind.index, JobKind.resolution} if high else {JobKind.index}
        )
        assert all(
            job.status == JobStatus.queued and job.case_version == job.input_revision == 2
            for job in followups
        )


async def test_provider_failure_retries_then_exhausts_visibly(factory):
    cid, jid = await _seed(factory, max_attempts=2)
    classifier = _StubClassifier(_outcome(succeeded=False))
    worker = ClassificationWorker(classifier, session_factory=factory)
    await worker.run_once()
    async with factory() as session:
        job = await session.get(Job, jid)
        assert job.status == JobStatus.queued
        assert job.attempts == 1
        job.next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()
    await worker.run_once()
    assert not await worker.run_once()
    async with factory() as session:
        case = await session.get(Complaint, cid)
        job = await session.get(Job, jid)
        assert job.status == JobStatus.failed and job.attempts == 2
        assert job.last_error_code == "providers_unavailable"
        assert case.status == ComplaintStatus.needs_review
        assert case.classification_outcome == ClassificationState.unavailable
        assert case.sentiment is case.intent is case.urgency is None
        assert case.classification_source is None
        assert case.case_version == case.input_revision == 1
        logs = (await session.exec(select(LLMLog))).all()
        assert len(logs) == 2
        assert all(log.operation == "classify_unavailable" and log.cost_usd is None for log in logs)
        assert len((await session.exec(select(Job))).all()) == 1


async def test_failed_reclassification_retains_provenance(factory):
    cid, _ = await _seed(
        factory,
        status=ComplaintStatus.classified,
        sentiment="neutral",
        intent="information_request",
        urgency=1,
        classification_source="groq",
        classification_was_fallback=True,
        classification_outcome=ClassificationState.succeeded,
    )
    await ClassificationWorker(
        _StubClassifier(_outcome(succeeded=False)), session_factory=factory
    ).run_once()
    async with factory() as session:
        case = await session.get(Complaint, cid)
        assert (case.sentiment, case.intent, case.urgency) == ("neutral", "information_request", 1)
        assert case.classification_source == "groq" and case.classification_was_fallback
        assert case.classification_stale


@pytest.mark.parametrize("manual", [False, True])
async def test_approved_or_manual_case_supersedes_trigger(factory, manual):
    cid, jid = await _seed(
        factory,
        status=ComplaintStatus.classified if manual else ComplaintStatus.response_approved,
        classification_source="manual" if manual else "ollama",
    )
    classifier = _StubClassifier(_outcome())
    await ClassificationWorker(classifier, session_factory=factory).run_once()
    assert classifier.calls == []
    async with factory() as session:
        assert (await session.get(Job, jid)).status == JobStatus.superseded
        assert (await session.exec(select(LLMLog))).all() == []


async def test_failed_followup_insert_rolls_back_result_and_telemetry(factory, monkeypatch):
    from app.workers import classification_worker

    cid, jid = await _seed(factory)

    async def fail(*_a, **_kw):
        raise ConnectionError("private provider detail")

    monkeypatch.setattr(classification_worker, "enqueue_job", fail)
    await ClassificationWorker(
        _StubClassifier(_outcome(urgency=5)), session_factory=factory
    ).run_once()
    async with factory() as session:
        assert (await session.get(Complaint, cid)).case_version == 1
        assert (await session.exec(select(LLMLog))).all() == []
        job = await session.get(Job, jid)
        assert job.status == JobStatus.failed
        assert job.last_error_code == "dependency_unavailable"
