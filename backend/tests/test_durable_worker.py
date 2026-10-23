"""Exercise the running lease loop with deterministic provider and clock barriers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from neo4j.exceptions import AuthError, CypherSyntaxError, ServiceUnavailable
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import IntegrityError, OperationalError

from app.config import Settings
from app.models.job import Job, JobKind, JobStatus
from app.services import jobs
from app.services.jobs import complete_job, lock_owned_job
from app.workers import durable
from app.workers.durable import DurableWorker
from tests.test_classification_worker import _seed
from tests.test_classification_worker import factory as factory


class ControlledWorker(DurableWorker):
    kind = JobKind.classification

    def __init__(self, process, **kwargs):
        super().__init__(**kwargs)
        self.process = process

    async def process_job(self, job):
        await self.process(job)


async def test_cancelled_execution_keeps_claim_until_expiry_then_recovers(factory):
    _, job_id = await _seed(factory, max_attempts=3)
    entered = asyncio.Event()
    tokens = []

    async def interrupted_provider(job):
        tokens.append(job.lease_token)
        entered.set()
        await asyncio.Event().wait()

    interrupted = ControlledWorker(interrupted_provider, session_factory=factory)
    execution = asyncio.create_task(interrupted.run_once())
    await asyncio.wait_for(entered.wait(), 5)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    async with factory() as session:
        stored = await session.get(Job, job_id)
        assert stored.status == JobStatus.running
        assert stored.attempts == 1
        assert stored.lease_token == tokens[0]
        assert stored.last_error_code is None

    persisted = []

    async def replacement_provider(job):
        async with factory() as session:
            owned = await lock_owned_job(session, job.id, job.lease_token)
            assert owned is not None
            persisted.append(job.lease_token)
            complete_job(owned)
            await session.commit()

    replacement = ControlledWorker(replacement_provider, session_factory=factory)
    assert not await replacement.run_once(), "A live abandoned lease cannot be stolen early"
    async with factory() as session:
        stored = await session.get(Job, job_id)
        stored.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()
    assert await replacement.run_once()
    assert not await replacement.run_once()
    async with factory() as session:
        stored = await session.get(Job, job_id)
        assert stored.status == JobStatus.succeeded
        assert stored.attempts == stored.total_attempts == 2
        assert stored.lease_token is None
    assert len(persisted) == 1 and persisted[0] != tokens[0]


async def test_actual_heartbeat_renews_lease_while_provider_is_paused(factory, monkeypatch):
    _, job_id = await _seed(factory, max_attempts=3)
    entered, provider_release = asyncio.Event(), asyncio.Event()
    timer_entered, timer_release, renewed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clock = [datetime.utcnow()]
    snapshots = []

    async def database_now(_session):
        return clock[0]

    # Advance database time deliberately; the heartbeat timer is released by an
    # event instead of relying on wall-clock scheduling to win a race.
    monkeypatch.setattr(jobs, "database_now", database_now)

    async def controlled_timer(seconds):
        assert seconds == 30
        timer_entered.set()
        await timer_release.wait()
        timer_release.clear()

    monkeypatch.setattr(
        durable,
        "asyncio",
        SimpleNamespace(
            Event=asyncio.Event,
            create_task=asyncio.create_task,
            wait_for=asyncio.wait_for,
            CancelledError=asyncio.CancelledError,
            sleep=controlled_timer,
        ),
    )
    actual_heartbeat = jobs.heartbeat_job

    async def observed_heartbeat(session, *args, **kwargs):
        result = await actual_heartbeat(session, *args, **kwargs)
        session.info["renewal_written"] = result
        return result

    monkeypatch.setattr(durable, "heartbeat_job", observed_heartbeat)

    @asynccontextmanager
    async def observed_sessions():
        async with factory() as session:
            yield session
            # This runs after the runner's commit, so renewal is observable by
            # an independent session rather than just its identity map.
            if session.info.get("renewal_written"):
                renewed.set()

    async def paused_provider(job):
        snapshots.append((job.lease_token, job.lease_expires_at))
        entered.set()
        await provider_release.wait()
        async with factory() as session:
            owned = await lock_owned_job(session, job.id, job.lease_token)
            assert owned is not None
            complete_job(owned)
            await session.commit()

    worker = ControlledWorker(
        paused_provider,
        session_factory=observed_sessions,
        settings=Settings(job_lease_seconds=120, job_heartbeat_seconds=30),
    )
    execution = asyncio.create_task(worker.run_once())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(timer_entered.wait(), 5)
        clock[0] += timedelta(seconds=30)
        timer_release.set()
        await asyncio.wait_for(renewed.wait(), 5)
        async with factory() as session:
            stored = await session.get(Job, job_id)
            assert stored.status == JobStatus.running
            assert stored.lease_token == snapshots[0][0]
            assert stored.heartbeat_at == clock[0]
            assert stored.lease_expires_at == snapshots[0][1] + timedelta(seconds=30)
        assert not execution.done(), "The lease was renewed while the provider was still running"
    finally:
        provider_release.set()
        await asyncio.wait_for(execution, 5)
    async with factory() as session:
        assert (await session.get(Job, job_id)).status == JobStatus.succeeded


@pytest.mark.parametrize(
    "error,code",
    [
        (TimeoutError("private upstream detail"), "dependency_unavailable"),
        (ConnectionError("private upstream detail"), "dependency_unavailable"),
        (httpx.ConnectTimeout("private upstream detail"), "dependency_unavailable"),
        (
            ResponseHandlingException(httpx.ReadError("private upstream detail")),
            "dependency_unavailable",
        ),
        (
            UnexpectedResponse(503, "Unavailable", b"private upstream detail", httpx.Headers()),
            "dependency_unavailable",
        ),
        (ServiceUnavailable("private upstream detail"), "dependency_unavailable"),
        (
            httpx.HTTPStatusError(
                "private upstream detail",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(429),
            ),
            "dependency_unavailable",
        ),
        (
            OperationalError("synthetic query", {}, Exception("private upstream detail")),
            "database_unavailable",
        ),
    ],
)
async def test_transient_failures_retry_with_backoff_then_stop_at_budget(
    factory, caplog, error, code
):
    _, job_id = await _seed(factory, max_attempts=2)
    calls = []

    async def failing_provider(job):
        calls.append(job.id)
        raise error

    worker = ControlledWorker(failing_provider, session_factory=factory)
    assert await worker.run_once()
    async with factory() as session:
        stored = await session.get(Job, job_id)
        assert stored.status == JobStatus.queued
        assert stored.attempts == 1
        assert stored.next_attempt_at > stored.updated_at
        assert stored.last_error_code == code
        assert stored.lease_token is None
    assert not await worker.run_once(), "Backoff must postpone the second call"
    async with factory() as session:
        stored = await session.get(Job, job_id)
        stored.next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()
    assert await worker.run_once()
    assert not await worker.run_once()
    async with factory() as session:
        stored = await session.get(Job, job_id)
        assert stored.status == JobStatus.failed
        assert stored.attempts == stored.total_attempts == 2
        assert stored.last_error_code == code
        assert stored.completed_at is not None
    assert len(calls) == 2
    assert "private upstream detail" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ValueError("private invalid input"),
        RuntimeError("private invalid input"),
        ResponseHandlingException(ValueError("private invalid input")),
        UnexpectedResponse(400, "Bad request", b"private invalid input", httpx.Headers()),
        AuthError("private invalid input"),
        CypherSyntaxError("private invalid input"),
        httpx.HTTPStatusError(
            "private invalid input",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(401),
        ),
        IntegrityError("synthetic query", {}, Exception("private invalid input")),
    ],
)
async def test_permanent_failure_is_visible_without_spending_remaining_attempts(
    factory, caplog, error
):
    _, job_id = await _seed(factory, max_attempts=3)
    calls = []

    async def broken_provider(job):
        calls.append(job.id)
        raise error

    worker = ControlledWorker(broken_provider, session_factory=factory)
    assert await worker.run_once()
    assert not await worker.run_once()
    async with factory() as session:
        stored = await session.get(Job, job_id)
        assert stored.status == JobStatus.failed
        assert stored.attempts == stored.total_attempts == 1
        assert stored.completed_at is not None
        assert stored.lease_token is None
        assert stored.last_error_code in {
            "worker_error",
            "database_unavailable",
            "dependency_unavailable",
        }
    assert len(calls) == 1
    assert "private invalid input" not in caplog.text
