"""Exact review and worker fencing with independent PostgreSQL transactions."""

import asyncio
import hashlib

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlmodel import select

from app.database import get_session
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.resolution import GuardrailStatus, Resolution
from app.models.review_event import ReviewEvent
from app.models.user import User, UserRole
from app.services.case_transitions import advance_case, lock_case
from app.workers.resolution_worker import ResolutionWorker
from tests.test_api_resolutions import _DUMMY_USER, _build_app
from tests.test_resolution_worker import _result, _stub_factory

pytestmark = pytest.mark.postgres


async def seed(factory, *, draft=True):
    async with factory() as session:
        user = User(
            id=_DUMMY_USER.id,
            email=_DUMMY_USER.email,
            full_name="Synthetic reviewer",
            hashed_password="unused",
            role=UserRole.analyst,
        )
        case = Complaint(
            narrative="Synthetic double charge dispute",
            sentiment="negative",
            intent="dispute_resolution",
            urgency=4,
            classification_outcome=ClassificationState.succeeded,
            status=ComplaintStatus.draft_ready if draft else ComplaintStatus.escalated,
        )
        session.add_all([user, case])
        await session.flush()
        resolution = None
        if draft:
            resolution = Resolution(
                complaint_id=case.id,
                draft_text="Exact saved response.",
                guardrail_status=GuardrailStatus.passed,
            )
            session.add(resolution)
        await session.commit()
        return case, resolution


def client(factory, redis):
    app = _build_app(factory, redis)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize(
    "actions", [("approve", "approve"), ("approve", "reject"), ("reject", "reject")]
)
async def test_competing_reviewers_have_one_success_one_actionable_conflict(
    pg_sessionmaker, actions
):
    case, draft = await seed(pg_sessionmaker)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app = _build_app(pg_sessionmaker, redis)
    barrier = asyncio.Barrier(2)
    connections = set()

    async def independent_sessions():
        async with pg_sessionmaker() as session:
            connections.add(await session.scalar(text("SELECT pg_backend_pid()")))
            await asyncio.wait_for(barrier.wait(), 5)
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = independent_sessions
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:

        async def review(action):
            body = {"resolution_id": str(draft.id), "expected_case_version": 1}
            if action == "reject":
                body["feedback"] = "Please explain the next step clearly."
            return await ac.post(f"/api/v1/resolutions/{case.id}/{action}", json=body)

        responses = await asyncio.wait_for(
            asyncio.gather(*(review(action) for action in actions)), 10
        )
    assert len(connections) == 2
    assert len([r for r in responses if r.status_code in (200, 202)]) == 1
    conflicts = [r for r in responses if r.status_code == 409]
    assert len(conflicts) == 1
    assert "Reload" in conflicts[0].json()["detail"]
    async with pg_sessionmaker() as session:
        events = (await session.exec(select(ReviewEvent))).all()
        assert len(events) == 1
        event = events[0]
        assert (event.expected_case_version, event.case_version, event.input_revision) == (1, 2, 1)
        assert event.resolution_id == draft.id
        assert event.draft_sha256 == hashlib.sha256(draft.draft_text.encode()).hexdigest()
        assert (await session.get(Complaint, case.id)).case_version == 2


async def test_duplicate_generation_requests_admit_only_one(pg_sessionmaker):
    case, _ = await seed(pg_sessionmaker, draft=False)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ready = asyncio.Barrier(2)
    async with client(pg_sessionmaker, redis) as ac:

        async def generate():
            await ready.wait()
            return await ac.post(f"/api/v1/resolutions/{case.id}/generate")

        results = await asyncio.wait_for(asyncio.gather(generate(), generate()), 10)
    assert sorted(r.status_code for r in results) == [202, 409]
    from app.config import settings

    assert await redis.xlen(settings.resolution_queue) == 1


async def test_paused_duplicate_result_cannot_overwrite_a_later_approval(pg_sessionmaker):
    case, _ = await seed(pg_sessionmaker, draft=False)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    started, release = asyncio.Event(), asyncio.Event()

    class PausedAgent:
        async def run(self):
            started.set()
            await asyncio.wait_for(release.wait(), 10)
            return _result()

    slow = ResolutionWorker(
        redis,
        session_factory=pg_sessionmaker,
        agent_factory=lambda *_args, **_kwargs: PausedAgent(),
    )
    fast = ResolutionWorker(
        redis, session_factory=pg_sessionmaker, agent_factory=_stub_factory(_result())
    )
    message = {"complaint_id": str(case.id), "expected_case_version": "1", "input_revision": "1"}
    task = asyncio.create_task(slow.process_message(message))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.wait_for(fast.process_message(message), 5)
        async with client(pg_sessionmaker, redis) as ac:
            draft = (await ac.get(f"/api/v1/resolutions/{case.id}")).json()
            response = await ac.post(
                f"/api/v1/resolutions/{case.id}/approve",
                json={"resolution_id": draft["id"], "expected_case_version": draft["case_version"]},
            )
            assert response.status_code == 200, response.text
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    # Even a replay carrying rejection feedback must not reopen approved text.
    await slow.process_message({**message, "feedback": "Old rejected draft feedback"})
    async with pg_sessionmaker() as session:
        stored = await session.get(Complaint, case.id)
        assert stored.status == ComplaintStatus.response_approved
        assert stored.case_version == 3
        assert len((await session.exec(select(Resolution))).all()) == 1
        assert len((await session.exec(select(ReviewEvent))).all()) == 1


async def test_concurrent_worker_results_allocate_one_revision(pg_sessionmaker):
    case, _ = await seed(pg_sessionmaker, draft=False)
    barrier = asyncio.Barrier(2)

    class SimultaneousAgent:
        async def run(self):
            await asyncio.wait_for(barrier.wait(), 5)
            return _result()

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    workers = [
        ResolutionWorker(
            redis,
            session_factory=pg_sessionmaker,
            agent_factory=lambda *_a, **_k: SimultaneousAgent(),
        )
        for _ in range(2)
    ]
    message = {"complaint_id": str(case.id), "expected_case_version": "1", "input_revision": "1"}
    await asyncio.wait_for(asyncio.gather(*(w.process_message(message) for w in workers)), 10)
    async with pg_sessionmaker() as session:
        drafts = (await session.exec(select(Resolution))).all()
        assert len(drafts) == 1
        assert drafts[0].version == 1
        assert (await session.get(Complaint, case.id)).case_version == 2


async def test_input_change_fences_slow_resolution_without_a_status_change(pg_sessionmaker):
    case, _ = await seed(pg_sessionmaker, draft=False)
    entered, release = asyncio.Event(), asyncio.Event()

    class PausedAgent:
        async def run(self):
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            return _result()

    worker = ResolutionWorker(
        fakeredis.aioredis.FakeRedis(),
        session_factory=pg_sessionmaker,
        agent_factory=lambda *_a, **_k: PausedAgent(),
    )
    task = asyncio.create_task(
        worker.process_message(
            {"complaint_id": str(case.id), "expected_case_version": "1", "input_revision": "1"}
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with pg_sessionmaker() as session:
            changed = await lock_case(session, case.id)
            changed.narrative = "Updated synthetic input"
            advance_case(changed, inputs_changed=True)
            session.add(changed)
            await session.commit()
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    async with pg_sessionmaker() as session:
        assert (await session.exec(select(Resolution))).all() == []
        stored = await session.get(Complaint, case.id)
        assert stored.case_version == stored.input_revision == 2


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE review_events SET feedback = 'changed'",
        "DELETE FROM review_events",
        "UPDATE resolutions SET draft_text = 'changed'",
        "DELETE FROM resolutions",
    ],
)
async def test_persisted_review_and_text_reject_raw_sql_mutation(pg_sessionmaker, statement):
    case, draft = await seed(pg_sessionmaker)
    async with client(pg_sessionmaker, fakeredis.aioredis.FakeRedis()) as ac:
        result = await ac.post(
            f"/api/v1/resolutions/{case.id}/approve",
            json={"resolution_id": str(draft.id), "expected_case_version": 1},
        )
        assert result.status_code == 200
    async with pg_sessionmaker() as session:
        with pytest.raises(DBAPIError):
            await session.exec(text(statement))
        await session.rollback()
        assert (await session.get(Resolution, draft.id)).draft_text == draft.draft_text
        assert len((await session.exec(select(ReviewEvent))).all()) == 1


async def test_downgrade_refuses_to_erase_review_history(postgres_database, pg_sessionmaker):
    case, draft = await seed(pg_sessionmaker)
    async with client(pg_sessionmaker, fakeredis.aioredis.FakeRedis()) as ac:
        response = await ac.post(
            f"/api/v1/resolutions/{case.id}/approve",
            json={"resolution_id": str(draft.id), "expected_case_version": 1},
        )
        assert response.status_code == 200
    with pytest.raises(pytest.fail.Exception, match="Refusing to remove review history"):
        await postgres_database.downgrade("c4a8e2d9f601")
    async with pg_sessionmaker() as session:
        assert len((await session.exec(select(ReviewEvent))).all()) == 1
        assert (
            await session.scalar(text("SELECT version_num FROM alembic_version")) == "e6b4a9d2c705"
        )
