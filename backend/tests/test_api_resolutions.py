"""Tests for the resolution API routes.

Real (SQLite) database via the session override, fakeredis behind the
get_redis seam, auth bypassed via get_current_user — so these exercise status
gating, the queue handoff, version selection, and the audit trail without a
worker or LLM anywhere. The worker side of the contract lives in
test_resolution_worker.py.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.models.audit_log import AuditLog
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind
from app.models.resolution import GuardrailStatus, Resolution
from app.models.user import User, UserRole

GENERATE_URL = "/api/v1/resolutions/{id}/generate"
LATEST_URL = "/api/v1/resolutions/{id}"
REVISIONS_URL = "/api/v1/resolutions/{id}/revisions"
APPROVE_URL = "/api/v1/resolutions/{id}/approve"
REJECT_URL = "/api/v1/resolutions/{id}/reject"

_DUMMY_USER = SimpleNamespace(id=uuid.uuid4(), email="analyst@test.com", role=UserRole.analyst)
_VIEWER_USER = SimpleNamespace(id=uuid.uuid4(), email="viewer@test.com", role=UserRole.viewer)


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401 — registers tables on SQLModel.metadata
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    # The auth override still represents persisted users: review events reference
    # their IDs through a real foreign key.
    async with sessions() as session:
        for user in (_DUMMY_USER, _VIEWER_USER):
            session.add(
                User(
                    id=user.id,
                    email=user.email,
                    full_name="Synthetic reviewer",
                    hashed_password="unused",
                    role=user.role,
                )
            )
        await session.commit()
    yield sessions


@pytest.fixture()
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _build_app(factory, fake_redis, *, bypass_auth=True, user=_DUMMY_USER):
    from app.core.deps import get_current_user, get_redis
    from app.database import get_session
    from app.main import create_app

    async def override_session():
        async with factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = lambda: fake_redis
    if bypass_auth:
        app.dependency_overrides[get_current_user] = lambda: user
    return app


@pytest_asyncio.fixture()
async def client(factory, fake_redis):
    app = _build_app(factory, fake_redis)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def _seed_complaint(factory, **over) -> uuid.UUID:
    fields = {
        "narrative": "They charged me twice for the same payment.",
        "product": "Mortgage",
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


async def _seed_resolution(factory, complaint_id, **over) -> uuid.UUID:
    fields = {
        "complaint_id": complaint_id,
        "version": 1,
        "draft_text": "Thank you for reaching out about the duplicate charge...",
        "guardrail_status": GuardrailStatus.passed,
    }
    fields.update(over)
    async with factory() as s:
        r = Resolution(**fields)
        s.add(r)
        await s.commit()
        return r.id


async def _audit_events(factory) -> list[AuditLog]:
    async with factory() as s:
        return (await s.exec(select(AuditLog))).all()


async def _identity(client, cid):
    response = await client.get(LATEST_URL.format(id=cid))
    if response.status_code == 404:
        return {"resolution_id": str(uuid.uuid4()), "expected_case_version": 1}
    draft = response.json()
    return {"resolution_id": draft["id"], "expected_case_version": draft["case_version"]}


class TestGenerate:
    @pytest.mark.parametrize("stale", [False, True])
    async def test_retained_labels_after_outage_are_not_drafting_input(
        self, client, factory, fake_redis, stale
    ):
        cid = await _seed_complaint(
            factory,
            classification_outcome=(
                ClassificationState.succeeded if stale else ClassificationState.unavailable
            ),
            classification_stale=stale,
        )
        response = await client.post(GENERATE_URL.format(id=cid))
        assert response.status_code == 409
        assert "current classification" in response.json()["detail"]
        assert await fake_redis.xrange(settings.resolution_queue) == []

    async def test_queues_and_marks_in_flight(self, client, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.classified)
        r = await client.post(GENERATE_URL.format(id=cid))

        assert r.status_code == 202, r.text
        assert r.json()["complaint_id"] == str(cid)
        assert r.json()["status"] == "queued"
        assert r.json()["job_id"]
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.agent_triggered.value
        async with factory() as session:
            job = (await session.exec(select(Job))).one()
            assert job.complaint_id == cid
            assert job.kind == JobKind.resolution

    async def test_unknown_complaint_is_404(self, client):
        r = await client.post(GENERATE_URL.format(id=uuid.uuid4()))
        assert r.status_code == 404

    async def test_unclassified_complaint_is_409(self, client, factory):
        cid = await _seed_complaint(
            factory, sentiment=None, intent=None, urgency=None, status=ComplaintStatus.pending
        )
        r = await client.post(GENERATE_URL.format(id=cid))
        assert r.status_code == 409
        assert "classification" in r.json()["detail"]

    async def test_in_flight_is_409(self, client, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.agent_triggered)
        r = await client.post(GENERATE_URL.format(id=cid))
        assert r.status_code == 409
        assert await fake_redis.xrange(settings.resolution_queue) == []  # nothing queued

    async def test_draft_ready_is_409(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        r = await client.post(GENERATE_URL.format(id=cid))
        assert r.status_code == 409
        assert "reject" in r.json()["detail"]

    async def test_resolved_is_409(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.resolved)
        r = await client.post(GENERATE_URL.format(id=cid))
        assert r.status_code == 409


class TestReads:
    async def test_latest_returns_newest_version(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid, version=1, draft_text="first try")
        violations = [{"layer": "tone", "code": "low_empathy", "message": "Score 4 < 6"}]
        await _seed_resolution(
            factory,
            cid,
            version=2,
            draft_text="second try",
            guardrail_status=GuardrailStatus.failed,
            guardrail_violations=violations,
        )

        r = await client.get(LATEST_URL.format(id=cid))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["version"] == 2
        assert body["draft_text"] == "second try"
        assert body["guardrail_status"] == "failed"
        # jsonb dicts come back as typed GuardrailViolation objects
        assert body["guardrail_violations"] == violations

    async def test_no_resolution_is_404(self, client, factory):
        cid = await _seed_complaint(factory)
        r = await client.get(LATEST_URL.format(id=cid))
        assert r.status_code == 404
        assert "No resolution" in r.json()["detail"]

    async def test_unknown_complaint_is_404(self, client):
        r = await client.get(LATEST_URL.format(id=uuid.uuid4()))
        assert r.status_code == 404

    async def test_revisions_newest_first(self, client, factory):
        cid = await _seed_complaint(factory)
        await _seed_resolution(factory, cid, version=1)
        await _seed_resolution(factory, cid, version=2)

        r = await client.get(REVISIONS_URL.format(id=cid))
        assert r.status_code == 200
        assert [item["version"] for item in r.json()] == [2, 1]

    async def test_revisions_empty_is_valid_200(self, client, factory):
        cid = await _seed_complaint(factory)
        r = await client.get(REVISIONS_URL.format(id=cid))
        assert r.status_code == 200
        assert r.json() == []


class TestApprove:
    async def test_approves_passed_draft(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        rid = await _seed_resolution(factory, cid)

        r = await client.post(APPROVE_URL.format(id=cid), json=await _identity(client, cid))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["action"] == "approved"
        assert body["resolution_id"] == str(rid)
        assert body["complaint_status"] == "response_approved"

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.response_approved.value
        events = await _audit_events(factory)
        assert events[0].event == "resolution_approved"
        assert events[0].metadata_["resolution_id"] == str(rid)

    async def test_failed_draft_is_not_approvable(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.needs_review)
        await _seed_resolution(factory, cid, guardrail_status=GuardrailStatus.failed)
        r = await client.post(APPROVE_URL.format(id=cid), json=await _identity(client, cid))
        assert r.status_code == 409
        assert "guardrails" in r.json()["detail"]

    async def test_already_resolved_is_409(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.resolved)
        await _seed_resolution(factory, cid)
        r = await client.post(APPROVE_URL.format(id=cid), json=await _identity(client, cid))
        assert r.status_code == 409

    async def test_unknown_revision_is_409(self, client, factory):
        cid = await _seed_complaint(factory)
        r = await client.post(APPROVE_URL.format(id=cid), json=await _identity(client, cid))
        assert r.status_code == 409


class TestReject:
    @pytest.mark.parametrize("stale", [False, True])
    async def test_outage_cannot_queue_rejection_regeneration(
        self, client, factory, fake_redis, stale
    ):
        cid = await _seed_complaint(
            factory,
            status=ComplaintStatus.draft_ready,
            classification_outcome=(
                ClassificationState.succeeded if stale else ClassificationState.unavailable
            ),
            classification_stale=stale,
        )
        await _seed_resolution(factory, cid)
        response = await client.post(
            REJECT_URL.format(id=cid),
            json={**await _identity(client, cid), "feedback": "Please revise this draft"},
        )
        assert response.status_code == 409
        assert await fake_redis.xrange(settings.resolution_queue) == []
        assert await _audit_events(factory) == []
        async with factory() as session:
            assert (await session.get(Complaint, cid)).status == ComplaintStatus.draft_ready

    async def test_queues_regeneration_with_feedback(self, client, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        rid = await _seed_resolution(factory, cid)

        r = await client.post(
            REJECT_URL.format(id=cid),
            json={
                **await _identity(client, cid),
                "feedback": "Cite the FCRA 30-day investigation window explicitly.",
            },
        )
        assert r.status_code == 202, r.text

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.agent_triggered.value
        async with factory() as session:
            job = (await session.exec(select(Job))).one()
            assert job.complaint_id == cid
            assert job.kind == JobKind.resolution
        assert "FCRA" in job.payload["feedback"]
        events = await _audit_events(factory)
        assert events[0].event == "resolution_rejected"
        assert events[0].metadata_["resolution_id"] == str(rid)
        assert "FCRA" in events[0].metadata_["feedback"]

    async def test_short_feedback_is_422(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        r = await client.post(
            REJECT_URL.format(id=cid), json={**await _identity(client, cid), "feedback": "bad"}
        )
        assert r.status_code == 422  # min_length=10 enforced by the schema

    async def test_unknown_revision_is_409(self, client, factory):
        cid = await _seed_complaint(factory)
        r = await client.post(
            REJECT_URL.format(id=cid),
            json={**await _identity(client, cid), "feedback": "needs more empathy"},
        )
        assert r.status_code == 409

    async def test_resolved_is_409(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.resolved)
        await _seed_resolution(factory, cid)
        r = await client.post(
            REJECT_URL.format(id=cid),
            json={**await _identity(client, cid), "feedback": "needs more empathy"},
        )
        assert r.status_code == 409


class TestAuthGate:
    async def test_unauthenticated_is_401(self, factory, fake_redis):
        app = _build_app(factory, fake_redis, bypass_auth=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get(LATEST_URL.format(id=uuid.uuid4()))
        assert r.status_code == 401


class TestWriterGate:
    """A viewer can read drafts but can't generate, approve, or reject."""

    def _viewer_client(self, factory, fake_redis):
        app = _build_app(factory, fake_redis, user=_VIEWER_USER)
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def test_viewer_cannot_generate(self, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.classified)
        async with self._viewer_client(factory, fake_redis) as ac:
            r = await ac.post(GENERATE_URL.format(id=cid))
        assert r.status_code == 403

    async def test_viewer_cannot_approve(self, factory, fake_redis):
        # the draft is passed + approvable, so 403 proves the gate fires before the body
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        async with self._viewer_client(factory, fake_redis) as ac:
            r = await ac.post(APPROVE_URL.format(id=cid))
        assert r.status_code == 403

    async def test_viewer_cannot_reject(self, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        async with self._viewer_client(factory, fake_redis) as ac:
            r = await ac.post(
                REJECT_URL.format(id=cid),
                json={**await _identity(ac, cid), "feedback": "needs more empathy"},
            )
        assert r.status_code == 403

    async def test_viewer_can_still_read_latest(self, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        async with self._viewer_client(factory, fake_redis) as ac:
            r = await ac.get(LATEST_URL.format(id=cid))
        assert r.status_code == 200


class TestVersionUniqueness:
    async def test_duplicate_complaint_version_rejected(self, factory):
        # UNIQUE(complaint_id, version) guards the max(version)+1 generation race.
        cid = await _seed_complaint(factory)
        await _seed_resolution(factory, cid, version=1)
        with pytest.raises(IntegrityError):
            await _seed_resolution(factory, cid, version=1)

    async def test_same_version_different_complaints_ok(self, factory):
        # The constraint is composite: version 1 for two different complaints is fine.
        cid1 = await _seed_complaint(factory)
        cid2 = await _seed_complaint(factory)
        await _seed_resolution(factory, cid1, version=1)
        await _seed_resolution(factory, cid2, version=1)  # must not raise

    async def test_incrementing_versions_same_complaint_ok(self, factory):
        cid = await _seed_complaint(factory)
        await _seed_resolution(factory, cid, version=1)
        await _seed_resolution(factory, cid, version=2)  # must not raise


class TestExactReview:
    @pytest.mark.parametrize("action", ["approve", "reject"])
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"expected_case_version": 1},
            {"resolution_id": "00000000-0000-0000-0000-000000000001"},
        ],
    )
    async def test_old_clients_fail_closed(self, client, factory, action, body):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        if action == "reject":
            body = {**body, "feedback": "Please clarify the response"}
        response = await client.post(f"/api/v1/resolutions/{cid}/{action}", json=body)
        assert response.status_code == 422
        assert await _audit_events(factory) == []

    @pytest.mark.parametrize(
        "change", ["new_draft", "case_version", "input_revision", "cross_case"]
    )
    async def test_identity_and_version_conflicts_preserve_state(self, client, factory, change):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        rid = await _seed_resolution(factory, cid)
        body = await _identity(client, cid)
        if change == "new_draft":
            await _seed_resolution(factory, cid, version=2, draft_text="A new draft")
        elif change == "cross_case":
            other = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
            body["resolution_id"] = str(await _seed_resolution(factory, other))
        else:
            async with factory() as session:
                case = await session.get(Complaint, cid)
                setattr(case, change, 2)
                session.add(case)
                await session.commit()
        response = await client.post(APPROVE_URL.format(id=cid), json=body)
        assert response.status_code == 409
        assert "Reload" in response.json()["detail"]
        assert await _audit_events(factory) == []
        async with factory() as session:
            assert (await session.get(Complaint, cid)).status == ComplaintStatus.draft_ready
            assert (await session.get(Resolution, rid)).draft_text != "A new draft"

    async def test_rejection_permanently_invalidates_revision(self, client, factory):
        from app.models.review_event import ReviewEvent

        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        rid = await _seed_resolution(factory, cid)
        original = await _identity(client, cid)
        rejected = await client.post(
            REJECT_URL.format(id=cid), json={**original, "feedback": "Please revise this response"}
        )
        assert rejected.status_code == 202
        # Refreshing only the case version must not make rejected text approvable.
        fresh = await _identity(client, cid)
        response = await client.post(APPROVE_URL.format(id=cid), json=fresh)
        assert response.status_code == 409
        async with factory() as session:
            events = (await session.exec(select(ReviewEvent))).all()
            assert len(events) == 1
            assert events[0].action == "rejected"
            assert events[0].resolution_id == rid
        assert (await client.get(LATEST_URL.format(id=cid))).json()["review_action"] == "rejected"

    async def test_job_insert_failure_rolls_back_review(self, client, factory, monkeypatch):
        from app.models.review_event import ReviewEvent

        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        body = {**await _identity(client, cid), "feedback": "Please revise this response"}

        async def unavailable(*args, **kwargs):
            raise ConnectionError("synthetic database insert failure")

        monkeypatch.setattr("app.api.routes.resolutions.enqueue_job", unavailable)
        with pytest.raises(ConnectionError):
            await client.post(REJECT_URL.format(id=cid), json=body)
        async with factory() as session:
            case = await session.get(Complaint, cid)
            assert (case.status, case.case_version) == (ComplaintStatus.draft_ready, 1)
            assert (await session.exec(select(ReviewEvent))).all() == []
            assert (await session.exec(select(Job))).all() == []
