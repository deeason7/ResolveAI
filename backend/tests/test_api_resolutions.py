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
from app.models.complaint import Complaint, ComplaintStatus
from app.models.resolution import GuardrailStatus, Resolution

GENERATE_URL = "/api/v1/resolutions/{id}/generate"
LATEST_URL = "/api/v1/resolutions/{id}"
REVISIONS_URL = "/api/v1/resolutions/{id}/revisions"
APPROVE_URL = "/api/v1/resolutions/{id}/approve"
REJECT_URL = "/api/v1/resolutions/{id}/reject"

_DUMMY_USER = SimpleNamespace(id=uuid.uuid4(), email="analyst@test.com")


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401 — registers tables on SQLModel.metadata
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture()
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _build_app(factory, fake_redis, *, bypass_auth=True):
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
        app.dependency_overrides[get_current_user] = lambda: _DUMMY_USER
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


class TestGenerate:
    async def test_queues_and_marks_in_flight(self, client, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.classified)
        r = await client.post(GENERATE_URL.format(id=cid))

        assert r.status_code == 202, r.text
        assert r.json() == {"complaint_id": str(cid), "status": "queued"}
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.agent_triggered.value
        entries = await fake_redis.xrange(settings.resolution_queue)
        assert entries[0][1]["complaint_id"] == str(cid)

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

        r = await client.post(APPROVE_URL.format(id=cid))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["action"] == "approved"
        assert body["resolution_id"] == str(rid)
        assert body["complaint_status"] == "resolved"

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.resolved.value
        events = await _audit_events(factory)
        assert events[0].event == "resolution_approved"
        assert events[0].metadata_["resolution_id"] == str(rid)

    async def test_failed_draft_is_not_approvable(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.needs_review)
        await _seed_resolution(factory, cid, guardrail_status=GuardrailStatus.failed)
        r = await client.post(APPROVE_URL.format(id=cid))
        assert r.status_code == 409
        assert "guardrails" in r.json()["detail"]

    async def test_already_resolved_is_409(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.resolved)
        await _seed_resolution(factory, cid)
        r = await client.post(APPROVE_URL.format(id=cid))
        assert r.status_code == 409

    async def test_no_resolution_is_404(self, client, factory):
        cid = await _seed_complaint(factory)
        r = await client.post(APPROVE_URL.format(id=cid))
        assert r.status_code == 404


class TestReject:
    async def test_queues_regeneration_with_feedback(self, client, factory, fake_redis):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        rid = await _seed_resolution(factory, cid)

        r = await client.post(
            REJECT_URL.format(id=cid),
            json={"feedback": "Cite the FCRA 30-day investigation window explicitly."},
        )
        assert r.status_code == 202, r.text

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.agent_triggered.value
        entries = await fake_redis.xrange(settings.resolution_queue)
        assert entries[0][1]["complaint_id"] == str(cid)
        assert "FCRA" in entries[0][1]["feedback"]
        events = await _audit_events(factory)
        assert events[0].event == "resolution_rejected"
        assert events[0].metadata_["resolution_id"] == str(rid)
        assert "FCRA" in events[0].metadata_["feedback"]

    async def test_short_feedback_is_422(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.draft_ready)
        await _seed_resolution(factory, cid)
        r = await client.post(REJECT_URL.format(id=cid), json={"feedback": "bad"})
        assert r.status_code == 422  # min_length=10 enforced by the schema

    async def test_no_resolution_is_404(self, client, factory):
        cid = await _seed_complaint(factory)
        r = await client.post(REJECT_URL.format(id=cid), json={"feedback": "needs more empathy"})
        assert r.status_code == 404

    async def test_resolved_is_409(self, client, factory):
        cid = await _seed_complaint(factory, status=ComplaintStatus.resolved)
        await _seed_resolution(factory, cid)
        r = await client.post(REJECT_URL.format(id=cid), json={"feedback": "needs more empathy"})
        assert r.status_code == 409


class TestAuthGate:
    async def test_unauthenticated_is_401(self, factory, fake_redis):
        app = _build_app(factory, fake_redis, bypass_auth=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get(LATEST_URL.format(id=uuid.uuid4()))
        assert r.status_code == 401


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
