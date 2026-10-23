"""HTTP intent replay, transactional admission and bounded staff retries."""

import uuid
from unittest.mock import AsyncMock

import pytest
from sqlmodel import select

from app.config import settings
from app.models.complaint import Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.models.user import UserRole
from app.services.jobs import claim_job, enqueue_job, fail_job
from tests.test_rbac_matrix import _build_app, _client
from tests.test_rbac_matrix import factory as factory
from tests.test_rbac_matrix import fake_redis as fake_redis

INTAKE = "/api/v1/complaints/"
BODY = {"narrative": "Synthetic complaint about an unexplained duplicate fee."}


async def test_intake_replay_returns_original_even_when_capacity_full(
    factory, fake_redis, monkeypatch
):
    monkeypatch.setattr(settings, "job_max_pending", 1)
    app = _build_app(factory, fake_redis, role=UserRole.analyst)
    async with _client(app) as client:
        first = await client.post(INTAKE, json=BODY, headers={"Idempotency-Key": "intake-1"})
        assert first.status_code == 201
        replay = await client.post(INTAKE, json=BODY, headers={"Idempotency-Key": "intake-1"})
        assert replay.status_code == 201
        assert replay.json()["id"] == first.json()["id"]
        changed = await client.post(
            INTAKE,
            json={"narrative": BODY["narrative"] + " Changed."},
            headers={"Idempotency-Key": "intake-1"},
        )
        assert changed.status_code == 409
        overflow = await client.post(INTAKE, json=BODY)
        assert overflow.status_code == 503
        assert "no changes" in overflow.text
    async with factory() as session:
        assert len((await session.exec(select(Complaint))).all()) == 1
        assert len((await session.exec(select(Job))).all()) == 1


async def test_intake_job_insert_failure_rolls_back_case(factory, fake_redis, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.complaints.enqueue_job",
        AsyncMock(side_effect=RuntimeError("synthetic insert failure")),
    )
    app = _build_app(factory, fake_redis, role=UserRole.analyst)
    async with _client(app) as client:
        with pytest.raises(RuntimeError, match="synthetic insert failure"):
            await client.post(INTAKE, json=BODY)
    async with factory() as session:
        assert (await session.exec(select(Complaint))).all() == []
        assert (await session.exec(select(Job))).all() == []


async def failed_job(factory):
    async with factory() as session:
        case = Complaint(narrative=BODY["narrative"])
        session.add(case)
        job = await enqueue_job(
            session,
            case,
            JobKind.classification,
            max_attempts=2,
            payload={"private": "not in operational API"},
        )
        owned = await claim_job(session, JobKind.classification)
        fail_job(owned, "invalid_input", retryable=False)
        await session.commit()
        return case, job


async def test_inspection_redacts_payload_and_retry_replays_one_intent(factory, fake_redis):
    _, job = await failed_job(factory)
    app = _build_app(factory, fake_redis, role=UserRole.analyst)
    async with _client(app) as client:
        listing = await client.get("/api/v1/jobs/")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1
        detail = await client.get(f"/api/v1/jobs/{job.id}")
        assert detail.json()["last_error_code"] == "invalid_input"
        assert {"payload", "lease_token", "idempotency_key"}.isdisjoint(detail.json())
        body = {"expected_attempts": 1, "expected_retry_generation": 0}
        first = await client.post(f"/api/v1/jobs/{job.id}/retry", json=body)
        repeated = await client.post(f"/api/v1/jobs/{job.id}/retry", json=body)
        assert first.status_code == repeated.status_code == 200
        assert first.json() == repeated.json()
        assert first.json()["attempts"] == 0
        assert first.json()["total_attempts"] == first.json()["retry_generation"] == 1
        wrong = await client.post(
            f"/api/v1/jobs/{job.id}/retry", json={**body, "expected_attempts": 2}
        )
        assert wrong.status_code == 409
        invalid = await client.post(
            f"/api/v1/jobs/{job.id}/retry", json={**body, "expected_attempts": True}
        )
        assert invalid.status_code == 422
        assert (await client.get(f"/api/v1/jobs/{uuid.uuid4()}")).status_code == 404
    async with factory() as session:
        assert len((await session.exec(select(Job))).all()) == 1


async def test_failed_job_retry_rejects_newer_approval(factory, fake_redis):
    case, job = await failed_job(factory)
    async with factory() as session:
        case.status = ComplaintStatus.response_approved
        session.add(case)
        await session.commit()
    async with _client(_build_app(factory, fake_redis, role=UserRole.analyst)) as client:
        response = await client.post(
            f"/api/v1/jobs/{job.id}/retry",
            json={
                "expected_attempts": 1,
                "expected_retry_generation": 0,
            },
        )
        assert response.status_code == 409
    async with factory() as session:
        assert (await session.get(Job, job.id)).status == JobStatus.failed
        assert (await session.get(Complaint, case.id)).status == ComplaintStatus.response_approved


async def test_generate_replay_has_one_transition_and_job(factory, fake_redis):
    async with factory() as session:
        case = Complaint(
            narrative=BODY["narrative"],
            status=ComplaintStatus.escalated,
            sentiment="negative",
            intent="dispute_resolution",
            urgency=4,
        )
        session.add(case)
        await session.commit()
    async with _client(_build_app(factory, fake_redis, role=UserRole.analyst)) as client:
        url = f"/api/v1/resolutions/{case.id}/generate"
        first = await client.post(url, headers={"Idempotency-Key": "generate-1"})
        again = await client.post(url, headers={"Idempotency-Key": "generate-1"})
        assert first.status_code == again.status_code == 202
        assert first.json()["job_id"] == again.json()["job_id"]
    async with factory() as session:
        assert (await session.get(Complaint, case.id)).case_version == 2
        assert len((await session.exec(select(Job))).all()) == 1


@pytest.mark.parametrize("running", [False, True])
async def test_generate_cannot_duplicate_automatic_draft_intent(factory, fake_redis, running):
    async with factory() as session:
        case = Complaint(
            narrative=BODY["narrative"],
            status=ComplaintStatus.escalated,
            sentiment="negative",
            intent="dispute_resolution",
            urgency=4,
        )
        session.add(case)
        original = await enqueue_job(session, case, JobKind.resolution)
        if running:
            await claim_job(session, JobKind.resolution)
        await session.commit()
    async with _client(_build_app(factory, fake_redis, role=UserRole.analyst)) as client:
        response = await client.post(
            f"/api/v1/resolutions/{case.id}/generate",
            headers={"Idempotency-Key": "manual-duplicate"},
        )
        assert response.status_code == 409
    async with factory() as session:
        assert (await session.get(Complaint, case.id)).case_version == 1
        jobs = (await session.exec(select(Job))).all()
        assert len(jobs) == 1 and jobs[0].id == original.id
