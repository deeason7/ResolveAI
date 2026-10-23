"""Derived writes retry separately and repair snapshots overwritten by late work."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.complaint import Complaint
from app.models.job import Job, JobKind, JobStatus
from app.services.jobs import claim_job, enqueue_job
from app.workers.index_worker import IndexWorker
from tests.test_classification_worker import factory as factory


async def _seed(factory):
    async with factory() as session:
        case = Complaint(
            narrative="Synthetic disputed fee",
            company="Fixture Bank",
            product="Credit card",
            issue="Fees",
        )
        session.add(case)
        job = await enqueue_job(
            session,
            case,
            JobKind.index,
            max_attempts=2,
            payload={"key_entities": [{"text": "fee", "type": "issue"}]},
        )
        await session.commit()
        return case.id, job.id


@pytest.fixture()
def stores(monkeypatch):
    monkeypatch.setattr("app.workers.index_worker.embed_text", lambda _: [0.1, 0.2])
    return Mock(), AsyncMock()


async def test_partial_store_failure_retries_same_payload_without_inference(factory, stores):
    cid, jid = await _seed(factory)
    vector, graph = stores
    graph.upsert_complaint_entities.side_effect = [ConnectionError("sensitive detail"), None]
    worker = IndexWorker(vector, graph, session_factory=factory)
    assert await worker.run_once()
    async with factory() as session:
        job = await session.get(Job, jid)
        assert job.status == JobStatus.queued and job.attempts == 1
        assert job.last_error_code == "dependency_unavailable"
        job.next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()
    assert await worker.run_once()
    assert not await worker.run_once()
    assert vector.upsert_complaint.call_count == 2
    assert vector.upsert_complaint.call_args_list[0] == vector.upsert_complaint.call_args_list[1]
    args = vector.upsert_complaint.call_args.args
    assert args[0] == str(cid) and args[2]["input_revision"] == 1
    assert "status" not in args[2]
    assert args[2]["key_entities"] == [{"text": "fee", "type": "issue"}]
    async with factory() as session:
        assert (await session.get(Job, jid)).status == JobStatus.succeeded
        assert (await session.get(Complaint, cid)).case_version == 1


@pytest.mark.parametrize("partial_failure", [False, True])
async def test_obsolete_write_reopens_current_completed_index_after_lease_loss(
    factory, stores, partial_failure
):
    cid, jid = await _seed(factory)
    vector, graph = stores
    new_id = None

    async def finish_old_write(**_kwargs):
        nonlocal new_id
        # Deterministic completion ordering: the newer snapshot has finished
        # while the old request is still inside its external write.
        async with factory() as session:
            case = await session.get(Complaint, cid)
            case.input_revision += 1
            case.case_version += 1
            new = await enqueue_job(
                session, case, JobKind.index, payload={"key_entities": ["new entity"]}
            )
            new.status = JobStatus.succeeded
            new.attempts = new.total_attempts = 1
            new.completed_at = datetime.utcnow()
            old = await session.get(Job, jid)
            old.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
            new_id = new.id
            await session.commit()
        if partial_failure:
            raise ConnectionError("second store unavailable")

    graph.upsert_complaint_entities.side_effect = finish_old_write
    await IndexWorker(vector, graph, session_factory=factory).run_once()
    async with factory() as session:
        new = await session.get(Job, new_id)
        assert new.status == JobStatus.queued
        assert new.attempts == 0 and new.total_attempts == 1
        assert new.retry_generation == 1
        assert new.payload == {"key_entities": ["new entity"]}
        assert new.last_error_code == "index_repair_requested"
        assert (await session.get(Job, jid)).status == JobStatus.running


async def test_expired_lease_cannot_complete_same_input_index(factory, stores):
    _, jid = await _seed(factory)
    vector, graph = stores

    async def expire(**_kwargs):
        async with factory() as session:
            job = await session.get(Job, jid)
            job.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
            await session.commit()

    graph.upsert_complaint_entities.side_effect = expire
    await IndexWorker(vector, graph, session_factory=factory).run_once()
    async with factory() as session:
        job = await session.get(Job, jid)
        assert job.status == JobStatus.running
        replacement = await claim_job(session, JobKind.index)
        assert replacement.id == jid and replacement.attempts == 2
        await session.commit()
