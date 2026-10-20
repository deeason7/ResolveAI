"""Classification persistence and an in-flight decision on real PostgreSQL."""

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest
from sqlmodel import select

from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.llm_log import LLMLog
from app.workers.classification_worker import ClassificationWorker
from tests.test_classification_worker import _outcome

pytestmark = pytest.mark.postgres


async def test_unavailable_result_and_unknown_usage_commit_together(pg_sessionmaker):
    case = Complaint(narrative="Synthetic intake during provider outage")
    async with pg_sessionmaker() as session:
        session.add(case)
        await session.commit()
    worker = ClassificationWorker(
        redis_client=AsyncMock(),
        vector_store=Mock(),
        graph_store=AsyncMock(),
        classifier=Mock(classify=Mock(return_value=_outcome(succeeded=False))),
        session_factory=pg_sessionmaker,
    )
    await worker.process_message(
        {"expected_case_version": "1", "input_revision": "1", "complaint_id": str(case.id)}
    )
    async with pg_sessionmaker() as session:
        stored = await session.get(Complaint, case.id)
        assert stored.classification_outcome == ClassificationState.unavailable
        assert stored.status == ComplaintStatus.needs_review
        assert stored.urgency is stored.sentiment is stored.priority_score is None
        assert stored.classification_latency_ms == 100
        log = (await session.exec(select(LLMLog))).one()
        assert log.complaint_id == case.id
        assert log.operation == "classify_unavailable"
        assert log.prompt_tokens is log.completion_tokens is log.cost_usd is None
    worker.redis.xadd.assert_not_awaited()
    worker.vector_store.upsert_complaint.assert_not_called()
    worker.graph_store.upsert_complaint_entities.assert_not_awaited()


@pytest.mark.parametrize("succeeded", [False, True])
async def test_human_commit_during_inference_wins(pg_sessionmaker, succeeded):
    started, release = threading.Event(), threading.Event()
    case = Complaint(narrative="Synthetic concurrent review")
    async with pg_sessionmaker() as session:
        session.add(case)
        await session.commit()

    def inference(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=10)
        return _outcome(succeeded=succeeded, urgency=5)

    worker = ClassificationWorker(
        redis_client=AsyncMock(),
        vector_store=Mock(),
        classifier=Mock(classify=inference),
        session_factory=pg_sessionmaker,
    )
    task = asyncio.create_task(
        worker.process_message(
            {"expected_case_version": "1", "input_revision": "1", "complaint_id": str(case.id)}
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        # Independent transaction commits while the worker is outside its row lock.
        async with pg_sessionmaker() as reviewer:
            changed = await reviewer.get(Complaint, case.id)
            changed.status = ComplaintStatus.resolved
            changed.classification_source = "manual"
            changed.classification_outcome = ClassificationState.manual
            changed.urgency = 1
            reviewer.add(changed)
            await reviewer.commit()
    finally:
        release.set()
        await asyncio.wait_for(task, 10)
    async with pg_sessionmaker() as session:
        stored = await session.get(Complaint, case.id)
        assert stored.status == ComplaintStatus.resolved
        assert stored.classification_outcome == ClassificationState.manual
        assert stored.classification_source == "manual"
        assert stored.urgency == 1
        assert (await session.exec(select(LLMLog))).all() == []
    worker.redis.xadd.assert_not_awaited()
    worker.vector_store.upsert_complaint.assert_not_called()
