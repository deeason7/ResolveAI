"""Tests for the Redis-stream classification worker.

The worker is driven against real fakeredis streams, an in-memory Qdrant, and a
stub classifier — no live Ollama/Postgres/Qdrant. ``embed_text`` is stubbed so
the sentence-transformer model never loads.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
import pytest_asyncio
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint, ComplaintStatus
from app.models.llm_log import LLMLog
from app.schemas.classification import ComplaintClassification
from app.services.classifier import ClassificationOutcome
from app.services.vector_store import VectorStore
from app.workers import classification_worker as cw
from app.workers.classification_worker import (
    ClassificationWorker,
    compute_priority_score,
    enqueue_complaint,
    is_high_priority,
)
from app.workers.stream_utils import reclaim_stale_messages

EMBED_DIM = 384


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


@pytest.fixture()
def vector_store():
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection()
    return store


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    monkeypatch.setattr(cw, "embed_text", lambda text: [0.01] * EMBED_DIM)


def _outcome(*, sentiment="negative", urgency=3, succeeded=True, is_fallback=False):
    cls = ComplaintClassification(
        sentiment=sentiment,
        intent="dispute_resolution",
        urgency=urgency,
        key_entities=[],
        reasoning="model reasoning text here",
    )
    return ClassificationOutcome(
        classification=cls,
        provider="ollama" if succeeded else "none",
        model="resolveai-sentiment" if succeeded else "none",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=100,
        is_fallback=is_fallback,
        succeeded=succeeded,
    )


class _StubClassifier:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[str] = []

    def classify(self, narrative, *, product=None, issue=None, company=None):
        self.calls.append(narrative)
        return self.outcome


async def _seed(factory, **over) -> uuid.UUID:
    fields = {
        "narrative": "I want to dispute a fee",
        "product": "Credit card",
        "issue": "Fees",
        "company": "Chase",
    }
    fields.update(over)
    async with factory() as s:
        c = Complaint(**fields)
        s.add(c)
        await s.commit()
        return c.id


def _worker(redis_client, factory, vector_store, outcome, graph_store=None):
    return ClassificationWorker(
        redis_client=redis_client,
        classifier=_StubClassifier(outcome),
        vector_store=vector_store,
        session_factory=factory,
        graph_store=graph_store,
    )


class TestPureHelpers:
    def test_priority_extremes(self):
        assert compute_priority_score(5, "extreme_negative") == 1.0
        assert compute_priority_score(1, "neutral") == 0.0

    def test_high_priority_conditions(self):
        assert is_high_priority(4, "neutral") is True
        assert is_high_priority(1, "extreme_negative") is True
        assert is_high_priority(2, "negative") is False


class TestProcessMessage:
    async def test_classifies_and_persists(self, redis_client, factory, vector_store):
        cid = await _seed(factory)
        w = _worker(redis_client, factory, vector_store, _outcome(urgency=3))
        await w.process_message({"complaint_id": str(cid)})

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.classified
            assert c.sentiment == "negative"
            assert c.urgency == 3
            assert c.priority_score is not None
            logs = (await s.exec(select(LLMLog))).all()
            assert len(logs) == 1
            assert logs[0].operation == "classify"
            assert logs[0].cost_usd == 0.0
        assert vector_store.collection_count() == 1

    async def test_high_priority_escalates(self, redis_client, factory, vector_store):
        cid = await _seed(factory)
        w = _worker(redis_client, factory, vector_store, _outcome(urgency=5))
        await w.process_message({"complaint_id": str(cid)})
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.escalated

    async def test_missing_complaint_is_poison(self, redis_client, factory, vector_store):
        w = _worker(redis_client, factory, vector_store, _outcome())
        # Returns normally so the caller acks; nothing to update.
        await w.process_message({"complaint_id": str(uuid.uuid4())})

    async def test_bad_uuid_is_poison(self, redis_client, factory, vector_store):
        w = _worker(redis_client, factory, vector_store, _outcome())
        await w.process_message({"complaint_id": "not-a-uuid"})

    async def test_vector_failure_still_commits(
        self, redis_client, factory, vector_store, monkeypatch
    ):
        cid = await _seed(factory)

        def _boom(text):
            raise RuntimeError("qdrant down")

        monkeypatch.setattr(cw, "embed_text", _boom)
        w = _worker(redis_client, factory, vector_store, _outcome())
        await w.process_message({"complaint_id": str(cid)})  # must not raise

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.classified  # postgres committed

    async def test_graph_upsert_called_with_structured_fields(
        self, redis_client, factory, vector_store
    ):
        cid = await _seed(factory, company="Wells Fargo", product="Mortgage", issue="Servicing")
        graph_store = AsyncMock()
        w = _worker(redis_client, factory, vector_store, _outcome(), graph_store=graph_store)
        await w.process_message({"complaint_id": str(cid)})

        graph_store.upsert_complaint_entities.assert_awaited_once_with(
            complaint_id=str(cid),
            company="Wells Fargo",
            product="Mortgage",
            issue="Servicing",
        )

    async def test_graph_failure_still_commits(self, redis_client, factory, vector_store):
        cid = await _seed(factory)
        graph_store = AsyncMock()
        graph_store.upsert_complaint_entities.side_effect = RuntimeError("neo4j down")
        w = _worker(redis_client, factory, vector_store, _outcome(), graph_store=graph_store)
        await w.process_message({"complaint_id": str(cid)})  # must not raise

        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.classified  # postgres committed

    async def test_no_graph_store_skips_upsert(self, redis_client, factory, vector_store):
        # The default wiring (graph_store=None) must not blow up — the write is skipped.
        cid = await _seed(factory)
        w = _worker(redis_client, factory, vector_store, _outcome())
        await w.process_message({"complaint_id": str(cid)})
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.classified


class TestStreamIntegration:
    async def test_ensure_group_is_idempotent(self, redis_client, factory, vector_store):
        w = _worker(redis_client, factory, vector_store, _outcome())
        await w.ensure_group()
        await w.ensure_group()  # second call must not raise BUSYGROUP

    async def test_enqueue_then_handle_acks(self, redis_client, factory, vector_store):
        cid = await _seed(factory)
        w = _worker(redis_client, factory, vector_store, _outcome())
        await w.ensure_group()
        await enqueue_complaint(redis_client, cid, stream=w.stream)

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

    async def test_reclaims_message_stranded_by_dead_consumer(
        self, redis_client, factory, vector_store
    ):
        # A different consumer reads the message and 'dies' without acking, so it
        # sits in the PEL. The reclaimer hands it to THIS worker's _handle, which
        # runs the real classify + persist + ack path.
        cid = await _seed(factory)
        w = _worker(redis_client, factory, vector_store, _outcome(urgency=3))
        await w.ensure_group()
        await enqueue_complaint(redis_client, cid, stream=w.stream)
        await redis_client.xreadgroup(
            groupname=w.group, consumername="dead-worker", streams={w.stream: ">"}, count=10
        )
        assert (await redis_client.xpending(w.stream, w.group))["pending"] == 1

        reclaimed = await reclaim_stale_messages(
            redis_client,
            stream=w.stream,
            group=w.group,
            consumer=w.consumer,
            handle=w._handle,
            min_idle_ms=0,
        )
        assert reclaimed == 1
        async with factory() as s:
            c = await s.get(Complaint, cid)
            assert c.status == ComplaintStatus.classified  # processed end-to-end
        assert (await redis_client.xpending(w.stream, w.group))["pending"] == 0  # acked, PEL clear


class TestPollCadence:
    """The knobs that bound idle Redis command volume on a command-billed tier
    (Upstash free). ``_tick`` is driven directly so the gate is tested without
    running the unbounded ``run`` loop.
    """

    async def test_sweeps_only_every_nth_cycle_and_honors_block(
        self, redis_client, factory, vector_store, monkeypatch
    ):
        from app.config import Settings

        sweeps = AsyncMock(return_value=0)
        monkeypatch.setattr(cw, "reclaim_stale_messages", sweeps)
        # Empty stream: stub the read so a real BLOCK never stalls the test.
        redis_client.xreadgroup = AsyncMock(return_value=[])

        settings = Settings(worker_reclaim_every=3, worker_block_ms=250)
        w = ClassificationWorker(
            redis_client=redis_client,
            classifier=_StubClassifier(_outcome()),
            vector_store=vector_store,
            session_factory=factory,
            settings=settings,
        )

        for cycle in range(6):
            await w._tick(cycle)

        assert sweeps.await_count == 2  # only cycles 0 and 3
        assert redis_client.xreadgroup.await_args.kwargs["block"] == 250
