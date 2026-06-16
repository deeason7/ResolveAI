"""Background worker: drain the classification stream, classify, persist.

Flow per message (spec Day 13):
    Redis stream  ->  fetch complaint from Postgres  ->  classify via the SLM
                  ->  write sentiment/intent/urgency/status + LLMLog (one txn)
                  ->  embed + upsert to Qdrant (best-effort secondary index)
                  ->  high-priority ones are flagged for the Phase 4 agent.

Two design choices worth calling out:

* **Redis Streams, not a list.** A consumer group gives us at-least-once
  delivery and a pending-entries list (PEL), so a crash mid-classification
  doesn't drop the message. A plain ``LPUSH``/``BRPOP`` list would.
* **Postgres is the consistency boundary; Qdrant is best-effort.** The complaint
  row and its LLMLog commit in one transaction. The vector upsert happens *after*
  that commit and is allowed to fail without blocking the ack — a stale search
  index is recoverable (re-run ``populate_vector_db.py``); a lost classification
  is not.

Runs at concurrency 1 by design: the fine-tuned model is served by a single
Ollama process on an 8 GB M3, so parallel calls just thrash swap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import uuid
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from app.config import Settings
from app.config import settings as default_settings
from app.database import AsyncSessionLocal
from app.models.complaint import Complaint, ComplaintStatus
from app.services.classifier import Classifier
from app.services.embedder import embed_text
from app.services.graph_store import GraphStore, get_default_graph_store
from app.services.llmops_tracker import LLMOpsTracker
from app.services.vector_store import VectorStore, get_default_store
from app.workers.resolution_worker import enqueue_resolution
from app.workers.stream_utils import reclaim_stale_messages

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "classifiers"
# How long XREADGROUP blocks before returning empty so the loop can re-check the
# stop flag. Long enough to mostly idle, short enough for a responsive shutdown.
BLOCK_MS = 5000
READ_COUNT = 1  # concurrency 1: pull one at a time, keep the model un-thrashed

# Priority blends urgency (primary) with sentiment (secondary) into [0, 1] so the
# Phase 5 triage queue has a single sortable score. urgency 1..5 -> 0..1 linear.
_SENTIMENT_WEIGHT = {"neutral": 0.0, "negative": 0.5, "extreme_negative": 1.0}


def compute_priority_score(urgency: int, sentiment: str) -> float:
    """Weighted blend of urgency and sentiment, clamped to [0, 1]."""
    raw = 0.7 * (urgency - 1) / 4 + 0.3 * _SENTIMENT_WEIGHT.get(sentiment, 0.5)
    return round(max(0.0, min(1.0, raw)), 4)


def is_high_priority(urgency: int, sentiment: str) -> bool:
    """Acute complaints route to the resolution agent (Phase 4)."""
    return urgency >= 4 or sentiment == "extreme_negative"


async def enqueue_complaint(
    redis_client: aioredis.Redis,
    complaint_id: uuid.UUID | str,
    *,
    stream: str | None = None,
) -> str:
    """Producer side of the stream contract: XADD a complaint id for classifying.

    Kept here so producers (the submit route in Phase 4, a backfill script) share
    one definition of the message shape instead of hand-rolling XADD calls.
    """
    stream = stream or default_settings.classification_queue
    return await redis_client.xadd(stream, {"complaint_id": str(complaint_id)})


class ClassificationWorker:
    """Consumes complaint ids from a Redis stream and classifies them.

    Every collaborator is injected so the whole pipeline is testable against
    fakeredis + an in-memory Qdrant + a stub classifier, with no live services.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        classifier: Classifier,
        vector_store: VectorStore,
        *,
        graph_store: GraphStore | None = None,
        tracker: LLMOpsTracker | None = None,
        settings: Settings | None = None,
        session_factory: Any = AsyncSessionLocal,
        consumer_name: str | None = None,
    ) -> None:
        self.redis = redis_client
        self.classifier = classifier
        self.vector_store = vector_store
        # Optional: when absent (e.g. most unit tests) the graph write is skipped.
        # Production wires the singleton in main(); graph upserts are best-effort.
        self.graph_store = graph_store
        self.tracker = tracker or LLMOpsTracker()
        self.settings = settings or default_settings
        self.session_factory = session_factory
        self.stream = self.settings.classification_queue
        self.group = CONSUMER_GROUP
        # Unique per process so two workers in the same group never collide in
        # the PEL. host+pid is stable across the process lifetime.
        self.consumer = consumer_name or f"{socket.gethostname()}-{os.getpid()}"
        self._stop = asyncio.Event()

    async def ensure_group(self) -> None:
        """Create the consumer group (and stream) if absent; idempotent.

        ``id="0"`` so a group created after a producer already enqueued still
        drains that backlog. On restart the group already exists (BUSYGROUP) and
        its last-delivered cursor is preserved, so we don't re-read history.
        """
        try:
            await self.redis.xgroup_create(
                name=self.stream, groupname=self.group, id="0", mkstream=True
            )
            logger.info("created consumer group %s on %s", self.group, self.stream)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def request_stop(self) -> None:
        """Break the loop after the in-flight message finishes (signal-safe)."""
        self._stop.set()

    async def run(self) -> None:
        """Main loop: block on the stream, process, ack. Exits on stop request."""
        await self.ensure_group()
        logger.info(
            "classification worker up: stream=%s group=%s consumer=%s",
            self.stream,
            self.group,
            self.consumer,
        )
        while not self._stop.is_set():
            # First rescue anything a dead/slow consumer left stranded in the
            # PEL — XREADGROUP '>' below would never redeliver those on its own.
            await reclaim_stale_messages(
                self.redis,
                stream=self.stream,
                group=self.group,
                consumer=self.consumer,
                handle=self._handle,
                min_idle_ms=self.settings.reclaim_min_idle_ms,
            )
            resp = await self.redis.xreadgroup(
                groupname=self.group,
                consumername=self.consumer,
                streams={self.stream: ">"},
                count=READ_COUNT,
                block=BLOCK_MS,
            )
            if not resp:
                continue  # block timed out — loop back and re-check the stop flag
            for _stream, messages in resp:
                for message_id, fields in messages:
                    await self._handle(message_id, fields)

    async def _handle(self, message_id: str, fields: dict[str, str]) -> None:
        """Process one message, then ack only if it's truly settled.

        Ack == "done with this, never redeliver": both success and *poison*
        (unparseable / missing complaint) ack, because retrying them is futile.
        A transient failure (DB down) raises out of ``process_message`` and is
        left unacked in the PEL for a future claim.
        """
        try:
            await self.process_message(fields)
        except Exception:
            logger.exception("transient failure on %s; leaving unacked for retry", message_id)
            return
        await self.redis.xack(self.stream, self.group, message_id)

    async def process_message(self, fields: dict[str, str]) -> None:
        """Classify one complaint and persist the result.

        Returns normally on success *and* on poison messages (so they get acked).
        Raises only on retryable infrastructure failures.
        """
        raw_id = fields.get("complaint_id")
        try:
            complaint_id = uuid.UUID(raw_id) if raw_id else None
        except ValueError:
            complaint_id = None
        if complaint_id is None:
            logger.warning("dropping message with bad complaint_id: %r", raw_id)
            return  # poison -> ack

        async with self.session_factory() as session:
            complaint = await session.get(Complaint, complaint_id)
            if complaint is None:
                logger.warning("complaint %s not found; dropping", complaint_id)
                return  # poison -> ack

            # The LLM call is synchronous and network-bound; trampoline it off the
            # event loop so blocking on Ollama doesn't stall the whole worker.
            outcome = await asyncio.to_thread(
                self.classifier.classify,
                complaint.narrative,
                product=complaint.product,
                issue=complaint.issue,
                company=complaint.company,
            )
            cls = outcome.classification

            complaint.sentiment = cls.sentiment
            complaint.intent = cls.intent
            complaint.urgency = cls.urgency
            complaint.priority_score = compute_priority_score(cls.urgency, cls.sentiment)
            high = is_high_priority(cls.urgency, cls.sentiment)
            complaint.status = ComplaintStatus.escalated if high else ComplaintStatus.classified
            complaint.updated_at = datetime.utcnow()
            session.add(complaint)

            # One transaction: the classification and its audit log land together.
            self.tracker.record(
                session,
                operation="classify",
                provider=outcome.provider,
                model=outcome.model,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                latency_ms=outcome.latency_ms,
                was_fallback=outcome.is_fallback,
                complaint_id=complaint.id,
            )
            await session.commit()

        if not outcome.succeeded:
            logger.warning(
                "complaint %s classified by deterministic fallback (providers down)",
                complaint_id,
            )
        if high:
            await self._trigger_resolution(complaint_id, cls)

        await self._index(complaint_id, complaint.narrative, complaint, cls)
        await self._update_graph(complaint)

    async def _trigger_resolution(self, complaint_id: uuid.UUID, cls: Any) -> None:
        """Best-effort: hand the escalated complaint to the resolution agent.

        Postgres has already committed status=escalated, and THAT is the durable
        signal — if this XADD fails, a sweep can re-enqueue every escalated
        complaint, exactly like a stale Qdrant index is rebuilt by the backfill.
        Raising here would redeliver the classification message and re-run a
        multi-second LLM call to redo work that's already committed.
        """
        try:
            await enqueue_resolution(
                self.redis, complaint_id, stream=self.settings.resolution_queue
            )
            logger.info(
                "complaint %s is high-priority (urgency=%d sentiment=%s) — queued "
                "for resolution agent",
                complaint_id,
                cls.urgency,
                cls.sentiment,
            )
        except Exception:
            logger.exception(
                "failed to enqueue %s for resolution (status=escalated is durable; "
                "re-enqueue via sweep)",
                complaint_id,
            )

    async def _index(
        self,
        complaint_id: uuid.UUID,
        narrative: str,
        complaint: Complaint,
        cls: Any,
    ) -> None:
        """Best-effort: embed + upsert to Qdrant with classification in the payload.

        Postgres is already committed, so a Qdrant failure here is logged and
        swallowed rather than re-running the (expensive) classification.
        """
        try:
            vector = await asyncio.to_thread(embed_text, narrative)
            payload = {
                "product": complaint.product,
                "issue": complaint.issue,
                "company": complaint.company,
                "state": complaint.state,
                "sentiment": cls.sentiment,
                "intent": cls.intent,
                "urgency": cls.urgency,
                "status": complaint.status.value,
                "key_entities": [e.model_dump() for e in cls.key_entities],
            }
            await asyncio.to_thread(
                self.vector_store.upsert_complaint, str(complaint_id), vector, payload
            )
        except Exception:
            logger.exception(
                "vector upsert failed for %s (postgres committed); index will lag",
                complaint_id,
            )

    async def _update_graph(self, complaint: Complaint) -> None:
        """Best-effort: fold this complaint's company/product/issue into the graph.

        Same contract as ``_index``: Postgres is already committed, so a Neo4j
        hiccup is logged and swallowed rather than re-running the (expensive)
        classification. Keeping the graph's HAS_COMPLAINTS_ABOUT / HAS_ISSUE
        counters live as complaints arrive is the point — a stale graph is
        recoverable by re-running the seeder; a lost classification is not.

        We pass only the structured CFPB fields. Mapping the classifier's
        free-text regulation entities ("15 USC 1681") to Regulation node ids
        ("FCRA") for VIOLATED edges needs a normalization table and is deferred
        with the regulation-set expansion (tracked in the phase notes).
        """
        if self.graph_store is None:
            return
        try:
            await self.graph_store.upsert_complaint_entities(
                complaint_id=str(complaint.id),
                company=complaint.company,
                product=complaint.product,
                issue=complaint.issue,
            )
        except Exception:
            logger.exception(
                "graph upsert failed for %s (postgres committed); graph aggregates will lag",
                complaint.id,
            )


async def main() -> None:
    """Wire real dependencies and run until SIGINT/SIGTERM."""
    logging.basicConfig(
        level=default_settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    redis_client = aioredis.from_url(default_settings.redis_url, decode_responses=True)
    worker = ClassificationWorker(
        redis_client=redis_client,
        classifier=Classifier(),
        vector_store=get_default_store(),
        graph_store=get_default_graph_store(),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.request_stop)

    try:
        await worker.run()
    finally:
        await redis_client.aclose()
        logger.info("classification worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
