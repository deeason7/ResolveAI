"""Background worker: drain the resolution stream, run the agent, persist.

Flow per message (spec Day 23-24):
    Redis stream  ->  fetch complaint from Postgres  ->  rebuild classification
                  ->  mark agent_triggered (visible to the triage queue)
                  ->  ResolutionAgent: gather context, draft, guardrails, retry
                  ->  write Resolution + complaint status + one LLMLog per
                      draft call and per judge call, all in ONE transaction.

The consumer-group skeleton (group create, blocking read, ack-on-settled)
deliberately mirrors the classification worker: same at-least-once + PEL
semantics, same poison-vs-transient ack rules. Two copies is tolerable; if a
third stream consumer ever appears, extract a shared base.

Status choreography (see ComplaintStatus):
    escalated -> agent_triggered -> draft_ready    guardrails passed
                                 -> needs_review   retries exhausted or LLM down

``agent_triggered`` commits in its own transaction *before* the agent runs, so
the dashboard can show in-flight work and a crash mid-run is observable (the
complaint sticks at agent_triggered with the message still pending in the PEL).

Redelivery is at-least-once, so a crash after the final commit but before the
ack can produce a duplicate draft version. That's an extra row in the revision
history, not consumer-facing damage — the same tradeoff the graph counters
made — and the draft_ready guard below catches the common stale-redelivery
case.

Runs at concurrency 1: drafting + judging is several seconds of LLM time per
complaint, and the volume of escalated complaints is a trickle, not a flood.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import uuid
from datetime import datetime
from typing import Any, Protocol

import redis.asyncio as aioredis
from pydantic import ValidationError
from redis.exceptions import ResponseError
from sqlalchemy import func
from sqlmodel import select

from app.config import Settings
from app.config import settings as default_settings
from app.database import AsyncSessionLocal
from app.models.complaint import Complaint, ComplaintStatus
from app.models.resolution import GuardrailStatus, Resolution
from app.schemas.classification import ComplaintClassification
from app.services.agent.orchestrator import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    AgentResult,
    ResolutionAgent,
)
from app.services.graph_store import GraphStore, get_default_graph_store
from app.services.guardrails import GuardrailEngine
from app.services.llm_client import LLMClient, get_llm_client
from app.services.llmops_tracker import LLMOpsTracker
from app.services.vector_store import VectorStore, get_default_store
from app.workers.stream_utils import reclaim_stale_messages

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "resolvers"
READ_COUNT = 1  # one complaint at a time — each costs seconds of LLM calls

# LLMOps operation names. Draft and judge calls have different cost/latency
# profiles, so they get distinct operations the dashboard can split on.
OPERATION_DRAFT = "draft_response"
OPERATION_TONE = "tone_check"

# Agent verdict -> (resolution row status, complaint status).
# UNAVAILABLE maps to GuardrailStatus.escalated: there IS a draft but no
# guardrail verdict was possible, which is exactly what that value means.
_OUTCOME_MAP: dict[str, tuple[GuardrailStatus, ComplaintStatus]] = {
    STATUS_PASSED: (GuardrailStatus.passed, ComplaintStatus.draft_ready),
    STATUS_FAILED: (GuardrailStatus.failed, ComplaintStatus.needs_review),
    STATUS_UNAVAILABLE: (GuardrailStatus.escalated, ComplaintStatus.needs_review),
}

# A no-feedback redelivery for these statuses is stale: the work it asked for
# is already done (or the case is closed). Explicit feedback always reprocesses.
# Value strings, not enum members: the status column is a plain VARCHAR, so rows
# loaded from the DB carry str values even though we assign enum members.
_SKIP_WITHOUT_FEEDBACK = frozenset(
    {ComplaintStatus.draft_ready.value, ComplaintStatus.resolved.value}
)


class _AgentLike(Protocol):
    """What the worker needs from an agent — lets tests stub the whole pipeline."""

    async def run(self) -> AgentResult: ...


async def enqueue_resolution(
    redis_client: aioredis.Redis,
    complaint_id: uuid.UUID | str,
    *,
    feedback: str | None = None,
    stream: str | None = None,
) -> str:
    """Producer side of the stream contract: XADD a complaint id for resolving.

    Shared by every producer (classification worker on escalation, the generate
    route, the reject route) so the message shape has one definition.
    ``feedback`` carries a human reviewer's rejection notes into the next draft.
    """
    stream = stream or default_settings.resolution_queue
    fields: dict[str, str] = {"complaint_id": str(complaint_id)}
    if feedback:
        fields["feedback"] = feedback
    return await redis_client.xadd(stream, fields)


class ResolutionWorker:
    """Consumes complaint ids from the resolution stream and runs the agent.

    Every collaborator is injected. ``agent_factory`` builds the per-complaint
    agent — production uses the default (real ``ResolutionAgent`` wired to this
    worker's stores and guardrails); tests inject a factory returning a stub so
    no tool, LLM, or guardrail code runs.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        *,
        llm_client: LLMClient | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        guardrails: GuardrailEngine | None = None,
        tracker: LLMOpsTracker | None = None,
        settings: Settings | None = None,
        session_factory: Any = AsyncSessionLocal,
        consumer_name: str | None = None,
        agent_factory: Any = None,
    ) -> None:
        self.redis = redis_client
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.guardrails = guardrails or GuardrailEngine(llm_client=llm_client)
        self.tracker = tracker or LLMOpsTracker()
        self.settings = settings or default_settings
        self.session_factory = session_factory
        self.stream = self.settings.resolution_queue
        self.group = CONSUMER_GROUP
        self.consumer = consumer_name or f"{socket.gethostname()}-{os.getpid()}"
        self.agent_factory = agent_factory or self._build_agent
        self._stop = asyncio.Event()

    def _build_agent(
        self,
        complaint: Complaint,
        classification: ComplaintClassification,
        *,
        initial_feedback: str | None = None,
        previous_draft_text: str | None = None,
    ) -> _AgentLike:
        """Default factory: a real agent wired to this worker's collaborators."""
        return ResolutionAgent(
            complaint,
            classification,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            llm_client=self.llm_client,
            guardrails=self.guardrails,
            initial_feedback=initial_feedback,
            previous_draft_text=previous_draft_text,
        )

    async def ensure_group(self) -> None:
        """Create the consumer group (and stream) if absent; idempotent."""
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
            "resolution worker up: stream=%s group=%s consumer=%s",
            self.stream,
            self.group,
            self.consumer,
        )
        cycle = 0
        while not self._stop.is_set():
            await self._tick(cycle)
            cycle += 1

    async def _tick(self, cycle: int) -> None:
        """One loop iteration: a periodic PEL sweep, then one blocking read.

        Mirrors the classification worker (see its ``_tick``): split out of
        ``run`` for testability, with ``worker_reclaim_every`` / ``worker_block_ms``
        thinning idle Redis command volume to fit a command-billed managed tier.
        """
        if cycle % self.settings.worker_reclaim_every == 0:
            # Rescue anything a dead/slow consumer left stranded in the PEL —
            # XREADGROUP '>' below would never redeliver those on its own.
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
            block=self.settings.worker_block_ms,
        )
        if not resp:
            return
        for _stream, messages in resp:
            for message_id, fields in messages:
                await self._handle(message_id, fields)

    async def _handle(self, message_id: str, fields: dict[str, str]) -> None:
        """Process one message; ack on success and on poison, never on transient."""
        try:
            await self.process_message(fields)
        except Exception:
            logger.exception("transient failure on %s; leaving unacked for retry", message_id)
            return
        await self.redis.xack(self.stream, self.group, message_id)

    async def process_message(self, fields: dict[str, str]) -> None:
        """Run the agent for one complaint and persist the outcome.

        Returns normally on success *and* on poison messages (bad id, missing or
        unclassified complaint) so they get acked. Raises only on retryable
        infrastructure failures.
        """
        raw_id = fields.get("complaint_id")
        try:
            complaint_id = uuid.UUID(raw_id) if raw_id else None
        except ValueError:
            complaint_id = None
        if complaint_id is None:
            logger.warning("dropping message with bad complaint_id: %r", raw_id)
            return  # poison -> ack

        feedback = fields.get("feedback") or None

        async with self.session_factory() as session:
            complaint = await session.get(Complaint, complaint_id)
            if complaint is None:
                logger.warning("complaint %s not found; dropping", complaint_id)
                return  # poison -> ack

            # EnumString round-trips status as the enum, so .value is always safe.
            status_value = complaint.status.value
            if status_value in _SKIP_WITHOUT_FEEDBACK and not feedback:
                logger.info(
                    "complaint %s already %s; dropping stale trigger",
                    complaint_id,
                    status_value,
                )
                return  # duplicate/stale delivery -> ack

            classification = self._rebuild_classification(complaint)
            if classification is None:
                logger.warning("complaint %s has no usable classification; dropping", complaint_id)
                return  # poison -> ack (the API refuses these upfront; belt and braces)

            previous_draft_text: str | None = None
            if feedback:
                previous_draft_text = await self._latest_draft_text(session, complaint_id)

            # Own transaction, before the slow part: the triage queue shows the
            # agent working, and a crash mid-run leaves a visible in-flight state.
            complaint.status = ComplaintStatus.agent_triggered
            complaint.updated_at = datetime.utcnow()
            session.add(complaint)
            await session.commit()

            agent = self.agent_factory(
                complaint,
                classification,
                initial_feedback=feedback,
                previous_draft_text=previous_draft_text,
            )
            result: AgentResult = await agent.run()

            await self._persist(session, complaint, result)

        logger.info(
            "complaint %s resolved by agent: status=%s attempts=%d drafts=%d judge_calls=%d",
            complaint_id,
            result.status,
            result.attempts,
            len(result.llm_calls),
            len(result.judge_calls),
        )

    async def _persist(
        self, session: Any, complaint: Complaint, result: AgentResult
    ) -> Resolution | None:
        """One transaction: Resolution row (if drafted), status, and all LLMLogs.

        A resolution and its cost/audit trail must land together — a Resolution
        with no LLMLogs under-reports spend, and orphan LLMLogs point at a draft
        that doesn't exist. Same consistency boundary as classification.
        """
        resolution: Resolution | None = None
        if result.drafted is not None:
            guardrail_status, complaint_status = _OUTCOME_MAP[result.status]
            resolution = Resolution(
                complaint_id=complaint.id,
                version=await self._next_version(session, complaint.id),
                draft_text=result.drafted.response_text,
                guardrail_status=guardrail_status,
                guardrail_notes=result.guardrail_feedback or None,
                guardrail_violations=(
                    [v.model_dump() for v in result.guardrail_violations]
                    if result.guardrail_violations
                    else None
                ),
                reasoning_summary=result.reasoning_summary or None,
            )
            session.add(resolution)
        else:
            # Nothing drafted at all (LLM down on attempt 1) — no row to write,
            # but the complaint still needs a human.
            complaint_status = ComplaintStatus.needs_review

        complaint.status = complaint_status
        complaint.updated_at = datetime.utcnow()
        session.add(complaint)

        for call in result.llm_calls:
            self.tracker.record(
                session,
                operation=OPERATION_DRAFT,
                provider=call.provider,
                model=call.model,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                latency_ms=call.latency_ms,
                was_fallback=call.is_fallback,
                complaint_id=complaint.id,
            )
        for judge in result.judge_calls:
            self.tracker.record(
                session,
                operation=OPERATION_TONE,
                provider=judge.provider,
                model=judge.model,
                prompt_tokens=judge.prompt_tokens,
                completion_tokens=judge.completion_tokens,
                latency_ms=judge.latency_ms,
                was_fallback=judge.is_fallback,
                complaint_id=complaint.id,
            )

        await session.commit()
        return resolution

    @staticmethod
    def _rebuild_classification(complaint: Complaint) -> ComplaintClassification | None:
        """Rebuild the schema the agent expects from the persisted row fields.

        Day 13 persisted sentiment/intent/urgency on the complaint; entities and
        reasoning went to the Qdrant payload and the model's response only. The
        drafting prompt needs the row fields; the rest degrades gracefully.
        Returns None when the row was never classified (or holds junk) — that's
        a poison message, not a crash.
        """
        if complaint.sentiment is None or complaint.intent is None or complaint.urgency is None:
            return None
        try:
            return ComplaintClassification(
                sentiment=complaint.sentiment,
                intent=complaint.intent,
                urgency=complaint.urgency,
                key_entities=[],
                reasoning="Rebuilt from stored classification; original reasoning not persisted.",
            )
        except ValidationError:
            logger.warning("complaint %s has out-of-vocabulary classification fields", complaint.id)
            return None

    @staticmethod
    async def _next_version(session: Any, complaint_id: uuid.UUID) -> int:
        """1 + the highest existing version for this complaint (1 when none).

        Safe at concurrency 1 (our deployment); two workers racing the same
        complaint could pick the same version — add a unique constraint before
        ever scaling out.
        """
        stmt = select(func.max(Resolution.version)).where(Resolution.complaint_id == complaint_id)
        current = (await session.exec(stmt)).one()
        return (current or 0) + 1

    @staticmethod
    async def _latest_draft_text(session: Any, complaint_id: uuid.UUID) -> str | None:
        """Text of the newest draft — quoted in the regeneration prompt on reject."""
        stmt = (
            select(Resolution)
            .where(Resolution.complaint_id == complaint_id)
            .order_by(Resolution.version.desc())
            .limit(1)
        )
        row = (await session.exec(stmt)).first()
        return row.draft_text if row else None


async def main() -> None:
    """Wire real dependencies and run until SIGINT/SIGTERM."""
    logging.basicConfig(
        level=default_settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    redis_client = aioredis.from_url(default_settings.redis_url, decode_responses=True)
    llm_client = get_llm_client()
    worker = ResolutionWorker(
        redis_client=redis_client,
        llm_client=llm_client,
        vector_store=get_default_store(),
        graph_store=get_default_graph_store(),
        guardrails=GuardrailEngine(llm_client=llm_client),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.request_stop)

    try:
        await worker.run()
    finally:
        await redis_client.aclose()
        logger.info("resolution worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
