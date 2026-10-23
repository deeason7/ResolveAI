"""Run resolution agents from PostgreSQL jobs with exact-version lease fencing."""

from __future__ import annotations

import asyncio
import logging
import signal
import uuid
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy import func
from sqlmodel import select

from app.config import Settings
from app.config import settings as default_settings
from app.database import AsyncSessionLocal
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.models.resolution import GuardrailStatus, Resolution
from app.schemas.classification import ComplaintClassification
from app.services.agent.orchestrator import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    AgentResult,
    ResolutionAgent,
)
from app.services.case_transitions import advance_case, lock_case
from app.services.graph_store import GraphStore, get_default_graph_store
from app.services.guardrails import GuardrailEngine
from app.services.jobs import complete_job, fail_job, lock_owned_job, supersede_job
from app.services.llm_client import LLMClient, get_llm_client
from app.services.llmops_tracker import LLMOpsTracker
from app.services.vector_store import VectorStore, get_default_store
from app.workers.durable import DurableWorker

logger = logging.getLogger(__name__)
OPERATION_DRAFT = "draft_response"
OPERATION_TONE = "tone_check"
_OUTCOME_MAP = {
    STATUS_PASSED: (GuardrailStatus.passed, ComplaintStatus.draft_ready),
    STATUS_FAILED: (GuardrailStatus.failed, ComplaintStatus.needs_review),
    STATUS_UNAVAILABLE: (GuardrailStatus.escalated, ComplaintStatus.needs_review),
}


class _AgentLike(Protocol):
    async def run(self) -> AgentResult: ...


class ResolutionWorker(DurableWorker):
    kind = JobKind.resolution

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        guardrails: GuardrailEngine | None = None,
        tracker: LLMOpsTracker | None = None,
        settings: Settings | None = None,
        session_factory: Any = AsyncSessionLocal,
        agent_factory: Any = None,
    ) -> None:
        super().__init__(settings=settings, session_factory=session_factory)
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.guardrails = guardrails or GuardrailEngine(llm_client=llm_client)
        self.tracker = tracker or LLMOpsTracker()
        self.agent_factory = agent_factory or self._build_agent

    def _build_agent(
        self,
        complaint: Complaint,
        classification: ComplaintClassification,
        *,
        initial_feedback: str | None = None,
        previous_draft_text: str | None = None,
    ) -> _AgentLike:
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

    async def process_job(self, job: Job) -> None:
        async with self.session_factory() as session:
            complaint = await session.get(Complaint, job.complaint_id)
            owned = await lock_owned_job(session, job.id, job.lease_token)
            if owned is None:
                return
            if complaint is None or not self._eligible(complaint, job):
                supersede_job(owned)
                await session.commit()
                return
            classification = self._rebuild_classification(complaint)
            if classification is None:
                fail_job(owned, "classification_unavailable", retryable=False)
                await session.commit()
                return
            feedback = (job.payload or {}).get("feedback")
            previous_draft = (
                await self._latest_draft_text(session, complaint.id) if feedback else None
            )
            await session.commit()
        agent = self.agent_factory(
            complaint, classification, initial_feedback=feedback, previous_draft_text=previous_draft
        )
        result: AgentResult = await agent.run()
        await self._persist(job, result)

    async def _persist(self, job: Job, result: AgentResult) -> Resolution | None:
        async with self.session_factory() as session:
            complaint = await lock_case(session, job.complaint_id)
            owned = await lock_owned_job(session, job.id, job.lease_token)
            if owned is None:
                return None
            if complaint is None or not self._eligible(complaint, job):
                supersede_job(owned)
                await session.commit()
                return None
            resolution = None
            if result.drafted is not None:
                guardrail_status, complaint.status = _OUTCOME_MAP[result.status]
                resolution = Resolution(
                    complaint_id=complaint.id,
                    version=await self._next_version(session, complaint.id),
                    input_revision=complaint.input_revision,
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
                advance_case(complaint)
                complete_job(owned)
            else:
                fail_job(owned, "providers_unavailable", retryable=True)
                if owned.status == JobStatus.failed:
                    complaint.status = ComplaintStatus.needs_review
                # No new draft or input exists. Preserve the intent's versions
                # so an authorized retry can continue after provider recovery.
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
    def _eligible(complaint: Complaint, job: Job) -> bool:
        return (
            (complaint.case_version, complaint.input_revision)
            == (job.case_version, job.input_revision)
            and complaint.status
            in {
                ComplaintStatus.escalated,
                ComplaintStatus.agent_triggered,
                ComplaintStatus.needs_review,
            }
            and not complaint.classification_stale
            and complaint.classification_outcome != ClassificationState.unavailable
        )

    @staticmethod
    def _rebuild_classification(complaint: Complaint) -> ComplaintClassification | None:
        """Rebuild the schema the agent expects from the persisted row fields.

        Day 13 persisted sentiment/intent/urgency on the complaint; entities and
        reasoning went to the Qdrant payload and the model's response only. The
        drafting prompt needs the row fields; the rest degrades gracefully.
        Returns None when the row was never classified (or holds junk) — that's
        a poison message, not a crash.
        """
        if (
            complaint.classification_outcome == ClassificationState.unavailable
            or complaint.classification_stale
            or complaint.sentiment is None
            or complaint.intent is None
            or complaint.urgency is None
        ):
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
        """Allocate while the caller holds the parent case lock."""
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
    logging.basicConfig(level=default_settings.log_level)
    llm_client = get_llm_client()
    worker = ResolutionWorker(
        llm_client=llm_client,
        vector_store=get_default_store(),
        graph_store=get_default_graph_store(),
        guardrails=GuardrailEngine(llm_client=llm_client),
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.request_stop)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
