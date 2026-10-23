"""Classify durable PostgreSQL jobs; persist output and downstream intents together."""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime
from typing import Any

from app.config import Settings
from app.config import settings as default_settings
from app.database import AsyncSessionLocal
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.services.case_transitions import advance_case, lock_case
from app.services.classifier import Classifier
from app.services.jobs import complete_job, enqueue_job, fail_job, lock_owned_job, supersede_job
from app.services.llmops_tracker import LLMOpsTracker
from app.workers.durable import DurableWorker

logger = logging.getLogger(__name__)
_SENTIMENT_WEIGHT = {"neutral": 0.0, "negative": 0.5, "extreme_negative": 1.0}


def compute_priority_score(urgency: int, sentiment: str) -> float:
    raw = 0.7 * (urgency - 1) / 4 + 0.3 * _SENTIMENT_WEIGHT.get(sentiment, 0.5)
    return round(max(0.0, min(1.0, raw)), 4)


def is_high_priority(urgency: int, sentiment: str) -> bool:
    return urgency >= 4 or sentiment == "extreme_negative"


class ClassificationWorker(DurableWorker):
    kind = JobKind.classification

    def __init__(
        self,
        classifier: Classifier,
        *,
        tracker: LLMOpsTracker | None = None,
        settings: Settings | None = None,
        session_factory: Any = AsyncSessionLocal,
    ) -> None:
        super().__init__(settings=settings, session_factory=session_factory)
        self.classifier = classifier
        self.tracker = tracker or LLMOpsTracker()

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
            await session.commit()
        outcome = await asyncio.to_thread(
            self.classifier.classify,
            complaint.narrative,
            product=complaint.product,
            issue=complaint.issue,
            company=complaint.company,
        )
        async with self.session_factory() as session:
            complaint = await lock_case(session, job.complaint_id)
            owned = await lock_owned_job(session, job.id, job.lease_token)
            if owned is None:
                return
            if complaint is None or not self._eligible(complaint, job):
                supersede_job(owned)
                await session.commit()
                return
            cls = outcome.classification
            succeeded = outcome.succeeded and cls is not None
            now = datetime.utcnow()
            complaint.classification_attempted_at = now
            complaint.classification_latency_ms = outcome.latency_ms
            if succeeded:
                complaint.sentiment = cls.sentiment
                complaint.intent = cls.intent
                complaint.urgency = cls.urgency
                complaint.priority_score = compute_priority_score(cls.urgency, cls.sentiment)
                high = is_high_priority(cls.urgency, cls.sentiment)
                complaint.status = ComplaintStatus.escalated if high else ComplaintStatus.classified
                complaint.classification_outcome = ClassificationState.succeeded
                complaint.classification_source = outcome.provider
                complaint.classification_was_fallback = outcome.is_fallback
                complaint.classification_stale = False
                complaint.classification_error = None
                complaint.classified_at = now
                advance_case(complaint, inputs_changed=True)
                session.add(complaint)
                if high:
                    await enqueue_job(
                        session,
                        complaint,
                        JobKind.resolution,
                        idempotency_key=f"classification:{job.id}:resolution",
                        max_attempts=self.settings.job_max_attempts,
                    )
                await enqueue_job(
                    session,
                    complaint,
                    JobKind.index,
                    idempotency_key=f"classification:{job.id}:index",
                    payload={"key_entities": [entity.model_dump() for entity in cls.key_entities]},
                    max_attempts=self.settings.job_max_attempts,
                )
                complete_job(owned)
            else:
                fail_job(owned, "providers_unavailable", retryable=True)
                complaint.classification_outcome = ClassificationState.unavailable
                complaint.classification_error = "providers_unavailable"
                complaint.classification_stale = any(
                    value is not None
                    for value in (complaint.sentiment, complaint.intent, complaint.urgency)
                )
                if owned.status == JobStatus.failed and complaint.status == ComplaintStatus.pending:
                    complaint.status = ComplaintStatus.needs_review
                # A failed attempt supplies no new classification input/version.
                # Retrying this immutable intent remains valid until an actual
                # user transition or successful classification changes the case.
                session.add(complaint)
            self.tracker.record(
                session,
                operation="classify" if succeeded else "classify_unavailable",
                provider=outcome.provider,
                model=outcome.model,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                latency_ms=outcome.latency_ms,
                was_fallback=outcome.is_fallback,
                complaint_id=complaint.id,
            )
            await session.commit()

    @staticmethod
    def _eligible(complaint: Complaint, job: Job) -> bool:
        return (
            (complaint.case_version, complaint.input_revision)
            == (job.case_version, job.input_revision)
            and complaint.status
            in {
                ComplaintStatus.pending,
                ComplaintStatus.classified,
                ComplaintStatus.escalated,
                ComplaintStatus.needs_review,
            }
            and complaint.classification_source != "manual"
        )


async def main() -> None:
    logging.basicConfig(level=default_settings.log_level)
    worker = ClassificationWorker(classifier=Classifier())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.request_stop)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
