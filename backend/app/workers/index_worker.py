"""Retry rebuildable graph/vector writes independently of canonical inference.

External writes can repeat. Qdrant upserts use the complaint ID, and graph
contribution receipts suppress repeated counter increments. Index payloads carry
the input revision; PostgreSQL remains authoritative for case and review state.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from app.config import Settings
from app.config import settings as default_settings
from app.database import AsyncSessionLocal
from app.models.complaint import Complaint
from app.models.job import Job, JobKind
from app.services.case_transitions import lock_case
from app.services.embedder import embed_text
from app.services.graph_store import GraphStore, get_default_graph_store
from app.services.jobs import complete_job, lock_owned_job, request_index_repair, supersede_job
from app.services.vector_store import VectorStore, get_default_store
from app.workers.durable import DurableWorker


class IndexWorker(DurableWorker):
    kind = JobKind.index

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        *,
        settings: Settings | None = None,
        session_factory: Any = AsyncSessionLocal,
    ) -> None:
        super().__init__(settings=settings, session_factory=session_factory)
        self.vector_store = vector_store
        self.graph_store = graph_store

    async def process_job(self, job: Job) -> None:
        async with self.session_factory() as session:
            complaint = await session.get(Complaint, job.complaint_id)
            owned = await lock_owned_job(session, job.id, job.lease_token)
            if owned is None:
                return
            if complaint is None or complaint.input_revision != job.input_revision:
                supersede_job(owned)
                await session.commit()
                return
            await session.commit()
        vector = await asyncio.to_thread(embed_text, complaint.narrative)
        payload = {
            "product": complaint.product,
            "issue": complaint.issue,
            "company": complaint.company,
            "state": complaint.state,
            "sentiment": complaint.sentiment,
            "intent": complaint.intent,
            "urgency": complaint.urgency,
            "input_revision": complaint.input_revision,
            "key_entities": (job.payload or {}).get("key_entities", []),
        }
        succeeded = False
        try:
            await asyncio.to_thread(
                self.vector_store.upsert_complaint, str(complaint.id), vector, payload
            )
            await self.graph_store.upsert_complaint_entities(
                complaint_id=str(complaint.id),
                company=complaint.company,
                product=complaint.product,
                issue=complaint.issue,
            )
            succeeded = True
        finally:
            # A partial external write can be stale even when a later store fails.
            await self._reconcile_write(job, succeeded=succeeded)

    async def _reconcile_write(self, job: Job, *, succeeded: bool) -> None:
        async with self.session_factory() as session:
            current = await lock_case(session, job.complaint_id)
            owned = await lock_owned_job(session, job.id, job.lease_token)
            if current is not None and current.input_revision != job.input_revision:
                if owned is not None:
                    supersede_job(owned)
                # External stores cannot participate in the lease transaction.
                # Even a lost lease can finish an old write after its replacement;
                # reopen current indexing so that stale data is not left silently.
                await request_index_repair(
                    session, current, max_attempts=self.settings.job_max_attempts
                )
            elif owned is not None:
                if current is None:
                    supersede_job(owned)
                elif succeeded:
                    complete_job(owned)
            await session.commit()


async def main() -> None:
    logging.basicConfig(level=default_settings.log_level)
    worker = IndexWorker(get_default_store(), get_default_graph_store())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.request_stop)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
