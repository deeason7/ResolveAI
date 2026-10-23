"""Short PostgreSQL claims, renewable leases, and fenced completion.

Provider calls can repeat after a crash. Only a live lease may persist the result.
No database transaction remains open while waiting for external work.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

import httpx
from neo4j.exceptions import DriverError, Neo4jError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.config import Settings
from app.config import settings as default_settings
from app.database import AsyncSessionLocal
from app.models.job import Job, JobKind
from app.services.jobs import claim_job, fail_job, heartbeat_job, lock_owned_job

logger = logging.getLogger(__name__)


def retryable_failure(exc: Exception) -> bool:
    # Malformed input and programming/configuration failures need operator repair.
    # Unknown failures are retained visibly instead of repeatedly invoking providers.
    if isinstance(exc, ResponseHandlingException):
        return retryable_failure(exc.source)
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, (DriverError, Neo4jError)):
        return exc.is_retryable()
    return isinstance(
        exc, (TimeoutError, ConnectionError, OSError, DBAPIError, httpx.TransportError)
    ) and not isinstance(exc, IntegrityError)


def failure_code(exc: Exception) -> str:
    """Persist bounded categories; provider exceptions can contain customer data."""
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            OSError,
            httpx.HTTPError,
            DriverError,
            Neo4jError,
            ResponseHandlingException,
            UnexpectedResponse,
        ),
    ):
        return "dependency_unavailable"
    if isinstance(exc, DBAPIError):
        return "database_unavailable"
    return "worker_error"


class DurableWorker:
    kind: JobKind

    def __init__(
        self, *, settings: Settings | None = None, session_factory: Any = AsyncSessionLocal
    ) -> None:
        self.settings = settings or default_settings
        self.session_factory = session_factory
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                processed = await self.run_once()
            except Exception as exc:
                # A failed claim remains in PostgreSQL and is retried on the next poll.
                logger.warning("%s poll failed: %s", self.kind.value, failure_code(exc))
                processed = False
            if not processed:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), self.settings.job_poll_seconds)

    async def run_once(self) -> bool:
        async with self.session_factory() as session:
            job = await claim_job(session, self.kind, lease_seconds=self.settings.job_lease_seconds)
            await session.commit()
        if job is None:
            return False
        heartbeat = asyncio.create_task(self._heartbeat(job))
        try:
            await self.process_job(job)
        except Exception as exc:
            logger.warning("%s job %s failed: %s", self.kind.value, job.id, failure_code(exc))
            # If this write also fails, expiry recovers the abandoned claim.
            async with self.session_factory() as session:
                owned = await lock_owned_job(session, job.id, job.lease_token)
                if owned is not None:
                    fail_job(owned, failure_code(exc), retryable=retryable_failure(exc))
                    await session.commit()
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        return True

    async def _heartbeat(self, job: Job) -> None:
        while True:
            await asyncio.sleep(self.settings.job_heartbeat_seconds)
            try:
                async with self.session_factory() as session:
                    owned = await heartbeat_job(
                        session,
                        job.id,
                        job.lease_token,
                        lease_seconds=self.settings.job_lease_seconds,
                    )
                    await session.commit()
                if not owned:
                    return
            except Exception as exc:
                # Uncertain renewal never authorizes persistence: completion rechecks
                # the lease using database time even when this heartbeat failed.
                logger.warning("job %s heartbeat failed: %s", job.id, failure_code(exc))

    async def process_job(self, job: Job) -> None:
        raise NotImplementedError
