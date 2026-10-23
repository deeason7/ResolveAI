"""Transactional admission and bounded HTTP request identities for durable jobs."""

from __future__ import annotations

import hashlib
import json
import uuid

from fastapi import HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.models.job import Job
from app.services.jobs import JobCapacityExceeded, reserve_capacity


def request_identity(actor: uuid.UUID, route: str, key: str | None) -> str | None:
    if key is None:
        return None
    value = json.dumps([str(actor), route, key], separators=(",", ":"))
    return "request:" + hashlib.sha256(value.encode()).hexdigest()


def fingerprint(payload: dict) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(value.encode()).hexdigest()


async def admit(session: AsyncSession, count: int = 0) -> None:
    """Lock admission before case rows; count zero lets full-queue replays succeed."""
    try:
        await reserve_capacity(session, count, max_pending=settings.job_max_pending)
    except JobCapacityExceeded:
        raise HTTPException(
            503, "Job capacity reached; no changes were accepted. Try later."
        ) from None


async def replay(session: AsyncSession, key: str | None, request_fingerprint: str) -> Job | None:
    if key is None:
        return None
    job = (await session.exec(select(Job).where(Job.idempotency_key == key))).one_or_none()
    if job is not None and job.request_fingerprint != request_fingerprint:
        raise HTTPException(409, "Idempotency-Key was already used for a different request")
    return job
