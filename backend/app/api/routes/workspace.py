"""
Pipeline Workspace endpoints — the live control surface over the
classify -> resolve pipeline:

  GET  /workspace/board                  — stage counts + stream telemetry
  POST /workspace/enqueue/classification — push N pending complaints to classify
  POST /workspace/enqueue/resolution     — push N escalated complaints to the agent

The board reads stage counts from complaint.status (the durable signal the
workers write transactionally) and overlays best-effort Redis stream state
(in-flight / consumers / lag). The enqueue routes reuse the same producer
functions the submit and /generate routes use, so there's a single definition
of each stream's message shape.
"""

from __future__ import annotations

import logging
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.core.deps import get_current_user, get_redis
from app.database import get_session
from app.middleware.rate_limit import limiter
from app.models.complaint import Complaint, ComplaintStatus
from app.models.user import User
from app.schemas.workspace import EnqueueResult, StreamInfo, WorkspaceBoard
from app.workers.classification_worker import enqueue_complaint
from app.workers.resolution_worker import enqueue_resolution

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workspace", tags=["workspace"])

# Cap per enqueue so one click can't flood a stream with the whole 200K backlog.
_BATCH_MAX = 500

# These two routes each kick off real work (up to _BATCH_MAX XADDs, and for
# resolution a same-transaction status flip on every row). They ride the global
# 200/min like everything else, but also get a tighter dedicated cap so a stuck
# client or a refresh loop can't machine-gun batches. The board GET stays on the
# global limit — it's a cheap read meant for polling.
_ENQUEUE_RATE_LIMIT = "10/minute"


async def _stream_info(redis: aioredis.Redis, stream: str) -> StreamInfo:
    """Sum pending / consumers / lag across a stream's consumer groups.

    Wrapped in try/except: the stream and its groups don't exist until a worker
    runs ensure_group, and the XINFO lag field is Redis 7+. A miss returns
    zeros / None so the board never fails on the transient telemetry layer.
    """
    in_flight = consumers = 0
    lag: int | None = None
    try:
        for group in await redis.xinfo_groups(stream):
            in_flight += int(group.get("pending", 0) or 0)
            consumers += int(group.get("consumers", 0) or 0)
            group_lag = group.get("lag")
            if group_lag is not None:
                lag = (lag or 0) + int(group_lag)
    except Exception:  # noqa: BLE001 — best-effort telemetry, never fatal
        logger.debug("stream info unavailable for %s (no consumer group yet?)", stream)
    return StreamInfo(name=stream, in_flight=in_flight, consumers=consumers, lag=lag)


@router.get("/board", response_model=WorkspaceBoard)
async def board(
    session: AsyncSession = Depends(get_session),
    redis: aioredis.Redis = Depends(get_redis),
    _: User = Depends(get_current_user),
) -> WorkspaceBoard:
    """Live pipeline board: one count per complaint status, plus stream state."""
    rows = (
        await session.exec(
            select(Complaint.status, func.count(Complaint.id)).group_by(Complaint.status)
        )
    ).all()
    # status round-trips as the enum (EnumString), so key on .value — str(enum)
    # would yield "ComplaintStatus.pending" on 3.11 and miss every lookup.
    counts = {status_value.value: count for status_value, count in rows}

    def n(stage: ComplaintStatus) -> int:
        return int(counts.get(stage.value, 0))

    return WorkspaceBoard(
        pending=n(ComplaintStatus.pending),
        classified=n(ComplaintStatus.classified),
        escalated=n(ComplaintStatus.escalated),
        agent_triggered=n(ComplaintStatus.agent_triggered),
        draft_ready=n(ComplaintStatus.draft_ready),
        needs_review=n(ComplaintStatus.needs_review),
        resolved=n(ComplaintStatus.resolved),
        total=sum(int(c) for c in counts.values()),
        classification_stream=await _stream_info(redis, settings.classification_queue),
        resolution_stream=await _stream_info(redis, settings.resolution_queue),
    )


@router.post("/enqueue/classification", response_model=EnqueueResult)
@limiter.limit(_ENQUEUE_RATE_LIMIT)
async def enqueue_classification(
    request: Request,  # required by slowapi's per-route limiter for the key func
    limit: int = Query(default=50, ge=1, le=_BATCH_MAX),
    session: AsyncSession = Depends(get_session),
    redis: aioredis.Redis = Depends(get_redis),
    _: User = Depends(get_current_user),
) -> EnqueueResult:
    """Push up to `limit` still-pending complaints onto the classification stream.

    Status stays `pending` — that's the durable signal; the worker flips it to
    classified / escalated when it lands. Re-running is safe (at-least-once): a
    complaint the worker hasn't reached yet may be enqueued again and simply
    re-classified, never corrupted.
    """
    ids = (
        await session.exec(
            select(Complaint.id).where(Complaint.status == ComplaintStatus.pending).limit(limit)
        )
    ).all()
    for complaint_id in ids:
        await enqueue_complaint(redis, complaint_id)
    logger.info("workspace: enqueued %d complaints for classification", len(ids))
    return EnqueueResult(enqueued=len(ids), stream=settings.classification_queue)


@router.post("/enqueue/resolution", response_model=EnqueueResult)
@limiter.limit(_ENQUEUE_RATE_LIMIT)
async def enqueue_resolution_batch(
    request: Request,  # required by slowapi's per-route limiter for the key func
    limit: int = Query(default=50, ge=1, le=_BATCH_MAX),
    session: AsyncSession = Depends(get_session),
    redis: aioredis.Redis = Depends(get_redis),
    _: User = Depends(get_current_user),
) -> EnqueueResult:
    """Push up to `limit` escalated complaints to the resolution agent.

    Mirrors /resolutions/{id}/generate for a batch: flip escalated ->
    agent_triggered in this transaction so a re-run can't double-draft the same
    complaint, flush, then XADD each. If an enqueue raises, the request fails and
    the session rolls the flips back with it.
    """
    complaints = (
        await session.exec(
            select(Complaint).where(Complaint.status == ComplaintStatus.escalated).limit(limit)
        )
    ).all()
    for complaint in complaints:
        # Flip escalated -> agent_triggered (mirrors /generate's per-object flip)
        # so a re-run can't re-select and double-draft the same complaint.
        complaint.status = ComplaintStatus.agent_triggered
        complaint.updated_at = datetime.utcnow()
        session.add(complaint)
    await session.flush()
    for complaint in complaints:
        await enqueue_resolution(redis, complaint.id)
    logger.info("workspace: enqueued %d complaints for resolution", len(complaints))
    return EnqueueResult(enqueued=len(complaints), stream=settings.resolution_queue)
