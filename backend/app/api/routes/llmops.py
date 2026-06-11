"""
LLMOps observability endpoints (read-only, authed):

  GET /llmops/costs       — daily cost/token spend per provider
  GET /llmops/latency     — latency percentiles per operation
  GET /llmops/routing     — local vs cloud vs fail-closed call routing
  GET /llmops/drift       — classifier output distribution on the classify clock
  GET /llmops/guardrails  — flattened guardrail violation log

All of these read the llm_logs table the workers write on every model call
(and resolutions for the violation log). Aggregation happens in SQL where
it's portable; percentiles and JSON flattening happen in Python because
SQLite (tests) has no percentile_cont and JSON-array filtering is dialect
pain — both datasets are bounded (one row per LLM call / per draft), so the
Python side stays cheap. Revisit if llm_logs ever outgrows that.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import get_current_user
from app.database import engine, get_session
from app.models.complaint import Complaint
from app.models.llm_log import LLMLog
from app.models.resolution import Resolution
from app.models.user import User
from app.schemas.llmops import (
    CostPoint,
    CostsResponse,
    DriftPoint,
    DriftResponse,
    GuardrailLogResponse,
    GuardrailViolationRow,
    LatencyResponse,
    LatencyStat,
    RoutingResponse,
    RoutingSlice,
)

router = APIRouter(prefix="/llmops", tags=["llmops"])


def _day_bucket(col):
    """Day-truncation for the active dialect as 'YYYY-MM-DD' text.

    Same dialect seam as analytics: SQLite has date(), Postgres has
    to_char(), both emit the same string.
    """
    if engine.dialect.name == "postgresql":
        return func.to_char(col, "YYYY-MM-DD")
    return func.date(col)


def _cutoff(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


@router.get("/costs", response_model=CostsResponse)
async def costs(
    days: int = Query(default=30, ge=1, le=3650),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> CostsResponse:
    """Daily call counts, token volume and spend, grouped per provider."""
    day = _day_bucket(LLMLog.created_at)
    stmt = (
        select(
            day.label("day"),
            LLMLog.provider,
            func.count(LLMLog.id),
            func.coalesce(func.sum(LLMLog.cost_usd), 0.0),
            func.coalesce(func.sum(LLMLog.prompt_tokens), 0),
            func.coalesce(func.sum(LLMLog.completion_tokens), 0),
        )
        .where(LLMLog.created_at >= _cutoff(days))
        .group_by("day", LLMLog.provider)
        .order_by("day")
    )
    rows = (await session.exec(stmt)).all()
    points = [
        CostPoint(
            day=d,
            provider=provider,
            calls=calls,
            cost_usd=round(cost, 6),
            prompt_tokens=ptok,
            completion_tokens=ctok,
        )
        for d, provider, calls, cost, ptok, ctok in rows
    ]
    return CostsResponse(
        days=days,
        total_cost_usd=round(sum(p.cost_usd for p in points), 6),
        total_calls=sum(p.calls for p in points),
        points=points,
    )


@router.get("/latency", response_model=LatencyResponse)
async def latency(
    days: int = Query(default=90, ge=1, le=3650),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> LatencyResponse:
    """p50/p95/avg/max latency per operation type."""
    stmt = select(LLMLog.operation, LLMLog.latency_ms).where(
        LLMLog.latency_ms.is_not(None), LLMLog.created_at >= _cutoff(days)
    )
    rows = (await session.exec(stmt)).all()

    by_op: dict[str, list[int]] = {}
    for operation, ms in rows:
        by_op.setdefault(operation, []).append(ms)

    items = []
    for operation, vals in sorted(by_op.items()):
        if len(vals) == 1:
            p50 = p95 = float(vals[0])
        else:
            p50 = statistics.median(vals)
            # quantiles(n=20) → cut points at 5% steps; index 18 is the 95th.
            p95 = statistics.quantiles(vals, n=20)[18]
        items.append(
            LatencyStat(
                operation=operation,
                calls=len(vals),
                avg_ms=round(statistics.fmean(vals), 1),
                p50_ms=round(p50, 1),
                p95_ms=round(p95, 1),
                max_ms=max(vals),
            )
        )
    return LatencyResponse(days=days, items=items)


@router.get("/routing", response_model=RoutingResponse)
async def routing(
    days: int = Query(default=90, ge=1, le=3650),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> RoutingResponse:
    """Call volume per (provider, was_fallback) — the local/cloud split."""
    stmt = (
        select(
            LLMLog.provider,
            LLMLog.was_fallback,
            func.count(LLMLog.id),
            func.coalesce(func.sum(LLMLog.cost_usd), 0.0),
        )
        .where(LLMLog.created_at >= _cutoff(days))
        .group_by(LLMLog.provider, LLMLog.was_fallback)
        .order_by(LLMLog.provider)
    )
    rows = (await session.exec(stmt)).all()
    return RoutingResponse(
        days=days,
        items=[
            RoutingSlice(
                provider=provider, was_fallback=fallback, calls=calls, cost_usd=round(cost, 6)
            )
            for provider, fallback, calls, cost in rows
        ],
    )


@router.get("/drift", response_model=DriftResponse)
async def drift(
    days: int = Query(default=90, ge=1, le=3650),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> DriftResponse:
    """Sentiment distribution of classify calls, bucketed by call date.

    Joins classify-operation logs to the complaint's current label: a
    re-classified complaint counts once per classify event, under its
    current sentiment (per-call outputs aren't stored). Event-level honest,
    label-level approximate — good enough to watch the mix move.
    """
    day = _day_bucket(LLMLog.created_at)
    stmt = (
        select(day.label("day"), Complaint.sentiment, func.count(LLMLog.id))
        .join(Complaint, Complaint.id == LLMLog.complaint_id)
        .where(
            LLMLog.operation == "classify",
            LLMLog.created_at >= _cutoff(days),
            Complaint.sentiment.is_not(None),
        )
        .group_by("day", Complaint.sentiment)
        .order_by("day")
    )
    rows = (await session.exec(stmt)).all()
    return DriftResponse(
        days=days,
        points=[DriftPoint(day=d, sentiment=s, count=c) for d, s, c in rows],
    )


@router.get("/guardrails", response_model=GuardrailLogResponse)
async def guardrail_log(
    layer: str | None = Query(default=None, max_length=50),
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> GuardrailLogResponse:
    """Violation log: one row per violation, newest resolutions first.

    Flattened in Python — filtering inside a JSON array column is dialect
    pain, and the resolutions table is one row per draft version, so the
    scan is small by construction.
    """
    stmt = (
        select(Resolution)
        .where(Resolution.guardrail_violations.is_not(None))
        .order_by(Resolution.created_at.desc())
    )
    resolutions = (await session.exec(stmt)).all()

    items: list[GuardrailViolationRow] = []
    total = 0
    for res in resolutions:
        for violation in res.guardrail_violations or []:
            if layer is not None and violation.get("layer") != layer:
                continue
            total += 1
            if len(items) < limit:
                items.append(
                    GuardrailViolationRow(
                        complaint_id=res.complaint_id,
                        resolution_id=res.id,
                        version=res.version,
                        guardrail_status=res.guardrail_status,
                        layer=violation.get("layer", "unknown"),
                        code=violation.get("code", "unknown"),
                        message=violation.get("message", ""),
                        created_at=res.created_at,
                    )
                )
    return GuardrailLogResponse(items=items, total_violations=total)
