"""
API contracts for the LLMOps observability endpoints.

Everything here is read-only telemetry shaped for charts: small grouped
aggregates, never raw log dumps. The one exception is the guardrail
violation log, which is intentionally row-level — reviewers need to read
individual violations, and the resolutions table is small by design (one
row per draft version).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel


class CostPoint(BaseModel):
    """Daily spend/volume for one provider."""

    day: date
    provider: str
    calls: int
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class CostsResponse(BaseModel):
    days: int
    total_cost_usd: float
    total_calls: int
    points: list[CostPoint]


class LatencyStat(BaseModel):
    """Latency profile of one operation type."""

    operation: str
    calls: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: int


class LatencyResponse(BaseModel):
    days: int
    items: list[LatencyStat]


class RoutingSlice(BaseModel):
    """Call volume per (provider, fallback-flag) pair.

    provider "none" is the deterministic fail-closed path (no model ran);
    was_fallback marks cloud calls that covered for an unavailable local
    model. They are different failure stories and must not be merged.
    """

    provider: str
    was_fallback: bool
    calls: int
    cost_usd: float


class RoutingResponse(BaseModel):
    days: int
    items: list[RoutingSlice]


class DriftPoint(BaseModel):
    day: date
    sentiment: str
    count: int


class DriftResponse(BaseModel):
    """Classifier output distribution over time, on the classification clock.

    Differs from /analytics/sentiment/trends, which buckets by complaint
    event date — this buckets by when the classify call actually ran, which
    is what drift monitoring cares about.
    """

    days: int
    points: list[DriftPoint]


class GuardrailViolationRow(BaseModel):
    """One violation, flattened from a resolution's violation list."""

    complaint_id: uuid.UUID
    resolution_id: uuid.UUID
    version: int
    guardrail_status: str
    layer: str
    code: str
    message: str
    created_at: datetime


class GuardrailLogResponse(BaseModel):
    items: list[GuardrailViolationRow]
    total_violations: int
