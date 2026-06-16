"""
Pydantic request/response schemas for the complaints API.

ORM (app.models.complaint.Complaint) defines the row shape; these define
the wire shape. Keeping them separate lets the table evolve without
breaking the contract, and prevents accidentally exposing internal
fields like priority_score before it's been calibrated.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.complaint import Complaint, ComplaintStatus


class ComplaintCreate(BaseModel):
    """Body for POST /complaints/ — user-submitted complaint."""

    narrative: str = Field(min_length=10, max_length=20_000)
    product: str | None = Field(default=None, max_length=255)
    sub_product: str | None = Field(default=None, max_length=255)
    issue: str | None = Field(default=None, max_length=255)
    sub_issue: str | None = Field(default=None, max_length=255)
    company: str | None = Field(default=None, max_length=255)
    state: str | None = Field(default=None, min_length=2, max_length=2)


class ComplaintPublic(BaseModel):
    """Full complaint as returned by GET /complaints/{id} and POST /complaints/."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    cfpb_complaint_id: str | None
    narrative: str
    product: str | None
    sub_product: str | None
    issue: str | None
    sub_issue: str | None
    company: str | None
    company_response: str | None
    state: str | None
    date_received: datetime | None
    status: ComplaintStatus
    sentiment: str | None
    intent: str | None
    urgency: int | None
    priority_score: float | None
    created_at: datetime
    updated_at: datetime


class ComplaintListResponse(BaseModel):
    """Paginated list of complaints."""

    items: list[ComplaintPublic]
    total: int
    limit: int
    offset: int


QUEUE_PREVIEW_CHARS = 200


class ComplaintQueueItem(BaseModel):
    """Lean triage-queue row.

    The queue is a table view — shipping each row's full narrative (up to
    20K chars) just to render 50 table rows wastes most of the payload, so
    this carries a preview and the detail page fetches the rest by id.
    """

    id: uuid.UUID
    company: str | None
    product: str | None
    issue: str | None
    state: str | None
    status: ComplaintStatus
    sentiment: str | None
    intent: str | None
    urgency: int | None
    priority_score: float | None
    narrative_preview: str
    date_received: datetime | None
    created_at: datetime

    @classmethod
    def from_complaint(cls, complaint: Complaint) -> ComplaintQueueItem:
        return cls(
            id=complaint.id,
            company=complaint.company,
            product=complaint.product,
            issue=complaint.issue,
            state=complaint.state,
            status=complaint.status,
            sentiment=complaint.sentiment,
            intent=complaint.intent,
            urgency=complaint.urgency,
            priority_score=complaint.priority_score,
            narrative_preview=complaint.narrative[:QUEUE_PREVIEW_CHARS],
            date_received=complaint.date_received,
            created_at=complaint.created_at,
        )


class ComplaintQueueResponse(BaseModel):
    """Priority-ordered triage queue page."""

    items: list[ComplaintQueueItem]
    total: int
    limit: int
    offset: int


class SimilarComplaintItem(ComplaintQueueItem):
    """One vector-search hit for GET /complaints/{id}/similar.

    Same lean row shape as the queue, plus the cosine score and the
    historical company_response — the response is what makes a similar
    complaint useful as a precedent. Fields come from Postgres, not the
    Qdrant payload: the payload is a search index with two coexisting
    vintages, the database is the source of truth.
    """

    similarity_score: float
    company_response: str | None

    @classmethod
    def from_hit(cls, complaint: Complaint, score: float) -> SimilarComplaintItem:
        base = ComplaintQueueItem.from_complaint(complaint)
        return cls(
            **base.model_dump(),
            similarity_score=score,
            company_response=complaint.company_response,
        )


class SimilarComplaintsResponse(BaseModel):
    """Top-K similarity hits. No total/offset — K-nearest search doesn't paginate."""

    items: list[SimilarComplaintItem]


class BulkImportRequest(BaseModel):
    """Body for POST /complaints/bulk-import (admin only)."""

    path: str = Field(description="Server-local CSV path (must be under /fine_tuning/data/)")
    batch_size: int = Field(default=10_000, ge=100, le=50_000)


class BulkImportResponse(BaseModel):
    """Result counts from a bulk-import run."""

    rows_read: int
    rows_inserted: int
    rows_skipped: int
    batches: int
    elapsed_seconds: float


class ComplaintFacets(BaseModel):
    """Distinct filter values for the triage/list dropdowns.

    Exact-match filters are unusable when you can't recall the precise string
    ("EQUIFAX, INC." vs "Equifax"), so the frontend turns these into selectable
    lists instead of free text. Both are static between reseeds — the frontend
    caches the response.
    """

    products: list[str]
    companies: list[str]
