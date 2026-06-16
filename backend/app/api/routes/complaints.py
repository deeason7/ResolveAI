"""
Complaint endpoints:

  GET    /complaints/             — paginated list with filters
  GET    /complaints/queue        — priority-sorted triage queue
  GET    /complaints/{id}         — fetch one
  GET    /complaints/{id}/similar — top-K nearest narratives (vector search)
  POST   /complaints/             — user submits a new complaint
  POST   /complaints/bulk-import  — admin-only CSV ingest (server-side path)

Listed and detail routes are authenticated (the dataset contains consumer
PII — even read access needs an account). Submitting needs a writer role
(analyst or admin) — a viewer is read-only. Bulk-import is admin-gated so even
an analyst can't trigger a 200K-row insert.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import get_current_user, get_redis, require_admin, require_writer
from app.database import get_session
from app.models.complaint import Complaint, ComplaintStatus
from app.models.user import User
from app.schemas.complaint import (
    BulkImportRequest,
    BulkImportResponse,
    ComplaintCreate,
    ComplaintFacets,
    ComplaintListResponse,
    ComplaintPublic,
    ComplaintQueueItem,
    ComplaintQueueResponse,
    SimilarComplaintItem,
    SimilarComplaintsResponse,
)
from app.services.data_ingestion import ingest_cfpb_csv
from app.services.embedder import embed_text
from app.services.vector_store import get_default_store
from app.workers.classification_worker import enqueue_complaint

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/complaints", tags=["complaints"])

# Sandbox bulk-import to mounted data dirs so an admin token can't
# accidentally (or intentionally) read /etc/passwd.
ALLOWED_IMPORT_ROOTS = (Path("/fine_tuning"), Path("/scripts"))


@router.get("/", response_model=ComplaintListResponse)
async def list_complaints(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
    status_: ComplaintStatus | None = Query(default=None, alias="status"),
    product: str | None = Query(default=None, max_length=255),
    company: str | None = Query(default=None, max_length=255),
    state: str | None = Query(default=None, min_length=2, max_length=2),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ComplaintListResponse:
    """Paginated complaint list with optional exact-match filters."""
    conditions = []
    if status_ is not None:
        conditions.append(Complaint.status == status_)
    if product is not None:
        conditions.append(Complaint.product == product)
    if company is not None:
        conditions.append(Complaint.company == company)
    if state is not None:
        conditions.append(Complaint.state == state.upper())

    count_stmt = select(func.count(Complaint.id))
    list_stmt = select(Complaint).order_by(Complaint.created_at.desc())
    for cond in conditions:
        count_stmt = count_stmt.where(cond)
        list_stmt = list_stmt.where(cond)

    total = (await session.exec(count_stmt)).one()
    rows = (await session.exec(list_stmt.offset(offset).limit(limit))).all()

    return ComplaintListResponse(
        items=[ComplaintPublic.model_validate(c) for c in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# Statuses a reviewer can act on. pending rows have no classification yet and
# resolved rows are done — neither belongs in a "needs attention" queue.
ACTIONABLE_STATUSES = (
    ComplaintStatus.classified,
    ComplaintStatus.escalated,
    ComplaintStatus.agent_triggered,
    ComplaintStatus.draft_ready,
    ComplaintStatus.needs_review,
)


# NOTE: declared before /{complaint_id} — route matching is declaration-order,
# and the path-param route would otherwise swallow "queue" as a (bad) UUID.
@router.get("/queue", response_model=ComplaintQueueResponse)
async def triage_queue(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
    status_: ComplaintStatus | None = Query(default=None, alias="status"),
    sentiment: str | None = Query(default=None, max_length=50),
    urgency_min: int | None = Query(default=None, ge=1, le=5),
    urgency_max: int | None = Query(default=None, ge=1, le=5),
    product: str | None = Query(default=None, max_length=255),
    company: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ComplaintQueueResponse:
    """Triage queue: highest-priority complaints first.

    Order is priority_score desc, then urgency desc, then oldest-first as the
    tiebreak (two equally urgent complaints shouldn't queue-jump by recency).
    NULLS LAST is explicit because Postgres sorts nulls first on desc and
    SQLite sorts them last — without it, prod would lead with unscored rows
    while tests happily passed.
    """
    if urgency_min is not None and urgency_max is not None and urgency_min > urgency_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="urgency_min cannot exceed urgency_max",
        )

    conditions = []
    if status_ is not None:
        conditions.append(Complaint.status == status_)
    else:
        conditions.append(Complaint.status.in_(ACTIONABLE_STATUSES))
    if sentiment is not None:
        conditions.append(Complaint.sentiment == sentiment)
    if urgency_min is not None:
        conditions.append(Complaint.urgency >= urgency_min)
    if urgency_max is not None:
        conditions.append(Complaint.urgency <= urgency_max)
    if product is not None:
        conditions.append(Complaint.product == product)
    if company is not None:
        conditions.append(Complaint.company == company)

    count_stmt = select(func.count(Complaint.id))
    queue_stmt = select(Complaint).order_by(
        Complaint.priority_score.desc().nulls_last(),
        Complaint.urgency.desc().nulls_last(),
        Complaint.created_at.asc(),
    )
    for cond in conditions:
        count_stmt = count_stmt.where(cond)
        queue_stmt = queue_stmt.where(cond)

    total = (await session.exec(count_stmt)).one()
    rows = (await session.exec(queue_stmt.offset(offset).limit(limit))).all()

    return ComplaintQueueResponse(
        items=[ComplaintQueueItem.from_complaint(c) for c in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/facets", response_model=ComplaintFacets)
async def complaint_facets(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> ComplaintFacets:
    """Distinct product and company values, for the filter dropdowns.

    Declared before /{complaint_id} so "facets" isn't parsed as a UUID — same
    reason /queue sits up here. SELECT DISTINCT over the corpus; both lists are
    static between reseeds, so the frontend caches them.
    """
    products = (
        await session.exec(
            select(Complaint.product)
            .where(Complaint.product.is_not(None))
            .distinct()
            .order_by(Complaint.product)
        )
    ).all()
    companies = (
        await session.exec(
            select(Complaint.company)
            .where(Complaint.company.is_not(None))
            .distinct()
            .order_by(Complaint.company)
        )
    ).all()
    return ComplaintFacets(products=list(products), companies=list(companies))


@router.get("/{complaint_id}", response_model=ComplaintPublic)
async def get_complaint(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> ComplaintPublic:
    """Fetch a single complaint by UUID."""
    complaint = await session.get(Complaint, complaint_id)
    if complaint is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    return ComplaintPublic.model_validate(complaint)


@router.get("/{complaint_id}/similar", response_model=SimilarComplaintsResponse)
async def similar_complaints(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
    product: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=5, ge=1, le=10),
) -> SimilarComplaintsResponse:
    """Top-K most similar complaints, by cosine similarity of the narratives.

    The complaint's own embedding lives in the collection, so it would come
    back as hit #1 with score ~1.0 — we over-fetch by one and drop it. Hits
    are then hydrated from Postgres (the payload is a thin search index;
    rows deleted since indexing are silently skipped). Embedding and search
    are sync CPU/network work, pushed off the event loop with to_thread —
    same pattern as the agent's search_precedents tool.
    """
    complaint = await session.get(Complaint, complaint_id)
    if complaint is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    if not complaint.narrative or not complaint.narrative.strip():
        return SimilarComplaintsResponse(items=[])

    filters = {"product": product} if product else None
    try:
        vector = await asyncio.to_thread(embed_text, complaint.narrative)
        store = get_default_store()
        hits = await asyncio.to_thread(store.search_similar, vector, filters, limit + 1)
    except Exception:
        # The detail page still works without this panel — degrade to 503
        # rather than a traceback; the log keeps the real cause.
        logger.exception("similarity search failed for %s", complaint_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Similarity search is temporarily unavailable",
        ) from None

    hits = [h for h in hits if h.complaint_id != str(complaint_id)][:limit]
    ids = [uuid.UUID(h.complaint_id) for h in hits]
    rows = (await session.exec(select(Complaint).where(Complaint.id.in_(ids)))).all()
    by_id = {c.id: c for c in rows}
    return SimilarComplaintsResponse(
        items=[
            SimilarComplaintItem.from_hit(by_id[i], h.score)
            for i, h in zip(ids, hits, strict=True)
            if i in by_id
        ]
    )


@router.post("/", response_model=ComplaintPublic, status_code=status.HTTP_201_CREATED)
async def submit_complaint(
    body: ComplaintCreate,
    session: AsyncSession = Depends(get_session),
    redis: aioredis.Redis = Depends(get_redis),
    _: User = Depends(require_writer),
) -> ComplaintPublic:
    """Submit a new complaint and queue it for classification.

    The enqueue is best-effort: status=pending is the durable signal, so a
    Redis hiccup must not lose the submission — a sweep can re-enqueue pending
    complaints later, exactly like the other recoverable side-channels.
    """
    complaint = Complaint(
        narrative=body.narrative,
        product=body.product,
        sub_product=body.sub_product,
        issue=body.issue,
        sub_issue=body.sub_issue,
        company=body.company,
        state=body.state.upper() if body.state else None,
    )
    session.add(complaint)
    await session.flush()
    await session.refresh(complaint)
    try:
        await enqueue_complaint(redis, complaint.id)
    except Exception:
        logger.exception("classification enqueue failed; %s stays pending", complaint.id)
    logger.info("Complaint submitted: %s", complaint.id)
    return ComplaintPublic.model_validate(complaint)


@router.post(
    "/bulk-import",
    response_model=BulkImportResponse,
    status_code=status.HTTP_200_OK,
)
async def bulk_import(
    body: BulkImportRequest,
    _: User = Depends(require_admin),
) -> BulkImportResponse:
    """Ingest a server-local normalized CFPB CSV. Idempotent."""
    csv_path = Path(body.path).resolve()
    if not any(
        str(csv_path).startswith(str(root.resolve()) + "/") for root in ALLOWED_IMPORT_ROOTS
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"path must be under one of {[str(r) for r in ALLOWED_IMPORT_ROOTS]}",
        )
    if not csv_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CSV not found")

    logger.info("bulk import requested: %s (batch_size=%d)", csv_path, body.batch_size)
    result = await ingest_cfpb_csv(csv_path, batch_size=body.batch_size)
    return BulkImportResponse(
        rows_read=result.rows_read,
        rows_inserted=result.rows_inserted,
        rows_skipped=result.rows_skipped,
        batches=result.batches,
        elapsed_seconds=result.elapsed_seconds,
    )
