"""
Complaint endpoints:

  GET    /complaints/             — paginated list with filters
  GET    /complaints/{id}         — fetch one
  POST   /complaints/             — user submits a new complaint
  POST   /complaints/bulk-import  — admin-only CSV ingest (server-side path)

Listed and detail routes are authenticated (the dataset contains consumer
PII — even read access needs an account). Bulk-import is admin-gated so a
viewer/analyst can't trigger a 200K-row insert.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import get_current_user, require_admin
from app.database import get_session
from app.models.complaint import Complaint, ComplaintStatus
from app.models.user import User
from app.schemas.complaint import (
    BulkImportRequest,
    BulkImportResponse,
    ComplaintCreate,
    ComplaintListResponse,
    ComplaintPublic,
)
from app.services.data_ingestion import ingest_cfpb_csv

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


@router.post("/", response_model=ComplaintPublic, status_code=status.HTTP_201_CREATED)
async def submit_complaint(
    body: ComplaintCreate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> ComplaintPublic:
    """Submit a new complaint. Goes into status=pending until the classifier picks it up."""
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
