"""
Analytics endpoints — SQL aggregates over the complaints table:

  GET /analytics/sentiment/trends     — daily sentiment counts (trailing window)
  GET /analytics/products/breakdown   — volume by product, split by urgency
  GET /analytics/companies/risk       — top companies by volume + severity columns

All three are authenticated reads over the same PII-bearing dataset as
/complaints. They aggregate in the database (GROUP BY), never by paging
rows through the API — 200K rows stay in Postgres.

Time axis: COALESCE(date_received, created_at). Bulk-imported CFPB rows
carry their real historical date_received but were all created_at'd on
ingest day; live submissions are the opposite. Coalescing gives every row
its truthful event date.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import get_current_user
from app.database import engine, get_session
from app.models.complaint import Complaint
from app.models.user import User
from app.schemas.analytics import (
    UNCLASSIFIED,
    CompaniesRiskResponse,
    CompanyRiskRow,
    ProductsBreakdownResponse,
    ProductUrgencyRow,
    SentimentTrendPoint,
    SentimentTrendsResponse,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])

URGENT_THRESHOLD = 4  # urgency >= 4 is the escalation line (same as the worker)

_event_date = func.coalesce(Complaint.date_received, Complaint.created_at)


def _day_bucket():
    """Day-truncation expression for the active dialect, as 'YYYY-MM-DD' text.

    The one place the ORM can't paper over the dialect gap: SQLite (tests)
    has date(), Postgres has to_char(). Both yield the same string, so the
    response schema parses it into a date either way.
    """
    if engine.dialect.name == "postgresql":
        return func.to_char(_event_date, "YYYY-MM-DD")
    return func.date(_event_date)


@router.get("/sentiment/trends", response_model=SentimentTrendsResponse)
async def sentiment_trends(
    days: int = Query(default=30, ge=1, le=3650),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> SentimentTrendsResponse:
    """Daily complaint counts per sentiment over the trailing `days` window.

    Rows the classifier hasn't reached yet come back as sentiment
    "unclassified" rather than being dropped — the dashboard's donut and
    trend charts should show coverage honestly, not just the labeled slice.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    day = _day_bucket()
    sentiment = func.coalesce(Complaint.sentiment, UNCLASSIFIED)

    stmt = (
        select(day.label("day"), sentiment.label("sentiment"), func.count(Complaint.id))
        .where(_event_date >= cutoff)
        .group_by(day, sentiment)
        .order_by(day)
    )
    rows = (await session.exec(stmt)).all()

    points = [SentimentTrendPoint(day=d, sentiment=s, count=n) for d, s, n in rows]
    return SentimentTrendsResponse(days=days, total=sum(p.count for p in points), points=points)


@router.get("/products/breakdown", response_model=ProductsBreakdownResponse)
async def products_breakdown(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> ProductsBreakdownResponse:
    """Complaint volume per product, split by urgency level.

    One GROUP BY (product, urgency) round-trip, pivoted in Python: at most
    products x 6 rows come back, and pivoting here beats a CASE-WHEN column
    per urgency level for readability. Feeds the urgency-by-product heatmap
    today and the treemap on the Analytics page later.
    """
    product = func.coalesce(Complaint.product, "Unspecified")
    stmt = select(product.label("product"), Complaint.urgency, func.count(Complaint.id)).group_by(
        product, Complaint.urgency
    )
    rows = (await session.exec(stmt)).all()

    by_product: dict[str, ProductUrgencyRow] = {}
    for name, urgency, count in rows:
        row = by_product.setdefault(
            name, ProductUrgencyRow(product=name, total=0, urgency_counts={}, unclassified=0)
        )
        row.total += count
        if urgency is None:
            row.unclassified += count
        else:
            row.urgency_counts[urgency] = row.urgency_counts.get(urgency, 0) + count

    items = sorted(by_product.values(), key=lambda r: r.total, reverse=True)
    return ProductsBreakdownResponse(items=items)


@router.get("/companies/risk", response_model=CompaniesRiskResponse)
async def companies_risk(
    limit: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> CompaniesRiskResponse:
    """Top `limit` companies by complaint volume, with severity columns.

    Severity here is what SQL alone can answer (avg urgency, urgent and
    extreme-negative counts). The graph's blended risk_score stays on
    /graph/company/{name}; rows without a company are excluded rather than
    lumped into a fake "Unknown" company that would dominate the chart.
    """
    urgent = func.sum(case((Complaint.urgency >= URGENT_THRESHOLD, 1), else_=0))
    extreme = func.sum(case((Complaint.sentiment == "extreme_negative", 1), else_=0))

    stmt = (
        select(
            Complaint.company,
            func.count(Complaint.id).label("total"),
            func.avg(Complaint.urgency),
            urgent,
            extreme,
        )
        .where(Complaint.company.is_not(None))
        .group_by(Complaint.company)
        .order_by(func.count(Complaint.id).desc())
        .limit(limit)
    )
    rows = (await session.exec(stmt)).all()

    items = [
        CompanyRiskRow(
            company=company,
            total_complaints=total,
            avg_urgency=round(avg, 2) if avg is not None else None,
            urgent_count=urgent_n,
            extreme_negative_count=extreme_n,
        )
        for company, total, avg, urgent_n, extreme_n in rows
    ]
    return CompaniesRiskResponse(items=items, limit=limit)
