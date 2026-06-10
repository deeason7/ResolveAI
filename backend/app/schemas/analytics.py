"""
Pydantic response schemas for the analytics API.

These are aggregate shapes, not row shapes — every payload here is the
output of a GROUP BY, so there's deliberately no from_attributes ORM
plumbing. Counts are split by sentiment/urgency with an explicit
"unclassified" bucket: most of the corpus predates the classifier, and
hiding that would make every chart lie about coverage.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

UNCLASSIFIED = "unclassified"


class SentimentTrendPoint(BaseModel):
    """Complaint count for one (day, sentiment) bucket."""

    day: date
    sentiment: str = Field(description=f"Model sentiment label, or '{UNCLASSIFIED}'")
    count: int


class SentimentTrendsResponse(BaseModel):
    """Daily sentiment counts over a trailing window.

    Buckets are always per-day; the client rolls them up into weeks/months
    as needed (pandas does that in one line, and it keeps the SQL portable).
    """

    days: int
    total: int
    points: list[SentimentTrendPoint]


class ProductUrgencyRow(BaseModel):
    """Complaint volume for one product, split by urgency level."""

    product: str
    total: int
    urgency_counts: dict[int, int] = Field(
        description="Counts keyed by urgency 1-5 (only levels present appear)"
    )
    unclassified: int = Field(description="Rows with no urgency assigned yet")


class ProductsBreakdownResponse(BaseModel):
    """Volume by product — feeds the urgency heatmap and the treemap."""

    items: list[ProductUrgencyRow]


class CompanyRiskRow(BaseModel):
    """Volume-and-severity scorecard for one company.

    SQL-derived only: the graph's risk_score (volume + adverse-response
    blend) stays on /graph/company/{name}; joining it across the top-N
    companies is a Phase 5 Day 27-28 item for the Analytics page.
    """

    company: str
    total_complaints: int
    avg_urgency: float | None = Field(description="Mean urgency over classified rows, else null")
    urgent_count: int = Field(description="Rows with urgency >= 4")
    extreme_negative_count: int


class CompaniesRiskResponse(BaseModel):
    """Top companies by complaint volume with severity columns."""

    items: list[CompanyRiskRow]
    limit: int
