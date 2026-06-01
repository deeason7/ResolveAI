"""
Response schemas for the knowledge-graph API.

These are the wire shapes returned by graph_store queries and the /graph
routes. They're deliberately flat and display-oriented — a frontend graph
view or the Phase 4 agent consumes them directly, so we shape them for the
consumer, not for Neo4j's internal record format.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Regulation(BaseModel):
    """A federal regulation node, as returned to callers."""

    id: str
    title: str
    cfr_reference: str
    summary: str
    key_provisions: list[str] = Field(default_factory=list)


class ProductBreakdown(BaseModel):
    """One row of a company's complaint distribution across products."""

    product: str
    count: int


class CompanyProfile(BaseModel):
    """Aggregated risk view of a single company."""

    name: str
    total_complaints: int = 0
    risk_score: float | None = None
    violations: list[str] = Field(
        default_factory=list,
        description="Titles of regulations this company has been linked to violating.",
    )
    product_breakdown: list[ProductBreakdown] = Field(default_factory=list)


class ResolutionPattern(BaseModel):
    """A resolution outcome type, optionally scoped to one company's usage."""

    pattern_type: str
    description: str
    success_rate: float | None = None
    company_usage_count: int | None = Field(
        default=None,
        description="How many times the queried company used this pattern (None if no company scope).",
    )
