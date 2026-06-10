"""
Pydantic I/O contracts for the resolution agent's four tools.

Every tool gets an explicit input and output model so the orchestrator (and the
tests) pass validated, structured data between steps instead of loose dicts —
the same separation-of-schemas discipline used for classification.

Two output shapes deliberately diverge from the spec's idealized versions to
match what the Phase 3 knowledge graph actually stores. Those deviations are
documented on the models so the divergence is intentional, not accidental drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.classification import ComplaintClassification

# --- Tool 1: search_precedents (Qdrant vector search) ---


class SearchPrecedentsInput(BaseModel):
    """Find similar past complaints for grounding the draft."""

    complaint_text: str = Field(description="Complaint narrative to find similar past cases for")
    product: str | None = Field(
        default=None, description="Optional product filter applied to the vector search"
    )
    limit: int = Field(default=5, ge=1, le=10, description="Maximum precedents to return")


class PrecedentResult(BaseModel):
    """One similar past complaint, assembled from a Qdrant point payload.

    ``narrative_preview`` and ``company_response`` come from the payload the
    Phase 1 backfill wrote for the 200K seed complaints. Points written later by
    the live classification worker carry a *different* payload (intent/urgency,
    no preview), so these fields tolerate absence rather than being required —
    see the payload note in ``tools.search_precedents``.
    """

    complaint_id: str
    narrative_preview: str = ""
    sentiment: str | None = None
    company_response: str | None = Field(
        default=None, description="How the company resolved it — the precedent's outcome"
    )
    similarity_score: float = Field(description="Cosine similarity in [-1, 1]; higher is closer")


# --- Tool 2: lookup_regulations (Neo4j graph traversal) ---


class LookupRegulationsInput(BaseModel):
    """Resolve the regulations governing a product (optionally narrowed by issue)."""

    product: str = Field(description="Financial product category from the complaint")
    issue: str | None = Field(
        default=None,
        description="Optional issue type. Present -> Product->Issue->Regulation; "
        "absent -> product-level regulations.",
    )


class RegulationResult(BaseModel):
    """A federal regulation surfaced for the complaint, with a relevance note."""

    title: str
    cfr_reference: str
    summary: str
    key_provisions: list[str] = Field(default_factory=list)
    relevance: str = Field(description="Why this regulation applies to the complaint's context")


# --- Tool 3: check_company_history (Neo4j aggregation) ---


class CompanyHistoryInput(BaseModel):
    """Pull the risk profile for the company named in the complaint."""

    company_name: str = Field(description="Company named in the complaint")


class CompanyHistoryResult(BaseModel):
    """Company risk view, reconciled to what Phase 3's graph actually computes.

    This differs from the spec's idealized ``CompanyProfile`` on purpose:
      * ``risk_score`` is **0-1** (0.5*log10-volume + 0.5*adverse-rate), not 0-10.
      * No ``avg_response_time_days`` (our CFPB sample has no response-time
        column) and no ``resolution_rate`` (no ground-truth outcome) — both were
        dropped in Phase 3.
      * ``top_products`` replaces the spec's ``top_issues``: the graph aggregates
        a company's complaint volume by product, not by issue.
      * ``repeat_offender`` is derived here from the violation count, since the
        graph has no per-year violation timestamps yet.
    """

    company_name: str
    total_complaints: int = 0
    risk_score: float | None = Field(default=None, description="0-1; higher means riskier")
    violations: list[str] = Field(
        default_factory=list, description="Titles of regulations the company is linked to violating"
    )
    top_products: list[str] = Field(
        default_factory=list,
        description="Products this company is most complained about, by volume",
    )
    repeat_offender: bool = Field(
        default=False, description="Derived: more than two distinct linked violations"
    )


# --- Tool 4: draft_response (LLM structured generation) ---

Tone = Literal["empathetic", "firm", "escalatory"]


class DraftResponseInput(BaseModel):
    """Everything the drafting LLM needs, gathered by tools 1-3."""

    complaint_narrative: str
    classification: ComplaintClassification
    precedents: list[PrecedentResult] = Field(default_factory=list)
    regulations: list[RegulationResult] = Field(default_factory=list)
    company_profile: CompanyHistoryResult | None = None


class DraftedResponse(BaseModel):
    """Structured resolution draft — instructor-validated LLM output.

    Length is intentionally *not* constrained here (only ``min_length=1``):
    the guardrail engine (Phase 4 Day 21-22) owns the 200-3000 char policy, so
    pinning it twice would have instructor and the guardrails fighting over the
    same rule. ``cited_regulations`` is likewise unverified at this layer — the
    regulatory-accuracy guardrail cross-checks it against the graph.
    """

    response_text: str = Field(min_length=1, description="The drafted resolution response")
    tone: Tone = Field(description="Overall tone the response strikes")
    cited_regulations: list[str] = Field(
        default_factory=list, description="Regulation titles referenced in the response"
    )
    recommended_actions: list[str] = Field(
        default_factory=list, description="Concrete next steps for the consumer"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's self-rated confidence in [0, 1]")
