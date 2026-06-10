"""
The resolution agent's four tools.

Each tool is a standalone ``async`` function that wraps one capability the
system already has, and returns a validated schema from ``schemas.agent``:

    1. search_precedents  -> Qdrant similarity search   (vector_store + embedder)
    2. lookup_regulations -> Neo4j product->regulation   (graph_store)
    3. check_company_history -> Neo4j company aggregate   (graph_store)
    4. draft_response     -> structured LLM generation    (llm_client)

Dependencies are injected as keyword args that default to the process-wide
singletons, so production calls them with no arguments while tests pass fakes.

Concurrency note: the graph tools are genuinely async (Neo4j async driver), so
they ``await`` directly. The embedder, the Qdrant client, and the LLM client are
all *synchronous* and I/O- or CPU-bound, so they are trampolined off the event
loop with ``asyncio.to_thread`` — the same pattern the classification worker
uses. Blocking the single event loop on a 1-second embed or a multi-second LLM
call would stall every other coroutine in the process.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.schemas.agent import (
    CompanyHistoryInput,
    CompanyHistoryResult,
    DraftedResponse,
    DraftResponseInput,
    LookupRegulationsInput,
    PrecedentResult,
    RegulationResult,
    SearchPrecedentsInput,
)
from app.services.agent.prompts import (
    SYSTEM_PROMPT,
    build_draft_prompt,
    build_regeneration_prompt,
)
from app.services.embedder import embed_text
from app.services.graph_store import GraphStore, get_default_graph_store
from app.services.llm_client import LLMClient, LLMResponse, get_llm_client
from app.services.vector_store import VectorStore, get_default_store

logger = logging.getLogger(__name__)

# Three or more distinct linked violations reads as a pattern, not a one-off.
REPEAT_OFFENDER_MIN_VIOLATIONS = 3

# --- Tool 1: search_precedents ---


async def search_precedents(
    inp: SearchPrecedentsInput,
    *,
    vector_store: VectorStore | None = None,
) -> list[PrecedentResult]:
    """Return up to ``inp.limit`` complaints most similar to ``inp.complaint_text``.

    Embeds the narrative, runs a cosine search (optionally product-filtered), and
    reshapes each hit's payload into a :class:`PrecedentResult`. Payload fields are
    read with ``.get`` because two payload vintages coexist in the collection (see
    :class:`PrecedentResult`); a missing key degrades to a blank/None, never a KeyError.
    """
    store = vector_store or get_default_store()
    if not inp.complaint_text or not inp.complaint_text.strip():
        return []  # embed_text raises on empty; nothing to search for anyway

    vector = await asyncio.to_thread(embed_text, inp.complaint_text)
    filters = {"product": inp.product} if inp.product else None
    hits = await asyncio.to_thread(store.search_similar, vector, filters, inp.limit)

    return [
        PrecedentResult(
            complaint_id=h.complaint_id,
            narrative_preview=h.payload.get("narrative_preview", ""),
            sentiment=h.payload.get("sentiment"),
            company_response=h.payload.get("company_response"),
            similarity_score=h.score,
        )
        for h in hits
    ]


# --- Tool 2: lookup_regulations ---


def _regulation_relevance(product: str, issue: str | None) -> str:
    """A short, honest 'why this applies' string for the drafting prompt."""
    if issue:
        return f"Governs {issue!r} issues for {product!r}."
    return f"Applies to {product!r} complaints."


async def lookup_regulations(
    inp: LookupRegulationsInput,
    *,
    graph_store: GraphStore | None = None,
) -> list[RegulationResult]:
    """Return the regulations governing the complaint's product (and issue, if given)."""
    store = graph_store or get_default_graph_store()
    regs = await store.get_regulations(inp.product, inp.issue)
    relevance = _regulation_relevance(inp.product, inp.issue)
    return [
        RegulationResult(
            title=r.title,
            cfr_reference=r.cfr_reference,
            summary=r.summary,
            key_provisions=r.key_provisions,
            relevance=relevance,
        )
        for r in regs
    ]


# --- Tool 3: check_company_history ---


async def check_company_history(
    inp: CompanyHistoryInput,
    *,
    graph_store: GraphStore | None = None,
) -> CompanyHistoryResult | None:
    """Return the company's risk profile, or None if it isn't in the graph.

    Maps the graph's :class:`CompanyProfile` onto the agent-facing shape: the
    product breakdown (already sorted by volume desc in the Cypher query) becomes
    ``top_products``, and ``repeat_offender`` is derived from the violation count.
    """
    store = graph_store or get_default_graph_store()
    profile = await store.get_company_profile(inp.company_name)
    if profile is None:
        return None
    return CompanyHistoryResult(
        company_name=profile.name,
        total_complaints=profile.total_complaints,
        risk_score=profile.risk_score,
        violations=profile.violations,
        top_products=[pb.product for pb in profile.product_breakdown[:5]],
        repeat_offender=len(profile.violations) >= REPEAT_OFFENDER_MIN_VIOLATIONS,
    )


# --- Tool 4: draft_response ---


@dataclass
class DraftOutcome:
    """A drafted response plus the call metadata the LLMOps tracker persists.

    Mirrors :class:`ClassificationOutcome`: the tool returns *what the model said*
    and *what the call cost*, and leaves the DB write to the caller (the
    resolution worker) so logging stays atomic with the domain write.
    """

    drafted: DraftedResponse
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    is_fallback: bool


async def draft_response(
    inp: DraftResponseInput,
    *,
    llm_client: LLMClient | None = None,
    feedback: str | None = None,
    previous_draft: str | None = None,
) -> DraftOutcome:
    """Draft a resolution from the gathered context via a structured LLM call.

    Unlike classification there is no deterministic fallback — a resolution can't
    be meaningfully faked — so a total provider outage propagates as
    :class:`LLMUnavailableError` for the orchestrator to turn into a
    flagged-for-human outcome.

    On a regeneration pass the caller supplies ``previous_draft`` + guardrail
    ``feedback``; both are appended as a follow-up user turn so the model revises
    with the original context still in view.
    """
    client = llm_client or get_llm_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_draft_prompt(inp)},
    ]
    if feedback and previous_draft is not None:
        messages.append(
            {"role": "user", "content": build_regeneration_prompt(previous_draft, feedback)}
        )
    # client.structured is synchronous (OpenAI-compatible sync SDK); keep it off the loop.
    resp: LLMResponse[DraftedResponse] = await asyncio.to_thread(
        client.structured, DraftedResponse, messages
    )
    return DraftOutcome(
        drafted=resp.data,
        provider=resp.provider.value,
        model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
        is_fallback=resp.is_fallback,
    )
