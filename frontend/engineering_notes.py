"""
Per-page "under the hood" notes for the sidebar.

The pages demo the product; this panel demos the engineering. Each page
gets the architectural decisions behind it — what was chosen, what it beat,
and why — so a visitor or an interviewer sees the reasoning, not just the
pixels. Collapsed by default everywhere: business value leads, and the
tour's final step points the curious here.
"""

from __future__ import annotations

import streamlit as st

_STACK_FOOTER = (
    "Auth: short-lived JWT + httpOnly rotating refresh cookie, silent re-auth "
    "on 401; every page talks through one API-client seam that memoizes read-only "
    "aggregates (cost/latency, graph, trends) behind a short TTL while live pipeline "
    "views stay uncached. Stack: FastAPI · PostgreSQL · Redis Streams · Qdrant · "
    "Neo4j · fine-tuned SLM + cloud LLM fallback · Streamlit — fully Dockerized."
)

_NOTES = {
    "Dashboard": """
**One fetch, many views.** The cards, deltas and donut all slice one 365-day
daily-bucket payload client-side — the alternative was four near-identical
aggregate calls on every interaction.

**Aggregate in the database.** Charts read `GROUP BY` endpoints returning
O(groups) rows; nobody pages 200K rows through an API to count them.

**Data-clock, not wall-clock.** Windows anchor to the newest complaint on
record (override under the cards) — a "today" window over a historical
corpus would always read zero.

**Mixed-provenance time axis.** Bulk-imported rows carry real historical
dates but ingest-day timestamps; live submissions the reverse. The axis is
`COALESCE(date_received, created_at)`.

**Honest coverage.** Unclassified volume is a visible bucket, never a silent
drop — a dashboard that hides unlabeled rows is lying about its classifier.
""",
    "Triage Queue": """
**Server-side everything.** Ordering, filtering and pagination run in SQL
over the full corpus; this page is a thin view.

**Deterministic ordering.** Priority desc, urgency desc, then *oldest-first*
— equal-urgency complaints can't queue-jump by recency. `NULLS LAST` is
explicit because Postgres and SQLite disagree on null placement under
`DESC`: tests stay green while prod would lead with unscored rows.

**Route-matching discipline.** The static `/queue` route is declared before
`/{id}` — matching is declaration-order, and "queue" parses as a bad UUID
otherwise.

**Lean wire shape.** Rows ship 200-char previews, not 20K-char narratives;
the detail page fetches the rest by id.

**Filters that respect NULLs.** An untouched urgency slider sends no filter
at all, so not-yet-scored complaints stay visible.
""",
    "Complaint Detail": """
**Precedent search.** Similar complaints = cosine search over 384-dim
sentence-transformer embeddings of all 200K narratives (~140 ms warm). The
query's own vector lives in the collection, so the API over-fetches K+1 and
drops it.

**Index vs truth.** The vector store returns ids and scores; rows are
hydrated from Postgres. Search payloads drift, the database doesn't — and
hits whose rows were deleted simply drop out.

**Off the event loop.** Embedding and vector search are synchronous CPU/HTTP
work, pushed through `asyncio.to_thread` so one slow call can't freeze every
other request.

**Blast-radius failure policy.** Vector store down → this one panel degrades
to a clear 503 message; the complaint still loads. The classification
worker fails *closed* instead (max severity, human escalation) — same
system, different stakes, different policy.

**Three guardrail verdicts.** Violated, passed, and no-verdict render
differently: an escalated run's unevaluated layers show ⚠, because "never
checked" must not display as "passed".

**Honest controls.** The draft box is editable for copy-out, but approval
applies the stored draft — changing it means reject-with-feedback, which is
fed into the agent's regeneration prompt.
""",
    "Analytics": """
**Portable SQL, client-side shaping.** The trends API stays day-granular —
one SQL shape that runs identically on Postgres and SQLite — and the weekly
roll-up is a single pandas `Grouper(freq="W")`.

**Two stores, one table.** The scorecard joins Postgres severity aggregates
with per-company risk scores and regulation links from the Neo4j knowledge
graph; each column's source is named in the caption.

**Graceful partial failure.** A company missing from the graph blanks its
risk columns and raises a counted warning — it doesn't kill the page.

**Earned numbers only.** No resolution-rate column yet: against 200K
complaints the current resolution count would be statistical noise, and the
caption says exactly that instead of charting it.
""",
    "Graph Explorer": """
**Bounded traversals.** The backend caps depth at 3 — every hop fans out
across a dense company/product/issue web, and an unbounded walk would pull
most of the graph into one response.

**APOC subgraph in one query.** `apoc.path.subgraphAll` returns the whole
neighborhood in a single round-trip, shaped as loose node/edge dicts so the
frontend renders generically and the wire contract doesn't chase every
label's property set.

**Framework-decoupled rendering.** pyvis generates a self-contained vis.js
HTML document (`cdn_resources="in_line"`) embedded as a component — no
coupling to Streamlit's custom-component API, so Streamlit upgrades can't
break the canvas. The tradeoff: clicks stay in the browser, so re-centering
goes through the inspect panel instead of the node itself.

**One-shot recenter.** "Explore from here" hands the node off via a popped
session key applied before the search widget instantiates — the same
handoff pattern the triage→detail flow uses.

**Live cross-checks.** A company node's panel numbers come from the same
graph the agent queries when drafting — what you see is what the agent saw.
""",
    "LLMOps Observatory": """
**Metered at the source.** Workers write one llm_logs row per model call —
provider, model, tokens, cost, latency, fallback flag — inside the same
transaction as the work itself, so telemetry can't drift from reality.

**Aggregate where portable, compute where not.** Daily groupings are SQL;
percentiles are Python because SQLite (tests) lacks percentile_cont, and
the table is bounded at one row per call.

**Two failure stories, never merged.** Provider `none` is the deterministic
fail-closed path (nothing answered → flag maximum severity for humans);
`was_fallback` marks cloud covering for local. A routing chart that mixed
them would hide exactly the distinction that matters.

**Drift on the classify clock.** The distribution chart buckets by when
classify calls ran, not the complaint's event date — that's the clock drift
monitoring actually cares about.

**Violation log is row-level on purpose.** Reviewers read individual
guardrail catches; the resolutions table is one row per draft version, so
flattening its JSON violations in Python stays cheap by construction.
""",
    "Workspace": """
**Durable truth, transient overlay.** Stage counts read `complaint.status` —
the signal the workers write in the *same transaction* as the work — so the
board is always exact. Redis stream state (waiting / in-flight / workers) is
layered on top, best-effort: the moving picture over the settled one.

**One producer, many callers.** Enqueueing reuses the exact XADD functions the
submit and per-complaint generate routes already use — a single definition of
each stream's message shape, never a second copy.

**At-least-once by design.** A complaint stays `pending` until a worker
classifies it, so re-enqueuing an unprocessed one just re-runs it, never
corrupts it. The resolution batch flips escalated -> agent_triggered first, so
a re-run can't double-draft the same case.

**Bounded blast radius.** One enqueue is capped at 500 — a single click can't
flood the stream with the whole 200K backlog.
""",
}


def render(page_title: str) -> None:
    """Sidebar expander with the active page's engineering notes."""
    notes = _NOTES.get(page_title)
    if not notes:
        return
    with st.expander("🛠️ Under the hood", expanded=False):
        st.markdown(notes)
        st.caption(_STACK_FOOTER)
