"""
Per-page "under the hood" notes for the sidebar.

The pages demo the product; this panel demos the engineering. Each page
gets the architectural decisions behind it — what was chosen, what it beat,
and why — so a visitor or an interviewer sees the reasoning, not just the
pixels. Auto-expanded in demo sessions, tucked away for daily work.
"""

from __future__ import annotations

import streamlit as st

import api_client

_STACK_FOOTER = (
    "Auth: short-lived JWT + httpOnly rotating refresh cookie, silent re-auth "
    "on 401; every page talks through one API-client seam. Stack: FastAPI · "
    "PostgreSQL · Redis Streams · Qdrant · Neo4j · fine-tuned SLM + cloud LLM "
    "fallback · Streamlit — fully Dockerized."
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
}


def render(page_title: str) -> None:
    """Sidebar expander with the active page's engineering notes."""
    notes = _NOTES.get(page_title)
    if not notes:
        return
    with st.expander("🛠️ Under the hood", expanded=api_client.is_demo()):
        st.markdown(notes)
        st.caption(_STACK_FOOTER)
