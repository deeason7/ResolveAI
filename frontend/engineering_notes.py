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
    "Stack: FastAPI · PostgreSQL · Redis Streams · Qdrant · Neo4j · Streamlit. "
    "Ollama is optional; production examples use cloud inference. These notes describe "
    "implementation choices and limits, not verified production readiness."
)

_NOTES = {
    "Dashboard": """
**Shared aggregate payload.** Cards and chart windows reuse daily buckets instead
of loading complaint narratives into the browser. Date windows anchor to the
newest complaint, which makes historical records visible.

**Coverage matters.** Unclassified volume remains visible. Imported records,
classified records, drafted responses and approved cases are different counts.
Sentiment describes a model/teacher label, not a verified customer outcome.
""",
    "Triage Queue": """
**Database ordering and pagination.** Filters and sorting run in SQL; rows carry
narrative previews and the detail endpoint loads the full text. Explicit
`NULLS LAST` keeps unscored rows from leading a descending PostgreSQL sort.

**Priority is a suggestion.** Review the original complaint and unavailable/stale
classification evidence. There is no measured reviewer-time saving or recall
improvement attached to this queue yet.
""",
    "Complaint Detail": """
**Similarity, not proven precedent.** Qdrant searches 384-dimensional embeddings;
the similar-complaint endpoint hydrates matches from PostgreSQL and excludes the
query case. Historical response categories do not prove successful remediation.
Retrieval relevance and latency need workload-specific evaluation.

**Limited checks.** Pattern matching, citation-title matching and a tone judge
provide signals. They do not establish factual/legal accuracy. The current UI can
show an unrun tone layer as passed; inspect actual recorded violations and context.

**Review integrity still needs work.** Editing the displayed draft does not save
it. Approval currently selects a stored draft by complaint rather than requiring
the visible draft identity/version. The resulting resolved label records approval,
not delivery or a verified customer outcome.
""",
    "Analytics": """
**Mixed sources.** SQL produces complaint aggregates; the company table supplements
them with Neo4j-derived scores and reference links. Missing graph data reduces
coverage. These heuristic scores are not independently validated risk estimates.

**Known growth limits.** The page makes serial graph calls for company profiles.
Measure query count and page latency before increasing the dataset or claiming
interactive performance at a particular scale.
""",
    "Graph Explorer": """
**Bounded graph queries.** APOC supplies a neighborhood with a capped traversal
depth. pyvis renders that data inside a Streamlit HTML component; frontend behavior
still requires browser-level verification.

**Reference provenance matters.** Seeded company/product/issue/reference links are
context for review. A relationship does not prove legal applicability or a
successful remediation. Node titles and IDs must remain distinct when recentering.
""",
    "LLMOps Observatory": """
**Recorded usage, incomplete accounting.** Workers persist provider, model, tokens,
estimated cost and latency alongside some business writes. Failed attempts and
unknown pricing are not fully represented. A zero cost is not proof of a free call,
and the total is not the cost of operating or processing the whole corpus.

**Provider failure is distinct.** Provider `none` means no model answered;
`was_fallback` indicates a cloud response after local failure. The reviewed
baseline's deterministic urgency-3 result is not a reliable manual-triage state.

**Scale needs measurement.** Some latency/violation endpoints load matching rows
into Python. A bounded response does not imply bounded query or memory use.
""",
    "Workspace": """
**Business state and queue state differ.** Case counts read PostgreSQL status;
Redis supplies stream activity. The reads are not a single consistent snapshot,
and approval currently uses the resolved label without recording a real outcome.

**Publication is not atomic.** Producers share message helpers, but a request can
publish before its database commit. Replay can overwrite later state. Redis
consumer groups alone do not guarantee lossless work or prevent duplicate effects;
durable jobs and stale-worker fencing remain required improvements.

**Bounded submission.** A staff batch request has a size cap, but existing queued
work, retries and provider limits still determine completion time.
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
