"""
Guided tour for demo sessions.

Each page renders a walkthrough box explaining what's on screen, what's real,
and where to go next — so a visitor who skipped registration still gets the
full story without anyone standing next to them. The boxes only appear in
demo mode; signed-in reviewers know their own tools.
"""

from __future__ import annotations

import streamlit as st

import api_client

_WELCOME = """
**Welcome to ResolveAI** — an intelligent complaint-resolution engine running over
**200,000 real consumer complaints** from the CFPB's public database.

The pipeline behind this dashboard: a complaint arrives → a **fine-tuned small language
model** classifies its sentiment, intent and urgency → **vector search** digs up similar
past cases and a **knowledge graph** supplies the regulations in play → an **agent**
drafts a regulator-aware response → a **four-layer guardrail engine** validates the
draft → a **human reviewer** approves or rejects it. This dashboard is the human's seat.

You're in a **read-only demo session** — look at everything, the write buttons are
politely disabled. 💡 Every page also has **🛠️ Under the hood** in the sidebar:
the architectural decisions behind exactly what you're looking at.
"""

_PAGES = {
    "dashboard": _WELCOME
    + """
---
**This page — Dashboard Home, the at-a-glance view:**
- **Metric cards** — complaint volume over 1/7/30-day windows, anchored to the newest
  complaint on record (the caption says which date; the picker below changes it).
- **Sentiment donut** — neutral / negative / extreme-negative split for the window you
  pick; *unclassified* volume is shown honestly instead of hidden.
- **Urgency × product heatmap** — where the urgent complaints concentrate.
- **Top companies** — volume leaders; hover for severity stats.
- **Recent activity** — the latest complaints in the system.

→ Next: open **Triage Queue** in the sidebar — the reviewer's worklist.
""",
    "triage_queue": """
**Triage Queue — the reviewer's worklist.**

- Rows are sorted by **priority score**, then urgency, then oldest-first — so two
  equally urgent complaints can't queue-jump by recency. Sorting, filtering and
  pagination all happen server-side over the full corpus.
- **Filters** — status, sentiment, urgency range, product, company.
- **Select a row** to preview it. In a live session the ⚙️ button hands the complaint
  to the resolution agent; it's disabled here.

→ Select a row and click **🔍 Open detail** to follow one complaint all the way down.
""",
    "complaint_detail": """
**Complaint Detail — everything the system knows about one complaint.**

- **Narrative & classification** — the consumer's own words next to the model's read:
  sentiment, intent, urgency 1-5, computed priority.
- **Similar complaints** — nearest neighbors by embedding similarity across all 200K
  narratives (384-dim sentence-transformer vectors, cosine search), each shown with how
  it was historically resolved — the precedents a reviewer would reach for. Try the
  **same product only** toggle and the per-row **Open** buttons.
- **Resolution panel** — when the agent has drafted a response: its step-by-step
  reasoning, a verdict per guardrail layer (*structural · content safety · regulatory
  accuracy · tone*), the draft itself, and the approve / reject-with-feedback controls a
  reviewer uses (rejection feedback feeds the agent's regeneration). Disabled in demo.

→ Next: **Analytics** in the sidebar for the corpus-level picture.
""",
    "analytics": """
**Analytics — the whole corpus in aggregate.**

- **Weekly sentiment trend** — volume and classification coverage over time.
- **Product treemap** — sized by volume, colored by urgent-complaint count.
- **Urgency histogram** — including the honest *n/a* bucket: the classifier's backlog.
- **Company risk scorecard** — Postgres severity aggregates joined live with each
  company's risk score and linked regulation violations from the **Neo4j knowledge
  graph**. Two stores, one table; every column's source is in the caption.

→ Next: **Graph Explorer** — walk that knowledge graph visually.
""",
    "graph_explorer": """
**Graph Explorer — the knowledge graph, live.**

Companies, products, issues, regulations and resolution patterns live in a **Neo4j
graph** seeded from the complaint corpus; the agent queries it for regulations and
company history when drafting responses.

- **Search any node** — a company ("EQUIFAX, INC."), product, issue, or regulation —
  and the canvas renders its neighborhood out to the depth you pick (capped at 3:
  this graph is dense, and unbounded traversals would pull most of it back).
- **Drag, zoom, hover** for node properties; colors mark the node type (legend above
  the canvas).
- **Inspect panel** — pick a node for its details (companies show live complaint
  totals and risk score) and hit *Explore from here* to re-center the graph on it.

→ Last stop: **LLMOps Observatory** — what the AI itself costs, how fast it runs,
and every guardrail catch.
""",
    "llmops": """
**LLMOps Observatory — the dashboard about the models themselves.**

Every model call this system makes is metered into a log with provider, tokens, cost
and latency. This page is that telemetry:

- **Spend** — daily bars per provider with a cumulative line; the whole demo corpus
  was classified for pennies.
- **Call routing** — local fine-tuned model vs cloud vs the **fail-closed path**
  (when no provider answers, complaints get flagged maximum-severity for humans
  instead of guessed labels — red in the donut, by design).
- **Latency by operation** — p50/p95, log scale, because local CPU inference and
  cloud calls live in different worlds.
- **Classifier output over time** — drift watch, bucketed by when calls actually ran.
- **Guardrail violation log** — every rule the validation engine has tripped,
  filterable by layer.

That's the full tour — thanks for looking around! The sidebar's **🛠️ Under the
hood** has the architecture story for every page you just saw.
""",
}


def render(page: str) -> None:
    """Show the walkthrough box for `page` — demo sessions only."""
    if not api_client.is_demo():
        return
    copy = _PAGES.get(page)
    if copy:
        with st.expander("📖 Demo tour — what am I looking at?", expanded=True):
            st.markdown(copy)
