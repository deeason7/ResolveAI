"""
Guided tour for demo sessions — a Back/Next stepper, not a lecture.

Each step is two business-first sentences plus one thing to try; Next walks
the visitor across pages in a deliberate order. If they wander off-route
the tour quietly re-syncs to wherever they are (free play is the point —
the stepper is a rail, not a cage). Technical depth lives in the sidebar's
"Under the hood" panel, which only the final step advertises.
"""

from __future__ import annotations

import streamlit as st

import api_client

# api_client.clear_session() pops this on logout (string kept in sync there).
_STEP_KEY = "tour_step"

STEPS = [
    {
        "page": "dashboard",
        "file": "pages/dashboard.py",
        "title": "Welcome to ResolveAI",
        "body": "Explore public consumer complaints, inspect suggested priorities, and "
        "review generated responses. Imported cases, drafts, and approvals are "
        "different measures of work.",
        "try": "Press **Next** to walk through, or just wander — the tour follows you.",
    },
    {
        "page": "dashboard",
        "file": "pages/dashboard.py",
        "title": "The operations pulse",
        "body": "See loaded complaint volume, assigned sentiment, and urgency by product. "
        "These are signals to inspect, not verified findings about a company.",
        "try": "Switch the donut window, or pick a different reference date under the cards.",
    },
    {
        "page": "triage_queue",
        "file": "pages/triage_queue.py",
        "title": "The morning worklist",
        "body": "Sort and filter complaints by their recorded priority. Read the original "
        "narrative before relying on a model's urgency judgment.",
        "try": "Select any row, then hit **🔍 Open detail**.",
    },
    {
        "page": "complaint_detail",
        "file": "pages/complaint_detail.py",
        "title": "One complaint, full context",
        "body": "Read the complaint, its classification, and similar public records. "
        "Historical response categories do not prove that a customer's problem "
        "was solved.",
        "try": "Toggle **Same product only**, or jump into a similar case with **Open**.",
    },
    {
        "page": "complaint_detail",
        "file": "pages/complaint_detail.py",
        "title": "Draft → check → human",
        "body": "Inspect a generated draft, its source context, and recorded checks. "
        "Pattern checks and tone scores do not establish factual or legal accuracy; "
        "approval does not establish delivery or resolution.",
        "try": "Expand **Agent chain of thought** to inspect the tool/workflow trace.",
    },
    {
        "page": "analytics",
        "file": "pages/analytics.py",
        "title": "The leadership view",
        "body": "Compare trends and product counts with graph-derived company scores. "
        "Check each column's source and coverage before drawing conclusions.",
        "try": "Click a column header to sort the scorecard.",
    },
    {
        "page": "graph_explorer",
        "file": "pages/graph_explorer.py",
        "title": "Institutional memory, drawn",
        "body": "Explore companies, products, issues, and seeded reference material. "
        "A graph link alone does not establish that a rule applies to a complaint.",
        "try": "Search any company, pick a node, and **🧭 Explore from here**.",
    },
    {
        "page": "workspace",
        "file": "pages/workspace.py",
        "title": "The engine room",
        "body": "Inspect recorded case states and available worker information. "
        "Response approval records saved text approval; delivery and outcome remain unrecorded.",
        "try": "Compare the **Pipeline** counts with the queue; staff controls require "
        "write access.",
    },
    {
        "page": "llmops",
        "file": "pages/llmops.py",
        "title": "Inspect recorded model activity",
        "body": "Explore logged calls, estimated cost, latency, and violations. "
        "This is incomplete call telemetry, not a total operating-cost or safety report.",
        "try": "Hover the routing donut — provider none means no model produced that result.",
    },
    {
        "page": "llmops",
        "file": "pages/llmops.py",
        "title": "That's ResolveAI",
        "body": "This prototype helps inspect complaint data and proposed responses. "
        "For implementation details and known limits, open "
        "**🛠️ Under the hood** in the sidebar on any page.",
        "try": None,
    },
]


def restart() -> None:
    """Reset to step one. The sidebar's Restart-tour button calls this."""
    st.session_state[_STEP_KEY] = 0


def _goto(idx: int, current_page: str) -> None:
    st.session_state[_STEP_KEY] = idx
    if STEPS[idx]["page"] != current_page:
        st.switch_page(STEPS[idx]["file"])
    st.rerun()


def render(page: str) -> None:
    """Compact stepper banner for `page` — demo sessions only."""
    if not api_client.is_demo():
        return
    if _STEP_KEY not in st.session_state:
        st.session_state[_STEP_KEY] = 0  # fresh demo session → tour starts itself
    idx = st.session_state[_STEP_KEY]
    if idx is None:
        return  # tour finished or dismissed

    # Wandered off-route? Re-sync to this page's first step.
    if STEPS[idx]["page"] != page:
        synced = next((i for i, s in enumerate(STEPS) if s["page"] == page), None)
        if synced is None:
            return
        idx = st.session_state[_STEP_KEY] = synced

    step = STEPS[idx]
    last = idx == len(STEPS) - 1
    with st.container(border=True):
        st.markdown(f"🧭 **{step['title']}** · step {idx + 1} of {len(STEPS)}")
        st.markdown(step["body"])
        if step["try"]:
            st.caption(f"💡 {step['try']}")
        back, nxt, skip, _ = st.columns([1, 1, 1, 3])
        if back.button("⬅ Back", key="tour_back", disabled=idx == 0, use_container_width=True):
            _goto(idx - 1, page)
        if nxt.button(
            "Finish ✓" if last else "Next ➡",
            key="tour_next",
            type="primary",
            use_container_width=True,
        ):
            if last:
                st.session_state[_STEP_KEY] = None
                st.rerun()
            else:
                _goto(idx + 1, page)
        if not last and skip.button("End tour", key="tour_skip", use_container_width=True):
            st.session_state[_STEP_KEY] = None
            st.rerun()
