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
        "body": "A backlog of **200,000 real consumer complaints**, turned into a "
        "prioritized, explainable workflow — AI drafts responses in seconds, "
        "a human approves every word.",
        "try": "Press **Next** to walk through, or just wander — the tour follows you.",
    },
    {
        "page": "dashboard",
        "file": "pages/dashboard.py",
        "title": "The operations pulse",
        "body": "Volume, customer mood, urgency hot spots, and the companies driving "
        "risk — the picture an operations lead checks every morning.",
        "try": "Switch the donut window, or pick a different reference date under the cards.",
    },
    {
        "page": "triage_queue",
        "file": "pages/triage_queue.py",
        "title": "The morning worklist",
        "body": "Most damaging complaints first, so teams spend minutes deciding what "
        "matters instead of hours reading. Filters answer *“show me urgent "
        "cases at company X”* instantly.",
        "try": "Select any row, then hit **🔍 Open detail**.",
    },
    {
        "page": "complaint_detail",
        "file": "pages/complaint_detail.py",
        "title": "One complaint, full context",
        "body": "The customer's own words, how the AI read them, and how similar cases "
        "were actually resolved before — precedent on tap instead of tribal memory.",
        "try": "Toggle **Same product only**, or jump into a similar case with **Open**.",
    },
    {
        "page": "complaint_detail",
        "file": "pages/complaint_detail.py",
        "title": "Draft → check → human",
        "body": "An agent writes a regulation-aware draft, four guardrails vet it, and a "
        "reviewer approves it or sends it back with feedback the AI must address. "
        "Days of response time become minutes — accountability stays human.",
        "try": "Expand **Agent chain of thought** to watch it show its work.",
    },
    {
        "page": "analytics",
        "file": "pages/analytics.py",
        "title": "The leadership view",
        "body": "Trends, product hot spots, and a company risk scorecard — the view for "
        "deciding where to put people, process, and pressure.",
        "try": "Click a column header to sort the scorecard.",
    },
    {
        "page": "graph_explorer",
        "file": "pages/graph_explorer.py",
        "title": "Institutional memory, drawn",
        "body": "Companies, products, issues, and regulations as one connected map — the "
        "same map the AI consults when it cites a rule in a draft.",
        "try": "Search any company, pick a node, and **🧭 Explore from here**.",
    },
    {
        "page": "workspace",
        "file": "pages/workspace.py",
        "title": "The engine room",
        "body": "Every complaint rides one line — queued → classified → in resolution → "
        "human review → resolved — and this board shows how many sit at each stage right "
        "now. It doubles as the control panel: feed a batch in and watch the counts move.",
        "try": "Watch the **Pipeline** row — each case sits at exactly one stage. "
        "(Enqueueing is read-only in the demo.)",
    },
    {
        "page": "llmops",
        "file": "pages/llmops.py",
        "title": "AI you can govern",
        "body": "What the AI costs (this entire corpus: about **three cents**), how fast "
        "it answers, and receipts that it fails safe — every blocked draft and "
        "fallback is on the record.",
        "try": "Hover the routing donut — red means *the AI refused to guess*.",
    },
    {
        "page": "llmops",
        "file": "pages/llmops.py",
        "title": "That's ResolveAI",
        "body": "Explainable triage, guarded drafts, humans in charge. Keep playing — "
        "everything here is safe to click. Curious how it's built? Open "
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
