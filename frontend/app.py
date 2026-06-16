"""
App shell: auth gate + top navigation.

st.navigation must be called on EVERY script run, authenticated or not. A run
that skips it (the old st.stop()-before-navigation gate) makes Streamlit fall
back to the legacy pages/-directory auto-discovery, which happily renders a
sidebar of every page to anonymous visitors. So we always declare the real
pages — which also means a refreshed deep link like /workspace still resolves
instead of hitting Streamlit's "page not found" fallback. Access is gated
*after* navigation: a logged-out run renders the login screen and stops before
any page body executes, and the top nav is only drawn when authenticated, so
nothing leaks to anonymous visitors. (Sign in on a deep link and you land back
on that page.)

Navigation is a custom TOP bar, not the sidebar. Streamlit 1.41 has no native
position="top", so we pass position="hidden" to suppress the built-in sidebar
nav and render a horizontal row of st.page_link in its place. Page links and the
account menu both live on top; the sidebar holds only the optional, collapsed
"Under the hood" engineering panel.
"""

from __future__ import annotations

import streamlit as st

import api_client
import auth
import engineering_notes
import tour

st.set_page_config(
    page_title="ResolveAI",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Single source of truth for the authenticated nav. Each row is
# (script_path, page_title, nav_label, icon):
#   - page_title drives st.Page AND nav.title — it MUST match the
#     engineering_notes._NOTES keys, so it stays the full name.
#   - nav_label is the short label shown in the top bar, so six items fit on
#     one row without wrapping.
_NAV = [
    ("pages/dashboard.py", "Dashboard", "Dashboard", "📊"),
    ("pages/triage_queue.py", "Triage Queue", "Triage", "📋"),
    ("pages/complaint_detail.py", "Complaint Detail", "Detail", "🔍"),
    ("pages/analytics.py", "Analytics", "Analytics", "📈"),
    ("pages/graph_explorer.py", "Graph Explorer", "Graph", "🕸️"),
    ("pages/llmops.py", "LLMOps Observatory", "Observatory", "🔬"),
    ("pages/workspace.py", "Workspace", "Workspace", "⚙️"),
]


def _account_popover() -> None:
    """Account/demo controls, tucked into the top bar's right cell."""
    demo = api_client.is_demo()
    with st.popover("👀 Demo" if demo else "👤 Account", use_container_width=True):
        if demo:
            st.caption("**Demo mode** — read-only guided tour")
            if st.button("🧭 Restart tour", use_container_width=True):
                tour.restart()
                st.switch_page("pages/dashboard.py")
            signout_label = "Exit demo"
        else:
            user = api_client.current_user() or {}
            st.caption(f"Signed in as **{user.get('full_name', 'unknown')}**")
            signout_label = "Sign out"
        if st.button(signout_label, use_container_width=True):
            api_client.logout()
            st.rerun()


def _top_nav() -> None:
    """Horizontal page-link bar in place of the sidebar navigation."""
    with st.container(border=True):
        cols = st.columns(len(_NAV) + 1, vertical_alignment="center")
        for col, (path, _title, label, icon) in zip(cols, _NAV, strict=False):
            with col:
                st.page_link(path, label=label, icon=icon, use_container_width=True)
        with cols[-1]:
            _account_popover()


# Declare the real pages on EVERY run — logged out too — so a refreshed deep
# link like /workspace resolves instead of falling through to "page not found".
pages = [
    st.Page(path, title=title, icon=icon, default=(i == 0))
    for i, (path, title, _label, icon) in enumerate(_NAV)
]
# position="hidden": suppress the built-in sidebar nav; we draw our own on top.
nav = st.navigation(pages, position="hidden")

if not api_client.is_authenticated():
    # Gate AFTER navigation (not by swapping the page list): render login and
    # stop before any page body runs. The top nav isn't drawn, so an anonymous
    # visitor sees only the login screen even though the pages are declared.
    auth.render_login_page()
    st.stop()

_top_nav()
# The one remaining sidebar element: the page's engineering notes, collapsed.
with st.sidebar:
    engineering_notes.render(nav.title)
nav.run()
