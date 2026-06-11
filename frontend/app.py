"""
Dashboard entrypoint: auth gate + navigation.

st.navigation must be called on EVERY script run, authenticated or not.
A run that skips it (the old st.stop()-before-navigation gate) makes
Streamlit fall back to the legacy pages/-directory auto-discovery, which
happily renders a sidebar of every page to anonymous visitors. So the gate
doesn't skip navigation — it swaps the page list: logged-out sessions get a
login-only list (a single page renders no sidebar nav at all), logged-in
sessions get the real pages.
"""

from __future__ import annotations

import streamlit as st

import api_client
import auth
import engineering_notes

st.set_page_config(page_title="ResolveAI", page_icon="📨", layout="wide")

if api_client.is_authenticated():
    pages = [
        st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page("pages/triage_queue.py", title="Triage Queue", icon="📋"),
        st.Page("pages/complaint_detail.py", title="Complaint Detail", icon="🔍"),
        st.Page("pages/analytics.py", title="Analytics", icon="📈"),
    ]
else:
    pages = [st.Page(auth.render_login_page, title="Sign in", icon="🔒")]

# Created before the sidebar so the engineering-notes panel can key off the
# page the visitor is actually on (nav is the selected st.Page).
nav = st.navigation(pages)

if api_client.is_authenticated():
    with st.sidebar:
        if api_client.is_demo():
            st.caption("**Demo mode** — read-only guided tour")
            signout_label = "Exit demo"
        else:
            user = api_client.current_user() or {}
            st.caption(f"Signed in as **{user.get('full_name', 'unknown')}**")
            signout_label = "Sign out"
        engineering_notes.render(nav.title)
        if st.button(signout_label, use_container_width=True):
            api_client.logout()
            st.rerun()

nav.run()
