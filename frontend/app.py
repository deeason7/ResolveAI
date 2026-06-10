"""
Dashboard entrypoint: auth gate + navigation.

st.navigation (not the pages/ auto-discovery convention) so the page list
is built AFTER the auth check — an unauthenticated visitor sees only the
login screen, never a sidebar full of pages that would each error out.
"""

from __future__ import annotations

import streamlit as st

import api_client
import auth

st.set_page_config(page_title="ResolveAI", page_icon="📨", layout="wide")

if not api_client.is_authenticated():
    auth.render_login_page()
    st.stop()

nav = st.navigation(
    [
        st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page("pages/triage_queue.py", title="Triage Queue", icon="📋"),
    ]
)

with st.sidebar:
    user = api_client.current_user() or {}
    st.caption(f"Signed in as **{user.get('full_name', 'unknown')}**")
    if st.button("Sign out", use_container_width=True):
        api_client.logout()
        st.rerun()

nav.run()
