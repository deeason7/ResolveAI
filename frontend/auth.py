"""
Login / register screen.

Rendered by app.py whenever there's no access token in the session; on
success we st.rerun() and the gate in app.py falls through to the real
navigation. Forms are st.form so typing doesn't trigger a rerun per
keystroke — the script only re-executes on submit.
"""

from __future__ import annotations

import streamlit as st

import api_client


def render_login_page() -> None:
    st.title("ResolveAI")
    st.caption("Intelligent complaint resolution — sign in to continue")

    with st.container(border=True):
        st.markdown("**👀 Just looking around?**")
        st.caption(
            "Take a guided tour of the live system — 200,000 real CFPB complaints, "
            "AI triage, precedent search, and agent-drafted resolutions. "
            "No account needed; demo sessions are read-only."
        )
        if st.button("🚀 Launch the demo", type="primary", use_container_width=True):
            try:
                api_client.start_demo()
            except api_client.ApiError as exc:
                st.error(f"Demo unavailable: {exc.detail}")
            else:
                st.rerun()

    signin_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with signin_tab, st.form("signin"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary", use_container_width=True):
            try:
                api_client.login(email, password)
            except api_client.ApiError as exc:
                st.error(f"Sign-in failed: {exc.detail}")
            else:
                st.rerun()

    with register_tab, st.form("register"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password (min 8 chars)", type="password", key="reg_password")
        if st.form_submit_button("Create account", use_container_width=True):
            try:
                api_client.register(email, full_name, password)
            except api_client.ApiError as exc:
                st.error(f"Registration failed: {exc.detail}")
            else:
                st.success("Account created. Switch to the **Sign in** tab to log in.")
