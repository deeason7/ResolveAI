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
    st.caption("Complaint triage and review — sign in to continue")

    with st.container(border=True):
        st.markdown("**👀 Just looking around?**")
        st.caption(
            "Explore public complaint examples, triage, and response drafts. "
            "The shared demo has server-enforced viewer access."
        )
        if st.button("🚀 Launch the demo", type="primary", use_container_width=True):
            try:
                api_client.start_demo()
            except api_client.ApiError as exc:
                st.error(f"Demo unavailable: {exc.detail}")
            else:
                st.rerun()

    try:
        registration_enabled = api_client.auth_capabilities()["public_registration_enabled"]
    except api_client.ApiError:
        registration_enabled = False
        st.info("Access settings are unavailable. Sign in or contact an administrator.")

    if registration_enabled:
        signin_tab, register_tab = st.tabs(["Sign in", "Create viewer account"])
    else:
        signin_tab = st.container()
        st.caption("Public registration is disabled. Contact an administrator for access.")

    with signin_tab, st.form("signin"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        st.caption("If you cannot sign in, contact an administrator for a password reset.")
        if st.form_submit_button("Sign in", type="primary", use_container_width=True):
            try:
                api_client.login(email, password)
            except api_client.ApiError as exc:
                st.error(f"Sign-in failed: {exc.detail}")
            else:
                st.rerun()

    if not registration_enabled:
        return

    with register_tab, st.form("register"):
        st.caption("Self-service accounts can view cases. Staff access is provisioned separately.")
        full_name = st.text_input("Full name")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password (min 8 characters)", type="password", key="reg_password")
        st.caption("Passwords must fit within 72 UTF-8 bytes; some characters use multiple bytes.")
        if st.form_submit_button("Create account", use_container_width=True):
            try:
                api_client.register(email, full_name, password)
            except api_client.ApiError as exc:
                st.error(f"Registration failed: {exc.detail}")
            else:
                st.success("Account created. Switch to the **Sign in** tab to log in.")
