"""
Backend API wrapper for the dashboard.

Every page talks to the backend through this module, never through raw
httpx — one place owns the base URL, bearer-token injection, the silent
refresh-and-retry on 401, and error translation into ApiError.

Token model mirrors the backend's auth design: the short-lived access
token lives in st.session_state and goes out as a Bearer header; the
refresh token is an httpOnly cookie we never see, carried by a per-session
httpx.Client whose cookie jar survives Streamlit reruns.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
API_PREFIX = "/api/v1"
TIMEOUT_S = 30.0

_CLIENT_KEY = "_http_client"
_TOKEN_KEY = "access_token"
_USER_KEY = "user"
_DEMO_KEY = "demo_mode"

# Shared walk-up account for recruiters/visitors who won't register. Its
# credentials are deliberately not secret — it can do nothing a self-registered
# account couldn't, and the UI runs demo sessions read-only on top.
DEMO_EMAIL = os.environ.get("DEMO_EMAIL", "demo@resolveai-demo.com")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo-resolveai-2026")


class ApiError(Exception):
    """Backend call failed. status_code 0 means the API was unreachable."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


def _http() -> httpx.Client:
    """The session's HTTP client. Streamlit reruns the script top-to-bottom on
    every interaction, so anything that must outlive a rerun — here, the
    refresh-cookie jar — has to live in session_state, not module globals."""
    if _CLIENT_KEY not in st.session_state:
        st.session_state[_CLIENT_KEY] = httpx.Client(
            base_url=API_URL + API_PREFIX, timeout=TIMEOUT_S
        )
    return st.session_state[_CLIENT_KEY]


def _humanize_validation(errors: list[Any]) -> str:
    """Translate FastAPI's 422 validation list into plain language.

    The raw payload — e.g. ``[{'type': 'value_error', 'loc': ['body', 'email'],
    'msg': 'value is not a valid email address: ...'}]`` — is for developers;
    users get one readable sentence per offending field instead of dict syntax.
    """
    parts: list[str] = []
    for err in errors:
        if not isinstance(err, dict):
            parts.append(str(err))
            continue
        loc = err.get("loc") or []
        # loc is like ["body", "email"]; the last hop is the field name.
        field = str(loc[-1]) if loc else ""
        if field == "email":
            parts.append("Please enter a valid email address.")
            continue
        label = field.replace("_", " ").capitalize() if field else "Input"
        # Pydantic v2 prefixes custom-validator messages with "Value error, ".
        msg = str(err.get("msg", "is invalid"))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        parts.append(f"{label}: {msg}")
    return " ".join(parts) if parts else "Please check your input and try again."


def _detail(resp: httpx.Response) -> str:
    """Best human-readable explanation for a failed response.

    FastAPI returns a plain-string ``detail`` for HTTPExceptions (already
    user-facing) but a structured *list* for 422 validation errors — the latter
    is run through ``_humanize_validation`` instead of being str()'d into a wall
    of dict syntax.
    """
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"
    if isinstance(detail, list):
        return _humanize_validation(detail)
    return str(detail)


def _try_refresh() -> bool:
    """Trade the refresh cookie for a new access token. The jar sends the
    cookie and absorbs the rotated one automatically."""
    try:
        resp = _http().post("/auth/refresh")
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    st.session_state[_TOKEN_KEY] = resp.json()["access_token"]
    return True


def _request(method: str, path: str, *, _retried: bool = False, **kwargs: Any) -> httpx.Response:
    headers = dict(kwargs.pop("headers", {}))
    token = st.session_state.get(_TOKEN_KEY)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = _http().request(method, path, headers=headers, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(0, f"API unreachable: {exc}") from exc

    if resp.status_code == 401 and token and not _retried:
        if _try_refresh():
            return _request(method, path, _retried=True, **kwargs)
        clear_session()  # refresh dead too — force a clean re-login

    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _detail(resp))
    return resp


# ── auth & session ────────────────────────────────────────────────────────────


def is_authenticated() -> bool:
    return _TOKEN_KEY in st.session_state


def current_user() -> dict | None:
    return st.session_state.get(_USER_KEY)


def clear_session() -> None:
    # "tour_step" belongs to tour.py — popped here so a demo logout fully
    # resets the guided tour (importing tour would be circular).
    for key in (_TOKEN_KEY, _USER_KEY, _DEMO_KEY, "tour_step"):
        st.session_state.pop(key, None)


def is_demo() -> bool:
    return bool(st.session_state.get(_DEMO_KEY))


def start_demo() -> None:
    """Sign in as the shared demo account, creating it on first use.

    Self-healing on a fresh database: login 401 → register → proceed. Anyone
    could do the same through the open register endpoint, so this adds no new
    surface — it just removes the form for people who only want to look.
    """
    try:
        login(DEMO_EMAIL, DEMO_PASSWORD)
    except ApiError as exc:
        if exc.status_code != 401:
            raise
        register(DEMO_EMAIL, "Demo Viewer", DEMO_PASSWORD)
    st.session_state[_DEMO_KEY] = True


def login(email: str, password: str) -> None:
    resp = _http().post("/auth/login", json={"email": email, "password": password})
    if resp.status_code != 200:
        raise ApiError(resp.status_code, _detail(resp))
    st.session_state[_TOKEN_KEY] = resp.json()["access_token"]
    st.session_state[_USER_KEY] = me()


def register(email: str, full_name: str, password: str) -> None:
    resp = _http().post(
        "/auth/register",
        json={"email": email, "full_name": full_name, "password": password},
    )
    if resp.status_code != 201:
        raise ApiError(resp.status_code, _detail(resp))
    st.session_state[_TOKEN_KEY] = resp.json()["access_token"]
    st.session_state[_USER_KEY] = me()


def logout() -> None:
    try:
        _request("POST", "/auth/logout")
    except ApiError:
        pass  # revocation is best-effort; the local session dies regardless
    clear_session()


def me() -> dict:
    return _request("GET", "/auth/me").json()


# ── complaints ────────────────────────────────────────────────────────────────


def list_complaints(**filters: Any) -> dict:
    params = {k: v for k, v in filters.items() if v is not None}
    return _request("GET", "/complaints/", params=params).json()


def triage_queue(**filters: Any) -> dict:
    params = {k: v for k, v in filters.items() if v is not None}
    return _request("GET", "/complaints/queue", params=params).json()


def get_complaint(complaint_id: str) -> dict:
    return _request("GET", f"/complaints/{complaint_id}").json()


def submit_complaint(payload: dict) -> dict:
    return _request("POST", "/complaints/", json=payload).json()


def similar_complaints(complaint_id: str, limit: int = 5, product: str | None = None) -> dict:
    params: dict[str, Any] = {"limit": limit}
    if product:
        params["product"] = product
    return _request("GET", f"/complaints/{complaint_id}/similar", params=params).json()


# ── knowledge graph ───────────────────────────────────────────────────────────


def company_profile(name: str) -> dict:
    # quote(safe="") because company names contain commas, ampersands and the
    # odd slash — an unescaped "/" would split the path and 404.
    return _request("GET", f"/graph/company/{quote(name, safe='')}").json()


def graph_explore(node_id: str, depth: int = 2) -> dict:
    return _request("GET", "/graph/explore", params={"node_id": node_id, "depth": depth}).json()


# ── llmops ────────────────────────────────────────────────────────────────────


def llmops_costs(days: int = 90) -> dict:
    return _request("GET", "/llmops/costs", params={"days": days}).json()


def llmops_latency(days: int = 90) -> dict:
    return _request("GET", "/llmops/latency", params={"days": days}).json()


def llmops_routing(days: int = 90) -> dict:
    return _request("GET", "/llmops/routing", params={"days": days}).json()


def llmops_drift(days: int = 90) -> dict:
    return _request("GET", "/llmops/drift", params={"days": days}).json()


def llmops_guardrails(layer: str | None = None, limit: int = 100) -> dict:
    params: dict[str, Any] = {"limit": limit}
    if layer:
        params["layer"] = layer
    return _request("GET", "/llmops/guardrails", params=params).json()


# ── analytics ─────────────────────────────────────────────────────────────────


def sentiment_trends(days: int = 30) -> dict:
    return _request("GET", "/analytics/sentiment/trends", params={"days": days}).json()


def products_breakdown() -> dict:
    return _request("GET", "/analytics/products/breakdown").json()


def companies_risk(limit: int = 10) -> dict:
    return _request("GET", "/analytics/companies/risk", params={"limit": limit}).json()


# ── resolutions ───────────────────────────────────────────────────────────────


def generate_resolution(complaint_id: str) -> dict:
    return _request("POST", f"/resolutions/{complaint_id}/generate").json()


def get_resolution(complaint_id: str) -> dict:
    return _request("GET", f"/resolutions/{complaint_id}").json()


def list_revisions(complaint_id: str) -> list[dict]:
    return _request("GET", f"/resolutions/{complaint_id}/revisions").json()


def approve_resolution(complaint_id: str) -> dict:
    return _request("POST", f"/resolutions/{complaint_id}/approve").json()


def reject_resolution(complaint_id: str, feedback: str) -> dict:
    return _request(
        "POST", f"/resolutions/{complaint_id}/reject", json={"feedback": feedback}
    ).json()


# ── workspace ─────────────────────────────────────────────────────────────────


def workspace_board() -> dict:
    return _request("GET", "/workspace/board").json()


def enqueue_classification(limit: int = 50) -> dict:
    return _request("POST", "/workspace/enqueue/classification", params={"limit": limit}).json()


def enqueue_resolution_batch(limit: int = 50) -> dict:
    return _request("POST", "/workspace/enqueue/resolution", params={"limit": limit}).json()
