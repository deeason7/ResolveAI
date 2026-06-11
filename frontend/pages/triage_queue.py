"""
Triage Queue — the reviewer's worklist.

Server does the heavy lifting (priority ordering, filtering, pagination);
this page is a thin view over GET /complaints/queue plus a per-row action
panel. Selecting a row reveals the preview and the Generate Resolution
trigger — the full Complaint Detail page lands with Day 27-28.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
import tour
from api_client import ApiError
from theme import DEMO_HINT, SENTIMENT_BADGES

STATUS_OPTIONS = [
    "all actionable",
    "pending",
    "classified",
    "escalated",
    "agent_triggered",
    "draft_ready",
    "needs_review",
    "resolved",
]

PAGE_SIZES = [25, 50, 100]


def _filters() -> dict:
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.5, 1.5, 1])
    status = c1.selectbox("Status", STATUS_OPTIONS)
    sentiment = c2.selectbox("Sentiment", ["any", "neutral", "negative", "extreme_negative"])
    urgency = c3.slider("Urgency", 1, 5, (1, 5))
    product = c4.text_input("Product (exact)", placeholder="e.g. Mortgage")
    company = c4.text_input("Company (exact)", placeholder="e.g. EQUIFAX, INC.")
    page_size = c5.selectbox("Rows", PAGE_SIZES, index=1)
    page = c5.number_input("Page", min_value=1, value=1, step=1)

    params: dict = {"limit": page_size, "offset": (page - 1) * page_size}
    if status != "all actionable":
        params["status"] = status
    if sentiment != "any":
        params["sentiment"] = sentiment
    # An untouched (1, 5) slider means "no urgency filter" — sending it anyway
    # would also drop rows with no urgency assigned yet.
    if urgency != (1, 5):
        params["urgency_min"], params["urgency_max"] = urgency
    if product:
        params["product"] = product
    if company:
        params["company"] = company
    return params


def _queue_table(items: list[dict]) -> int | None:
    df = pd.DataFrame(items)
    df["sentiment"] = df["sentiment"].map(SENTIMENT_BADGES).fillna("⚪ n/a")
    df["received"] = pd.to_datetime(df["date_received"].fillna(df["created_at"]))
    view = df[
        [
            "priority_score",
            "urgency",
            "sentiment",
            "company",
            "product",
            "status",
            "received",
            "narrative_preview",
        ]
    ]
    event = st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "priority_score": st.column_config.ProgressColumn(
                "Priority", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "urgency": st.column_config.NumberColumn("Urgency", format="%d ⚡"),
            "sentiment": "Sentiment",
            "company": "Company",
            "product": "Product",
            "status": "Status",
            "received": st.column_config.DatetimeColumn("Received", format="YYYY-MM-DD"),
            "narrative_preview": st.column_config.TextColumn("Preview", width="large"),
        },
    )
    rows = event.selection.rows
    return rows[0] if rows else None


def _action_panel(item: dict) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{item['company'] or 'Unknown company'}** — {item['product'] or 'no product'} "
            f"· status `{item['status']}` · intent `{item['intent'] or 'n/a'}`"
        )
        st.write(item["narrative_preview"] + ("…" if len(item["narrative_preview"]) == 200 else ""))

        demo = api_client.is_demo()
        opener, trigger, hint = st.columns([1, 1, 2])
        if opener.button("🔍 Open detail", use_container_width=True):
            st.session_state["detail_complaint_id"] = item["id"]
            st.switch_page("pages/complaint_detail.py")
        if trigger.button(
            "⚙️ Generate resolution",
            type="primary",
            use_container_width=True,
            disabled=demo,
            help=DEMO_HINT if demo else None,
        ):
            try:
                api_client.generate_resolution(item["id"])
            except ApiError as exc:
                # 409s are the API being honest (already running / already
                # resolved / not classified yet) — show the reason, not a stack.
                if exc.status_code == 409:
                    st.warning(exc.detail)
                else:
                    st.error(exc.detail)
            else:
                st.success("Resolution queued — the agent is on it. Check back shortly.")
        hint.caption(f"Complaint `{item['id']}`")


st.title("📋 Triage Queue")
tour.render("triage_queue")

try:
    params = _filters()
    data = api_client.triage_queue(**params)

    if not data["items"]:
        st.info("Queue is empty for these filters.")
    else:
        first = data["offset"] + 1
        last = data["offset"] + len(data["items"])
        st.caption(f"Showing {first:,}–{last:,} of {data['total']:,} complaints")
        selected = _queue_table(data["items"])
        if selected is not None:
            _action_panel(data["items"][selected])
        else:
            st.caption("Select a row to preview and act on it.")
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(f"Could not load the queue: {exc.detail}")
