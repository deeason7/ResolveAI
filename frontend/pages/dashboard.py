"""
Dashboard Home — the at-a-glance view.

One trends fetch (365 days of daily buckets) feeds the metric cards, the
trend deltas, and the sentiment donut; slicing windows client-side with
pandas beats four near-identical aggregate calls per rerun. The heatmap,
company bar, and activity feed each have their own endpoint.

Card/donut windows anchor to the latest event date in the data, not
wall-clock today — the CFPB dump's date_received ends in the past, so a
"today" window would always read zero. A date_input overrides the anchor;
once live submissions arrive the default lands on the real today anyway.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
from api_client import ApiError

SENTIMENT_COLORS = {
    "neutral": "#2e7d32",
    "negative": "#f9a825",
    "extreme_negative": "#c62828",
    "unclassified": "#9e9e9e",
}


def _window_count(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    mask = (df["day"] >= start) & (df["day"] < end)
    return int(df.loc[mask, "count"].sum())


def _metric_cards(df: pd.DataFrame, anchor: pd.Timestamp) -> None:
    windows = [
        ("Complaints (1d)", anchor, 1),
        ("Complaints (7d)", anchor - timedelta(days=6), 7),
        ("Complaints (30d)", anchor - timedelta(days=29), 30),
    ]
    cols = st.columns(4)
    for col, (label, start, span) in zip(cols, windows, strict=False):
        current = _window_count(df, start, anchor + timedelta(days=1))
        previous = _window_count(df, start - timedelta(days=span), start)
        with col:
            st.metric(label, f"{current:,}", delta=current - previous)

    year_total = int(df["count"].sum())
    classified = int(df.loc[df["sentiment"] != "unclassified", "count"].sum())
    coverage = (classified / year_total * 100) if year_total else 0.0
    with cols[3]:
        st.metric(
            "Classifier coverage (1y)",
            f"{coverage:.1f}%",
            help="Share of complaints in the last 365 days that have a sentiment label",
        )


def _sentiment_donut(df: pd.DataFrame, anchor: pd.Timestamp) -> None:
    st.subheader("Sentiment distribution")
    days = st.radio(
        "Window",
        [7, 30, 90, 365],
        index=1,
        format_func=lambda d: f"{d}d",
        horizontal=True,
        label_visibility="collapsed",
    )
    start = anchor - timedelta(days=days - 1)
    mask = (df["day"] >= start) & (df["day"] <= anchor)
    window = df[mask].groupby("sentiment", as_index=False)["count"].sum()
    if window.empty:
        st.info("No complaints in this window.")
        return
    fig = px.pie(
        window,
        names="sentiment",
        values="count",
        hole=0.45,
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(margin={"t": 10, "b": 10, "l": 10, "r": 10}, height=320)
    st.plotly_chart(fig, use_container_width=True)


def _urgency_heatmap(breakdown: dict) -> None:
    st.subheader("Urgency by product")
    items = breakdown["items"][:10]  # already volume-sorted by the API
    if not items:
        st.info("No complaints yet.")
        return
    matrix = [[row["urgency_counts"].get(str(u), 0) for u in range(1, 6)] for row in items]
    classified = sum(sum(r) for r in matrix)
    total = sum(row["total"] for row in items)
    fig = px.imshow(
        matrix,
        x=[f"U{u}" for u in range(1, 6)],
        y=[row["product"] for row in items],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        margin={"t": 10, "b": 10, "l": 10, "r": 10}, height=360, coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Classified complaints only — {classified:,} of {total:,} in the top 10 products")


def _top_companies(risk: dict) -> None:
    st.subheader("Top companies by volume")
    items = risk["items"]
    if not items:
        st.info("No complaints yet.")
        return
    df = pd.DataFrame(items).sort_values("total_complaints")  # h-bar renders bottom-up
    fig = px.bar(
        df,
        x="total_complaints",
        y="company",
        orientation="h",
        hover_data=["avg_urgency", "urgent_count", "extreme_negative_count"],
    )
    fig.update_layout(
        margin={"t": 10, "b": 10, "l": 10, "r": 10}, height=360, xaxis_title=None, yaxis_title=None
    )
    st.plotly_chart(fig, use_container_width=True)


def _recent_activity() -> None:
    st.subheader("Recent activity")
    recent = api_client.list_complaints(limit=20)
    if not recent["items"]:
        st.info("Nothing submitted yet.")
        return
    df = pd.DataFrame(recent["items"])[
        ["created_at", "company", "product", "status", "sentiment", "urgency"]
    ]
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("Received", format="MMM D, HH:mm"),
            "company": "Company",
            "product": "Product",
            "status": "Status",
            "sentiment": "Sentiment",
            "urgency": st.column_config.NumberColumn("Urgency", format="%d"),
        },
    )


st.title("📊 Dashboard")

try:
    trends = api_client.sentiment_trends(days=365)
    points = pd.DataFrame(trends["points"], columns=["day", "sentiment", "count"])
    points["day"] = pd.to_datetime(points["day"])

    latest = points["day"].max().date() if not points.empty else date.today()
    # The widget sits below the cards, so read its value from session_state:
    # a change is committed there before the rerun reaches this line.
    anchor = pd.Timestamp(st.session_state.get("ref_date", latest))

    _metric_cards(points, anchor)
    if anchor.date() == latest:
        st.caption(
            f"Windows are relative to {anchor:%B %d, %Y} — the most recent complaint "
            "on record, not today's date. Override below."
        )
    else:
        st.caption(
            f"Windows are relative to {anchor:%B %d, %Y} "
            f"(most recent complaint: {latest:%B %d, %Y})."
        )
    ref_col, _ = st.columns([1, 4])
    with ref_col:
        st.date_input("Reference date", value=latest, key="ref_date")
    st.divider()

    left, right = st.columns(2)
    with left:
        _sentiment_donut(points, anchor)
    with right:
        _urgency_heatmap(api_client.products_breakdown())

    left, right = st.columns(2)
    with left:
        _top_companies(api_client.companies_risk(limit=10))
    with right:
        _recent_activity()
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(f"Could not load dashboard data: {exc.detail}")
