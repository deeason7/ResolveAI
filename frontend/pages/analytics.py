"""
Analytics — the deep-dive view.

The Dashboard answers "how are we doing right now"; this page answers
"what does the corpus look like". Weekly sentiment trend, product
treemap, urgency histogram, and a company scorecard that joins the
Postgres severity aggregates with the knowledge graph's risk score —
two stores, one table.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
from api_client import ApiError
from theme import PLOT_MARGIN, SENTIMENT_COLORS

SCORECARD_COMPANIES = 15


def _weekly_trend() -> None:
    st.subheader("Sentiment trend (weekly)")
    trends = api_client.sentiment_trends(days=365)
    df = pd.DataFrame(trends["points"], columns=["day", "sentiment", "count"])
    if df.empty:
        st.info("No complaints in the last year.")
        return
    df["day"] = pd.to_datetime(df["day"])
    # The API stays day-granular on purpose; the weekly roll-up is one line here.
    weekly = df.groupby([pd.Grouper(key="day", freq="W"), "sentiment"])["count"].sum().reset_index()
    fig = px.line(
        weekly,
        x="day",
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        markers=True,
    )
    fig.update_layout(margin=PLOT_MARGIN, height=340, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def _product_treemap(items: list[dict]) -> None:
    st.subheader("Product breakdown")
    df = pd.DataFrame(
        [
            {
                "product": row["product"],
                "total": row["total"],
                "urgent": row["urgency_counts"].get("4", 0) + row["urgency_counts"].get("5", 0),
            }
            for row in items
            if row["total"] > 0
        ]
    )
    if df.empty:
        st.info("No complaints yet.")
        return
    fig = px.treemap(
        df,
        path=["product"],
        values="total",
        color="urgent",
        color_continuous_scale="Reds",
        hover_data=["urgent"],
    )
    fig.update_layout(margin=PLOT_MARGIN, height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sized by complaint volume, colored by classified urgent (U4+U5) count.")


def _urgency_histogram(items: list[dict]) -> None:
    st.subheader("Urgency distribution")
    counts = {f"U{u}": 0 for u in range(1, 6)}
    unclassified = 0
    for row in items:
        unclassified += row["unclassified"]
        for u in range(1, 6):
            counts[f"U{u}"] += row["urgency_counts"].get(str(u), 0)
    df = pd.DataFrame(
        {"urgency": [*counts.keys(), "n/a"], "count": [*counts.values(), unclassified]}
    )
    fig = px.bar(
        df,
        x="urgency",
        y="count",
        color="urgency",
        color_discrete_sequence=px.colors.sequential.Reds + ["#9e9e9e"],
    )
    fig.update_layout(
        margin=PLOT_MARGIN, height=380, showlegend=False, xaxis_title=None, yaxis_title=None
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("n/a = not yet classified — the bar is the classifier's remaining backlog.")


def _risk_scorecard() -> None:
    st.subheader("Company risk scorecard")
    risk = api_client.companies_risk(limit=SCORECARD_COMPANIES)
    if not risk["items"]:
        st.info("No complaints yet.")
        return

    rows, graph_misses = [], 0
    with st.spinner("Joining knowledge-graph risk scores…"):
        for item in risk["items"]:
            # One graph hop per company. N is small (15) and the page has no
            # other reason to rerun; caching is a deliberate later step.
            try:
                profile = api_client.company_profile(item["company"])
                graph_risk, violations = profile["risk_score"], len(profile["violations"])
            except ApiError:
                graph_risk, violations = None, None
                graph_misses += 1
            rows.append({**item, "graph_risk_score": graph_risk, "violations": violations})

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "company": "Company",
            "total_complaints": st.column_config.NumberColumn("Complaints", format="%d"),
            "avg_urgency": st.column_config.NumberColumn("Avg urgency", format="%.2f"),
            "urgent_count": st.column_config.NumberColumn("Urgent (U4+)", format="%d"),
            "extreme_negative_count": st.column_config.NumberColumn("Extreme neg.", format="%d"),
            "graph_risk_score": st.column_config.NumberColumn("Graph risk", format="%.2f"),
            "violations": st.column_config.NumberColumn("Reg. violations", format="%d"),
        },
    )
    st.caption(
        "Volume and severity from Postgres aggregates; risk score and linked regulation "
        "violations from the Neo4j knowledge graph. Resolution rate lands once enough "
        "resolutions accumulate to make the number meaningful."
    )
    if graph_misses:
        st.caption(f"⚠️ {graph_misses} company(ies) missing from the graph — blank risk columns.")


st.title("📈 Analytics")

try:
    _weekly_trend()
    st.divider()
    breakdown = api_client.products_breakdown()["items"]
    left, right = st.columns(2)
    with left:
        _product_treemap(breakdown)
    with right:
        _urgency_histogram(breakdown)
    st.divider()
    _risk_scorecard()
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(f"Could not load analytics: {exc.detail}")
