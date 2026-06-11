"""
LLMOps Observatory — the dashboard about the models themselves.

Every model call the workers make lands in llm_logs with provider, tokens,
cost and latency; this page is that telemetry made visible: what the AI
spends, where calls route (local vs cloud vs fail-closed), how slow each
operation is, how the classifier's output mix moves, and every guardrail
violation the engine has caught.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
import tour
from api_client import ApiError
from theme import PLOT_MARGIN, SENTIMENT_COLORS

WINDOW_DAYS = 90

PROVIDER_COLORS = {
    "ollama": "#2e7d32",  # local — green
    "groq": "#1565c0",  # cloud — blue
    "openai": "#6a1b9a",  # cloud — purple
    "none": "#c62828",  # deterministic fail-closed — red
}

GUARDRAIL_LAYERS = ["all", "structural", "content_safety", "regulatory_accuracy", "tone"]


def _summary_cards(costs: dict, routing: dict) -> None:
    by_provider: dict[str, int] = {}
    for item in routing["items"]:
        by_provider[item["provider"]] = by_provider.get(item["provider"], 0) + item["calls"]
    total = sum(by_provider.values())
    local = by_provider.get("ollama", 0)
    fail_closed = by_provider.get("none", 0)

    cards = st.columns(4)
    cards[0].metric(f"Spend ({WINDOW_DAYS}d)", f"${costs['total_cost_usd']:.4f}")
    cards[1].metric("Model calls", f"{costs['total_calls']:,}")
    cards[2].metric(
        "Local share",
        f"{(local / total * 100):.0f}%" if total else "n/a",
        help="Calls served by the local fine-tuned SLM (ollama) instead of cloud",
    )
    cards[3].metric(
        "Fail-closed events",
        f"{fail_closed:,}",
        help="Calls where no provider answered — the deterministic fallback "
        "flagged maximum severity for human review instead of guessing",
    )


def _cost_chart(costs: dict) -> None:
    st.subheader("Token spend")
    if not costs["points"]:
        st.info("No model calls in this window.")
        return
    df = pd.DataFrame(costs["points"])
    fig = px.bar(
        df,
        x="day",
        y="cost_usd",
        color="provider",
        color_discrete_map=PROVIDER_COLORS,
        barmode="stack",
        hover_data=["calls", "prompt_tokens", "completion_tokens"],
    )
    daily = df.groupby("day", as_index=False)["cost_usd"].sum()
    daily["cumulative"] = daily["cost_usd"].cumsum()
    fig.add_scatter(
        x=daily["day"],
        y=daily["cumulative"],
        mode="lines+markers",
        name="cumulative",
        yaxis="y2",
        line={"color": "#555", "dash": "dot"},
    )
    fig.update_layout(
        margin=PLOT_MARGIN,
        height=340,
        xaxis_title=None,
        yaxis_title="daily $",
        yaxis2={"overlaying": "y", "side": "right", "showgrid": False, "title": "cumulative $"},
    )
    st.plotly_chart(fig, use_container_width=True)


def _routing_pie(routing: dict) -> None:
    st.subheader("Call routing")
    if not routing["items"]:
        st.info("No model calls in this window.")
        return
    df = pd.DataFrame(routing["items"])
    df["slice"] = df.apply(
        lambda r: r["provider"] + (" (fallback)" if r["was_fallback"] else ""), axis=1
    )
    fig = px.pie(
        df,
        names="slice",
        values="calls",
        hole=0.45,
        color="provider",
        color_discrete_map=PROVIDER_COLORS,
    )
    fig.update_layout(margin=PLOT_MARGIN, height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "`none` = deterministic fail-closed path (no provider answered); "
        "`(fallback)` marks cloud calls covering for an unavailable local model."
    )


def _latency_bars(latency: dict) -> None:
    st.subheader("Latency by operation")
    if not latency["items"]:
        st.info("No timed calls in this window.")
        return
    df = pd.DataFrame(latency["items"])
    melted = df.melt(
        id_vars=["operation", "calls"],
        value_vars=["p50_ms", "p95_ms"],
        var_name="percentile",
        value_name="ms",
    )
    fig = px.bar(
        melted,
        x="operation",
        y="ms",
        color="percentile",
        barmode="group",
        log_y=True,
        hover_data=["calls"],
        color_discrete_sequence=["#1565c0", "#c62828"],
    )
    fig.update_layout(margin=PLOT_MARGIN, height=320, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Log scale on purpose — local CPU inference and cloud calls differ by orders of magnitude."
    )


def _drift_area(drift: dict) -> None:
    st.subheader("Classifier output over time")
    if not drift["points"]:
        st.info("No classification events in this window.")
        return
    df = pd.DataFrame(drift["points"])
    fig = px.area(
        df,
        x="day",
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        markers=True,
    )
    fig.update_layout(margin=PLOT_MARGIN, height=320, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bucketed by when the classify call ran (not the complaint's event date) — "
        "a shifting mix here is drift worth investigating."
    )


def _violation_log() -> None:
    st.subheader("Guardrail violation log")
    layer = st.selectbox("Layer", GUARDRAIL_LAYERS)
    data = api_client.llmops_guardrails(layer=None if layer == "all" else layer)
    if not data["items"]:
        st.info("No guardrail violations recorded — drafts have been passing clean.")
        return
    df = pd.DataFrame(data["items"])[
        ["created_at", "layer", "code", "message", "version", "guardrail_status", "complaint_id"]
    ]
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("When", format="YYYY-MM-DD HH:mm"),
            "layer": "Layer",
            "code": "Rule",
            "message": st.column_config.TextColumn("Detail", width="large"),
            "version": st.column_config.NumberColumn("Draft v", format="%d"),
            "guardrail_status": "Run status",
            "complaint_id": "Complaint",
        },
    )
    st.caption(f"{data['total_violations']:,} violation(s) match this filter.")


st.title("🔬 LLMOps Observatory")
tour.render("llmops")

try:
    costs = api_client.llmops_costs(days=WINDOW_DAYS)
    routing = api_client.llmops_routing(days=WINDOW_DAYS)
    latency = api_client.llmops_latency(days=WINDOW_DAYS)

    _summary_cards(costs, routing)
    st.divider()
    _cost_chart(costs)

    left, right = st.columns(2)
    with left:
        _routing_pie(routing)
    with right:
        _latency_bars(latency)

    _drift_area(api_client.llmops_drift(days=WINDOW_DAYS))
    st.divider()
    _violation_log()
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(f"Could not load LLMOps data: {exc.detail}")
