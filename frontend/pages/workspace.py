"""
Workspace — the live control room for the classify -> resolve pipeline.

Every complaint flows: pending -> classified | escalated -> agent_triggered ->
draft_ready | needs_review -> resolved. This page shows how many sit at each
stage right now (the durable status counts) with the transient Redis stream
state layered on top, and lets an operator push a batch of pending complaints
into classification or escalated ones into the resolution agent, then watch
them move with Refresh.
"""

from __future__ import annotations

import streamlit as st

import api_client
from api_client import ApiError
from theme import DEMO_HINT


def _stage_board(board: dict) -> None:
    st.subheader("Pipeline")
    cols = st.columns(5)
    cols[0].metric("Queued", f"{board['pending']:,}", help="Awaiting classification")
    cols[1].metric("Classified", f"{board['classified']:,}", help="Low priority — no action needed")
    cols[2].metric(
        "In resolution",
        f"{board['escalated'] + board['agent_triggered']:,}",
        help=f"escalated {board['escalated']:,} + agent working {board['agent_triggered']:,}",
    )
    cols[3].metric(
        "Needs review",
        f"{board['draft_ready'] + board['needs_review']:,}",
        help=f"draft ready {board['draft_ready']:,} + agent failed {board['needs_review']:,}",
    )
    cols[4].metric("Resolved", f"{board['resolved']:,}", help="Human approved — closed")


def _stream_row(label: str, info: dict) -> None:
    lag = info["lag"]
    cols = st.columns([2, 1, 1])
    cols[0].markdown(f"**{label}**  \n`{info['name']}`")
    cols[1].metric(
        "Waiting",
        f"{lag:,}" if lag is not None else "—",
        help="Enqueued, not yet picked up by a worker",
    )
    cols[2].metric(
        "In flight",
        f"{info['in_flight']:,}",
        help="Claimed by a worker — processing right now",
    )


st.title("⚙️ Workspace")

try:
    board = api_client.workspace_board()
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(f"Could not load the workspace: {exc.detail}")
    st.stop()

head = st.columns([1, 4])
with head[0]:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
st.caption(f"{board['total']:,} complaints in the system — Refresh to re-poll the pipeline.")

_stage_board(board)

st.divider()
st.subheader("Work streams")
_stream_row("Classification", board["classification_stream"])
_stream_row("Resolution", board["resolution_stream"])

st.caption(
    "Enqueued work moves only while a worker is attached — **In flight** turns positive when "
    "one picks a task up. Watch the Pipeline counts above shift as a batch flows through."
)

st.divider()
st.subheader("Add work")
st.caption(
    "Start small — the cloud LLM's free tier rate-limits large bursts, which the worker "
    "absorbs as fail-closed (max-severity) results. Batches of ~10 classify cleanly."
)
demo = api_client.is_demo()
left, right = st.columns(2)

with left:
    st.markdown("**Classify pending complaints**")
    n_cls = st.number_input(
        "How many to classify",
        min_value=1,
        max_value=500,
        value=10,
        step=10,
        key="ws_cls_n",
        label_visibility="collapsed",
    )
    if st.button(
        "Enqueue for classification",
        use_container_width=True,
        disabled=demo or board["pending"] == 0,
        help=DEMO_HINT if demo else None,
    ):
        try:
            result = api_client.enqueue_classification(int(n_cls))
        except ApiError as exc:
            st.error(f"Enqueue failed: {exc.detail}")
        else:
            st.success(f"Enqueued {result['enqueued']} complaint(s) for classification.")
            st.rerun()
    st.caption(f"{board['pending']:,} pending available.")

with right:
    st.markdown("**Resolve escalated complaints**")
    n_res = st.number_input(
        "How many to resolve",
        min_value=1,
        max_value=500,
        value=10,
        step=10,
        key="ws_res_n",
        label_visibility="collapsed",
    )
    if st.button(
        "Enqueue for resolution",
        use_container_width=True,
        disabled=demo or board["escalated"] == 0,
        help=DEMO_HINT if demo else None,
    ):
        try:
            result = api_client.enqueue_resolution_batch(int(n_res))
        except ApiError as exc:
            st.error(f"Enqueue failed: {exc.detail}")
        else:
            st.success(f"Enqueued {result['enqueued']} complaint(s) for resolution.")
            st.rerun()
    st.caption(f"{board['escalated']:,} escalated awaiting the agent.")
