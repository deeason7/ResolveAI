"""Workspace: case progress, durable work health, and bounded operator actions."""

from __future__ import annotations

import streamlit as st

import api_client
from api_client import ApiError
from theme import DEMO_HINT


def _stage_board(board: dict) -> None:
    st.subheader("Pipeline")
    cols = st.columns(5)
    cols[0].metric("Queued", f"{board['pending']:,}", help="Awaiting classification")
    cols[1].metric("Classified", f"{board['classified']:,}", help="Classification recorded")
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
    cols[4].metric(
        "Response approved",
        f"{board['response_approved']:,}",
        help="Saved text approved; delivery and outcome unrecorded",
    )
    st.caption(f"Legacy approvals: {board['resolved']:,}; delivery and outcomes unknown.")
    st.metric(
        "Classification unavailable — needs triage",
        f"{board['classification_unavailable']:,}",
        help="These cases overlap lifecycle counts above; unavailability is not an urgency prediction.",
    )
    if board["classification_unavailable"]:
        st.warning(
            "Review unavailable classifications in the Triage Queue using the Needs triage filter. "
            "Existing labels, if present, are marked stale."
        )


def _job_row(info: dict) -> None:
    cols = st.columns([2, 1, 1, 1, 1])
    cols[0].markdown(f"**{info['kind'].title()}**")
    cols[1].metric("Waiting", f"{info['queued']:,}")
    cols[2].metric("In progress", f"{info['running']:,}")
    cols[3].metric("Failed", f"{info['failed']:,}")
    age = info["oldest_pending_age_seconds"]
    cols[4].metric("Oldest pending", f"{int(age)}s" if age is not None else "—")


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
st.subheader("Processing jobs")
for info in board["jobs"]:
    _job_row(info)
st.caption(
    f"New work is limited to {board['admission_limit']:,} outstanding jobs. "
    "Accepted follow-up work is retained. Failed jobs need staff inspection."
)

staff = (api_client.current_user() or {}).get("role") in {"admin", "analyst"}
if staff:
    try:
        failures = api_client.failed_jobs()
    except ApiError as exc:
        st.error(f"Could not load failed jobs: {exc.detail}")
    else:
        for job in failures["items"]:
            with st.expander(f"{job['kind'].title()} failed · {job['complaint_id']}"):
                st.write(f"Reason: {job['last_error_code'] or 'unknown'}")
                st.caption(
                    f"Attempts: {job['attempts']}/{job['max_attempts']} this retry; "
                    f"{job['total_attempts']} total. Case version {job['case_version']}."
                )
                if st.button(
                    "Retry failed job", key=f"retry_{job['id']}_{job['retry_generation']}"
                ):
                    try:
                        api_client.retry_job(
                            job["id"],
                            expected_attempts=job["attempts"],
                            expected_retry_generation=job["retry_generation"],
                        )
                    except ApiError as exc:
                        st.error(f"Retry failed: {exc.detail}")
                    else:
                        st.rerun()
        if failures["total"] > len(failures["items"]):
            st.caption(f"Showing {len(failures['items'])} of {failures['total']} failed jobs.")

st.divider()
st.subheader("Add work")
st.caption(
    "Start with a small batch. If classification providers fail, cases remain visible as "
    "unavailable and need manual triage. No urgency or response draft is inferred from the failure."
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
        disabled=not staff or board["pending"] == 0,
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
        disabled=not staff or board["escalated"] == 0,
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
