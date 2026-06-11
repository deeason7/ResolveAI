"""
Complaint Detail — the full story of one complaint.

Three stacked panels: narrative + classification, the nearest neighbors
from vector search (the precedents a reviewer would actually reach for),
and the resolution draft with the agent's chain of thought and the
guardrail verdict per layer. Reached from the Triage Queue via a one-shot
session_state handoff, or by pasting a complaint UUID directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
from api_client import ApiError
from theme import SENTIMENT_BADGES

HANDOFF_KEY = "detail_complaint_id"
ID_INPUT_KEY = "detail_id_input"

GUARDRAIL_LAYERS = ("structural", "content_safety", "regulatory_accuracy", "tone")
GUARDRAIL_BADGES = {
    "passed": "✅ passed",
    "failed": "❌ failed",
    "escalated": "⚠️ escalated",
    "pending": "⏳ pending",
}


def _layer_icon(layer: str, status: str, violated: set[str]) -> str:
    if layer in violated:
        return "❌"
    if status in ("passed", "failed"):
        # A failed run still validated every layer — the ones without
        # violations genuinely passed their checks.
        return "✅"
    if status == "escalated":
        return "⚠️"  # engine couldn't finish (e.g. judge down) — no verdict
    return "⏳"


def _open_complaint(complaint_id: str) -> None:
    st.session_state[HANDOFF_KEY] = complaint_id
    st.rerun()


def _header(c: dict) -> None:
    st.subheader(f"{c['company'] or 'Unknown company'} — {c['product'] or 'no product'}")
    received = (c["date_received"] or c["created_at"])[:10]
    st.caption(f"`{c['id']}` · received {received} · issue: {c['issue'] or 'n/a'}")

    chips = st.columns(4)
    chips[0].metric("Sentiment", SENTIMENT_BADGES.get(c["sentiment"], "⚪ n/a"))
    chips[1].metric("Intent", c["intent"] or "n/a")
    chips[2].metric("Urgency", f"{c['urgency']} / 5" if c["urgency"] else "n/a")
    chips[3].metric("Status", c["status"])


def _narrative_panel(c: dict) -> None:
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Narrative**")
        # st.text, not st.markdown: narratives are user prose full of $ signs
        # and stray symbols that markdown would happily turn into LaTeX.
        with st.container(height=380, border=True):
            st.text(c["narrative"])
    with right:
        st.markdown("**Classification**")
        with st.container(border=True):
            st.write(
                f"Priority score: `{c['priority_score']:.2f}`"
                if c["priority_score"]
                else "Priority score: n/a"
            )
            st.write(f"State: `{c['state'] or 'n/a'}`")
            st.write(f"Sub-product: {c['sub_product'] or 'n/a'}")
            st.write(f"Sub-issue: {c['sub_issue'] or 'n/a'}")
        if c["company_response"]:
            with st.expander("Historical company response (CFPB record)"):
                st.write(c["company_response"])


def _similar_panel(c: dict) -> None:
    st.subheader("Similar complaints")
    same_product = st.toggle(
        "Same product only",
        value=False,
        help="Restrict the vector search to complaints about the same product",
    )
    product = c["product"] if same_product and c["product"] else None
    try:
        with st.spinner("Searching nearest narratives…"):
            hits = api_client.similar_complaints(c["id"], limit=5, product=product)["items"]
    except ApiError as exc:
        if exc.status_code == 503:
            st.warning("Similarity search is temporarily unavailable.")
            return
        raise

    if not hits:
        st.info("No similar complaints found.")
        return
    for hit in hits:
        with st.container(border=True):
            head, opener = st.columns([5, 1])
            head.markdown(
                f"**{hit['similarity_score']:.2f}** · "
                f"{hit['company'] or 'Unknown'} — {hit['product'] or 'no product'} · "
                f"{SENTIMENT_BADGES.get(hit['sentiment'], '⚪ n/a')} · `{hit['status']}`"
            )
            if opener.button("Open", key=f"open_{hit['id']}", use_container_width=True):
                _open_complaint(hit["id"])
            st.caption(hit["narrative_preview"] + "…")
            if hit["company_response"]:
                st.caption(f"**Resolved as:** {hit['company_response']}")


def _generate_button(c: dict) -> None:
    if st.button("⚙️ Generate resolution", type="primary"):
        try:
            api_client.generate_resolution(c["id"])
        except ApiError as exc:
            if exc.status_code == 409:
                st.warning(exc.detail)
            else:
                st.error(exc.detail)
        else:
            st.success("Resolution queued — the agent is on it. Refresh in a moment.")


def _guardrail_strip(res: dict) -> None:
    violated = {v["layer"] for v in res["guardrail_violations"] or []}
    cols = st.columns(len(GUARDRAIL_LAYERS))
    for col, layer in zip(cols, GUARDRAIL_LAYERS, strict=True):
        col.markdown(f"{_layer_icon(layer, res['guardrail_status'], violated)} {layer}")
    if res["guardrail_notes"]:
        st.caption(res["guardrail_notes"])
    if res["guardrail_violations"]:
        with st.expander(f"Violations ({len(res['guardrail_violations'])})"):
            st.dataframe(
                pd.DataFrame(res["guardrail_violations"])[["layer", "code", "message"]],
                hide_index=True,
                use_container_width=True,
            )


def _reasoning(res: dict) -> None:
    if not res["reasoning_summary"]:
        return
    with st.expander("Agent chain of thought"):
        for step in res["reasoning_summary"].splitlines():
            if step.strip():
                st.markdown(f"- {step}")


def _review_actions(res: dict) -> None:
    approve_col, reject_col, _ = st.columns([1, 1, 2])
    if approve_col.button("✅ Approve", type="primary", use_container_width=True):
        try:
            outcome = api_client.approve_resolution(res["complaint_id"])
        except ApiError as exc:
            st.warning(exc.detail)
        else:
            st.toast(f"Approved v{outcome['version']} — complaint is now resolved.")
            st.rerun()

    with reject_col.popover("❌ Reject…", use_container_width=True):
        with st.form("reject_form", border=False):
            feedback = st.text_area(
                "What's wrong, and what must the next draft fix?",
                placeholder="e.g. The draft promises a refund we cannot commit to…",
            )
            if st.form_submit_button("Reject and regenerate"):
                if len(feedback.strip()) < 10:
                    st.error("Give the agent at least a sentence of feedback.")
                else:
                    try:
                        api_client.reject_resolution(res["complaint_id"], feedback.strip())
                    except ApiError as exc:
                        st.warning(exc.detail)
                    else:
                        st.toast("Rejected — regeneration queued with your feedback.")
                        st.rerun()


def _resolution_panel(c: dict) -> None:
    st.subheader("Resolution")
    try:
        res = api_client.get_resolution(c["id"])
    except ApiError as exc:
        if exc.status_code != 404:
            raise
        st.info("No resolution draft yet.")
        _generate_button(c)
        return

    badge = GUARDRAIL_BADGES.get(res["guardrail_status"], res["guardrail_status"])
    st.markdown(f"**Draft v{res['version']}** · guardrails: {badge}")
    _guardrail_strip(res)
    _reasoning(res)

    st.text_area("Draft response", value=res["draft_text"], height=280, key=f"draft_{res['id']}")
    st.caption(
        "Edits above are for copying out — approval applies the stored draft. "
        "To change the draft itself, reject it with feedback and the agent regenerates."
    )
    _review_actions(res)

    revisions = api_client.list_revisions(c["id"])
    if len(revisions) > 1:
        with st.expander(f"Revision history ({len(revisions)})"):
            st.dataframe(
                pd.DataFrame(revisions)[["version", "guardrail_status", "created_at"]],
                hide_index=True,
                use_container_width=True,
            )


st.title("🔍 Complaint Detail")

handoff = st.session_state.pop(HANDOFF_KEY, None)
if handoff:
    st.session_state[ID_INPUT_KEY] = handoff
complaint_id = st.text_input(
    "Complaint ID", key=ID_INPUT_KEY, placeholder="Paste a complaint UUID…"
)

if not complaint_id:
    st.info("Open a complaint from the Triage Queue, or paste its ID above.")
    st.stop()

try:
    try:
        complaint = api_client.get_complaint(complaint_id.strip())
    except ApiError as exc:
        if exc.status_code in (404, 422):
            st.error("No complaint with that ID.")
            st.stop()
        raise

    _header(complaint)
    _narrative_panel(complaint)
    st.divider()
    _similar_panel(complaint)
    st.divider()
    _resolution_panel(complaint)
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(f"Could not load the complaint: {exc.detail}")
