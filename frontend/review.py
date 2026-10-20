"""Keep a displayed saved draft stable across UI reruns and review submissions."""

from copy import deepcopy

SNAPSHOT_KEY = "review_draft_snapshot"


def displayed_draft(state, complaint_id, fetch, *, reload=False):
    cached = state.get(SNAPSHOT_KEY)
    if reload or cached is None or cached.get("complaint_id") != complaint_id:
        cached = deepcopy(fetch(complaint_id))
        state[SNAPSHOT_KEY] = cached
    return deepcopy(cached)


def review_identity(draft):
    version = draft.get("case_version")
    if not draft.get("id") or type(version) is not int or version < 1:
        return None
    return {"resolution_id": draft["id"], "expected_case_version": version}


def review_error(error):
    if error.status_code == 409:
        return (
            "This case changed. Reload the current draft, read it, then submit your decision again."
        )
    return error.detail


def case_status_label(status):
    return {
        "response_approved": "Response approved — delivery and outcome unrecorded",
        "resolved": "Legacy approval — delivery and outcome unknown",
    }.get(status, status)


def approval_notice(case, draft):
    if case.get("status") == "resolved":
        return "This case has a legacy approval; the approved revision, delivery and outcome are unverified."
    if case.get("status") == "response_approved":
        if draft.get("review_action") == "approved":
            return "This saved response is approved. Delivery and outcome are unrecorded."
        return "This case has an approved response. Reload the current draft to inspect the recorded approval."
    return None
