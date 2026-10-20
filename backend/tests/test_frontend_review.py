"""Review snapshot and HTTP contracts without a running browser."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_frontend_session import frontend as frontend
from tests.test_frontend_session import transport

spec = importlib.util.spec_from_file_location(
    "review_presentation", Path(__file__).resolve().parents[2] / "frontend/review.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_rerun_keeps_displayed_text_and_identity_until_explicit_reload():
    state, fetched = {}, []

    def fetch(case):
        fetched.append(case)
        return {
            "id": f"draft-{len(fetched)}",
            "complaint_id": case,
            "case_version": len(fetched),
            "draft_text": f"saved text {len(fetched)}",
        }

    first = review.displayed_draft(state, "case", fetch)
    assert review.displayed_draft(state, "case", fetch) == first
    assert fetched == ["case"]
    first["draft_text"] = "unsaved edit"
    assert review.displayed_draft(state, "case", fetch)["draft_text"] == "saved text 1"
    refreshed = review.displayed_draft(state, "case", fetch, reload=True)
    assert refreshed["draft_text"] == "saved text 2"
    assert review.review_identity(refreshed) == {
        "resolution_id": "draft-2",
        "expected_case_version": 2,
    }
    assert review.displayed_draft(state, "other-case", fetch)["complaint_id"] == "other-case"


@pytest.mark.parametrize(
    "draft", [{}, {"id": "x"}, {"id": "x", "case_version": True}, {"id": "x", "case_version": 0}]
)
def test_missing_or_invalid_identity_disables_review(draft):
    assert review.review_identity(draft) is None


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_api_sends_displayed_identity_and_does_not_retry_conflicts(frontend, monkeypatch, action):
    module, state = frontend
    state[module._USER_KEY] = {"role": "analyst"}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(409, json={"detail": "Case changed"})

    transport(module, monkeypatch, handler)
    args = {"resolution_id": "shown-draft", "expected_case_version": 7}
    with pytest.raises(module.ApiError) as caught:
        if action == "approve":
            module.approve_resolution("case", **args)
        else:
            module.reject_resolution("case", "Please revise this", **args)
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["resolution_id"] == "shown-draft"
    assert body["expected_case_version"] == 7
    assert "Reload" in review.review_error(caught.value)


def test_approval_label_does_not_claim_delivery_and_logout_clears_snapshot(frontend):
    module, state = frontend
    assert "unrecorded" in review.case_status_label("response_approved")
    assert "unknown" in review.case_status_label("resolved")
    state[review.SNAPSHOT_KEY] = {"draft_text": "saved text"}
    module.clear_session()
    assert review.SNAPSHOT_KEY not in state
    assert (
        review.review_error(SimpleNamespace(status_code=503, detail="Unavailable")) == "Unavailable"
    )


def test_cached_old_draft_is_not_presented_as_the_approved_response():
    case = {"status": "response_approved"}
    assert "Reload" in review.approval_notice(case, {"review_action": None})
    assert "This saved response is approved" in review.approval_notice(
        case, {"review_action": "approved"}
    )
    assert "unverified" in review.approval_notice({"status": "resolved"}, {})
