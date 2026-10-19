"""Classification presentation contracts, independent of a browser or Streamlit."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "classification_presentation",
    Path(__file__).resolve().parents[2] / "frontend/classification.py",
)
presentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(presentation)


def test_all_outcomes_have_distinct_labels_and_legacy_is_not_success():
    labels = [
        presentation.classification_label({"classification_outcome": outcome})
        for outcome in ["pending", "succeeded", "unavailable", "manual", "legacy_unknown"]
    ]
    assert len(set(labels)) == 5
    assert "unknown" in labels[-1]
    assert presentation.classification_label({}) == labels[-1]


def test_unavailable_unknown_values_are_not_assigned_normal_priority():
    case = {"classification_outcome": "unavailable", "urgency": None, "priority_score": None}
    assert "unknown" in presentation.classification_notice(case)
    assert "manual triage" in presentation.classification_notice(case)
    assert not presentation.can_generate(case)


def test_failed_reclassification_retains_explicit_stale_context():
    case = {
        "classification_outcome": "unavailable",
        "classification_stale": True,
        "urgency": 5,
        "priority_score": 0.9,
    }
    assert "stale" in presentation.classification_label(case)
    assert "Prior labels are preserved" in presentation.classification_notice(case)
    assert not presentation.can_generate(case)
    assert case["urgency"] == 5


def test_needs_triage_filter_matches_api_outcome_not_lifecycle():
    assert presentation.triage_filter("Needs triage") == {"classification_outcome": "unavailable"}
    assert presentation.triage_filter("All outcomes") == {}
    assert presentation.triage_filter("Pending classification") == {
        "classification_outcome": "pending"
    }


@pytest.mark.parametrize("outcome", ["succeeded", "manual", "legacy_unknown"])
def test_current_available_labels_allow_drafting(outcome):
    assert presentation.can_generate({"classification_outcome": outcome, "urgency": 2})


def test_successful_fallback_has_explicit_source():
    case = {
        "classification_outcome": "succeeded",
        "classification_source": "groq",
        "classification_was_fallback": True,
    }
    assert presentation.classification_notice(case) == "Label source: groq (fallback provider)."
