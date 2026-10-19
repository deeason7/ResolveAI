"""Presentation of classification evidence, independent of the Streamlit runtime."""

from __future__ import annotations

OUTCOME_LABELS = {
    "pending": "Pending classification",
    "succeeded": "Model classification succeeded",
    "unavailable": "Unavailable — needs manual triage",
    "manual": "Manually classified",
    "legacy_unknown": "Legacy labels — provenance unknown",
}


def classification_label(case: dict) -> str:
    outcome = case.get("classification_outcome", "legacy_unknown")
    label = OUTCOME_LABELS.get(outcome, "Classification outcome unknown")
    if case.get("classification_stale"):
        label += " · retained labels are stale"
    return label


def classification_notice(case: dict) -> str:
    outcome = case.get("classification_outcome", "legacy_unknown")
    if outcome == "unavailable":
        if case.get("classification_stale"):
            return (
                "The latest classification attempt was unavailable. Prior labels are preserved "
                "below for context; review them before using them for a response."
            )
        return "Classification was unavailable. Urgency and priority are unknown; manual triage is needed."
    if outcome == "pending":
        return "Classification is pending. Missing urgency does not mean normal or low priority."
    if outcome == "legacy_unknown":
        return (
            "These legacy labels have no recorded provenance. They are not verified model outcomes."
        )
    source = case.get("classification_source") or "unknown source"
    fallback = " (fallback provider)" if case.get("classification_was_fallback") else ""
    return f"Label source: {source}{fallback}."


def triage_filter(selection: str) -> dict:
    if selection == "Needs triage":
        return {"classification_outcome": "unavailable"}
    if selection == "All outcomes":
        return {}
    reverse_labels = {value: key for key, value in OUTCOME_LABELS.items()}
    return {"classification_outcome": reverse_labels[selection]}


def can_generate(case: dict) -> bool:
    """Unavailable or stale labels require review before a new automatic draft."""
    return (
        case.get("classification_outcome") in {"succeeded", "manual", "legacy_unknown"}
        and not case.get("classification_stale", False)
        and case.get("urgency") is not None
    )
