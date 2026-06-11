"""
Shared look-and-feel constants for the dashboard pages.

Sentiment is rendered in two ways — a hex palette for Plotly traces and an
emoji badge for tables/captions — and every page must agree on both, or the
donut's "negative" ends up a different color than the trend line's. One
module owns the mapping; pages import, never redefine.
"""

from __future__ import annotations

SENTIMENT_COLORS = {
    "neutral": "#2e7d32",
    "negative": "#f9a825",
    "extreme_negative": "#c62828",
    "unclassified": "#9e9e9e",
}

SENTIMENT_BADGES = {
    "neutral": "🟢 neutral",
    "negative": "🟡 negative",
    "extreme_negative": "🔴 extreme",
}

# Plotly default margins drown small charts in whitespace.
PLOT_MARGIN = {"t": 10, "b": 10, "l": 10, "r": 10}

# Tooltip for action buttons disabled in read-only demo sessions.
DEMO_HINT = "Read-only demo — sign in to take actions"
