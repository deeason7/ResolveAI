"""LLMOps tracking — one LLMLog row per inference call.

Tokens and latency come from the LLMClient; cost is derived from a small
per-model pricing table (local Ollama calls are free). The caller owns the DB
session and transaction, so the log row commits atomically with whatever domain
write triggered it (e.g. the complaint update in the classification worker).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.models.llm_log import LLMLog

logger = logging.getLogger(__name__)

# USD per 1,000,000 tokens, as (input_rate, output_rate). Local Ollama is free
# and intentionally absent. Adding a cloud model is a one-line change here.
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    # Current cloud model. Superseded entries stay so historical LLMLog rows
    # (logged under the old model) still price correctly after a migration.
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost for one call. Unknown/local models cost 0.0."""
    rates = _PRICING_PER_MTOK.get(model)
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    cost = prompt_tokens / 1_000_000 * in_rate + completion_tokens / 1_000_000 * out_rate
    return round(cost, 6)


class LLMOpsTracker:
    """Builds and persists LLMLog rows. Session lifecycle stays with the caller."""

    def build_log(
        self,
        *,
        operation: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        was_fallback: bool,
        complaint_id: uuid.UUID | None = None,
    ) -> LLMLog:
        # Local inference has no marginal cost; only meter cloud providers.
        cost = (
            0.0
            if provider == "ollama"
            else estimate_cost_usd(model, prompt_tokens, completion_tokens)
        )
        return LLMLog(
            complaint_id=complaint_id,
            operation=operation,
            model_used=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            was_fallback=was_fallback,
        )

    def record(self, session: Any, **kwargs: Any) -> LLMLog:
        """Stage an LLMLog row on the caller's session (no commit here).

        ``session.add`` is synchronous on both sync and async SQLAlchemy
        sessions, so this stays awaitable-free; the caller commits.
        """
        log = self.build_log(**kwargs)
        session.add(log)
        return log
