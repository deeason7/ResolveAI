"""Complaint classifier — raw narrative in, validated classification out.

This is the domain layer on top of the transport-only :class:`LLMClient`. It owns
three things the transport must never know about:

  * the exact training-time prompt (system + user), so inference matches training
    and the fine-tuned behavior holds,
  * the :class:`ComplaintClassification` contract, and
  * a deterministic fallback so the pipeline degrades instead of blocking when
    every model provider is down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.schemas.classification import ComplaintClassification
from app.services.llm_client import (
    LLMClient,
    LLMResponse,
    LLMUnavailableError,
    get_llm_client,
)

logger = logging.getLogger(__name__)

# Verbatim from fine_tuning/02_format_training_data.SYSTEM_PROMPT. Inference must
# reproduce the training prompt exactly; drift here silently degrades accuracy.
SYSTEM_PROMPT = (
    "You are a financial complaint classifier. Analyze the complaint "
    "and output a structured JSON classification."
)

OPERATION = "classify"


def build_user_prompt(
    narrative: str,
    product: str | None = None,
    issue: str | None = None,
    company: str | None = None,
) -> str:
    """Render the user turn.

    Mirrors ``fine_tuning/02_format_training_data._build_user_prompt`` verbatim:
    COMPLAINT first, then any present metadata, one field per line, **no blank
    separator**. Drift here breaks train/inference parity and silently degrades
    the fine-tuned model, so this stays byte-for-byte identical to the formatter.
    """
    parts = [f"COMPLAINT: {narrative.strip()}"]
    if product:
        parts.append(f"PRODUCT: {product}")
    if issue:
        parts.append(f"ISSUE: {issue}")
    if company:
        parts.append(f"COMPANY: {company}")
    return "\n".join(parts)


def _fallback_classification() -> ComplaintClassification:
    """Conservative default when every provider is unavailable.

    Mid urgency and an explicit reasoning string so a failed call surfaces for a
    human reviewer instead of being silently buried or wrongly escalated.
    """
    return ComplaintClassification(
        sentiment="negative",
        intent="dispute_resolution",
        urgency=3,
        key_entities=[],
        reasoning=(
            "Automated fallback: classification providers unavailable; flagged for manual review."
        ),
    )


@dataclass
class ClassificationOutcome:
    """Classifier result plus the metadata the LLMOps tracker persists."""

    classification: ComplaintClassification
    provider: str  # "ollama" | "groq" | "none"
    model: str  # concrete model name — the LLMOps cost table is keyed on this
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    is_fallback: bool  # True only when the cloud served after a local failure
    succeeded: bool  # False => deterministic fallback (no model output at all)


class Classifier:
    """Classifies a complaint via the fine-tuned model with cloud fallback."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_llm_client()

    def classify(
        self,
        narrative: str,
        *,
        product: str | None = None,
        issue: str | None = None,
        company: str | None = None,
    ) -> ClassificationOutcome:
        """Return a structured classification; never raises on provider failure."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(narrative, product, issue, company),
            },
        ]
        try:
            resp: LLMResponse[ComplaintClassification] = self.client.structured(
                ComplaintClassification, messages
            )
        except LLMUnavailableError as exc:
            logger.error("classification failed, using deterministic fallback: %s", exc)
            return ClassificationOutcome(
                classification=_fallback_classification(),
                provider="none",
                model="none",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                is_fallback=False,
                succeeded=False,
            )

        return ClassificationOutcome(
            classification=resp.data,
            provider=resp.provider.value,
            model=resp.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
            is_fallback=resp.is_fallback,
            succeeded=True,
        )
