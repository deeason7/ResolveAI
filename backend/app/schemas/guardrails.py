"""
Pydantic contracts for the guardrail engine.

``ToneValidation`` is the LLM-as-judge response contract — it lives here, next
to ``ComplaintClassification``, so every structured-output contract sits in one
layer. ``GuardrailOutcome`` is what the ``GuardrailValidator`` Protocol returns
to the orchestrator; the resolution worker (Day 23-24) dumps ``violations``
straight into the ``Resolution.guardrail_violations`` jsonb column, so the
shape here is also the persistence shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.agent import DraftedResponse

GuardrailLayer = Literal["structural", "content_safety", "regulatory_accuracy", "tone"]


class GuardrailViolation(BaseModel):
    """One specific rule failure. ``message`` doubles as regeneration feedback."""

    layer: GuardrailLayer
    code: str = Field(description="Stable machine-readable rule id, e.g. 'too_short'")
    message: str = Field(description="Actionable description the drafting model can fix from")


class ToneValidation(BaseModel):
    """LLM-as-judge scorecard for a drafted response.

    The deterministic threshold (every score >= 6) governs pass/fail;
    ``overall_pass`` records the judge's own verdict but does not override it.
    """

    empathy_score: int = Field(
        ge=1, le=10, description="How empathetic is the response to the consumer's situation?"
    )
    professionalism_score: int = Field(
        ge=1, le=10, description="How professional and measured is the tone?"
    )
    actionability_score: int = Field(
        ge=1, le=10, description="Are the next steps clear and specific?"
    )
    overall_pass: bool = Field(description="The judge's own holistic verdict")
    feedback: str = Field(
        default="", description="Specific, fixable feedback when any score is below 6"
    )


class JudgeCallMetadata(BaseModel):
    """Cost/latency accounting for one judge call — logged by LLMOps as 'tone_check'."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    is_fallback: bool = False


class GuardrailOutcome(BaseModel):
    """Result of validating one draft.

    ``feedback`` is the prompt-ready rendering of ``violations`` quoted by the
    regeneration turn; the structured ``violations`` list is what persists.
    ``sanitized_draft`` is set when PII redaction changed the response text —
    the orchestrator ships the sanitized copy, never the raw one.
    """

    passed: bool
    feedback: str = ""
    violations: list[GuardrailViolation] = Field(default_factory=list)
    scores: dict[str, int] = Field(
        default_factory=dict, description="Tone scores by dimension, when the judge ran"
    )
    judge_call: JudgeCallMetadata | None = None
    sanitized_draft: DraftedResponse | None = None
