"""
The resolution agent's orchestration loop.

``ResolutionAgent.run()`` executes a *fixed* pipeline (not an LLM-chooses-its-
tools "autonomous" agent): gather grounding context, draft, validate, regenerate
on failure. A fixed DAG is the right call for a four-tool pipeline served partly
by a small local model — it's cheaper, reproducible, and far easier to guardrail
than letting the model decide which tool to call next.

The three context tools are independent (Qdrant vs Neo4j vs Neo4j), so they run
concurrently with ``asyncio.gather``; one failing degrades grounding rather than
killing the resolution — the draft proceeds with whatever came back, and the
regulatory guardrail still blocks any citation that isn't grounded.

The agent does no database I/O. It returns an :class:`AgentResult`; the
resolution worker (Day 23-24) maps that onto a ``Resolution`` row and persists
the ``LLMLog`` rows — the same consistency-boundary discipline as classification.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.models.complaint import Complaint
from app.schemas.agent import (
    CompanyHistoryInput,
    CompanyHistoryResult,
    DraftedResponse,
    DraftResponseInput,
    LookupRegulationsInput,
    PrecedentResult,
    RegulationResult,
    SearchPrecedentsInput,
)
from app.schemas.classification import ComplaintClassification
from app.services.agent.tools import (
    DraftOutcome,
    check_company_history,
    draft_response,
    lookup_regulations,
    search_precedents,
)
from app.services.graph_store import GraphStore
from app.services.llm_client import LLMClient, LLMUnavailableError
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2  # spec: up to 2 regenerations after the first draft

# Agent-level status. The resolution worker maps these onto Resolution.guardrail_status.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"  # exhausted retries; needs human review
STATUS_UNAVAILABLE = "unavailable"  # no model output at all; needs human review


@dataclass
class GuardrailOutcome:
    """Result of validating one draft. ``feedback`` is fed into regeneration."""

    passed: bool
    feedback: str = ""


class GuardrailValidator(Protocol):
    """The seam the Day 21-22 GuardrailEngine implements.

    Async because the real engine's tone layer is an LLM call; the programmatic
    layers simply don't await anything.
    """

    async def validate(
        self, draft: DraftedResponse, context: DraftResponseInput
    ) -> GuardrailOutcome: ...


class NullGuardrail:
    """Default validator that passes everything — used until the real engine lands."""

    async def validate(
        self, draft: DraftedResponse, context: DraftResponseInput
    ) -> GuardrailOutcome:
        return GuardrailOutcome(passed=True)


@dataclass
class AgentResult:
    """What ``run()`` returns; the worker turns this into DB rows."""

    drafted: DraftedResponse | None
    status: str
    reasoning_steps: list[str] = field(default_factory=list)
    llm_calls: list[DraftOutcome] = field(default_factory=list)
    attempts: int = 0
    guardrail_feedback: str = ""

    @property
    def reasoning_summary(self) -> str:
        """Newline-joined chain-of-thought, for ``Resolution.reasoning_summary``."""
        return "\n".join(self.reasoning_steps)


class ResolutionAgent:
    """Orchestrates the four tools into a guardrail-validated resolution draft."""

    def __init__(
        self,
        complaint: Complaint,
        classification: ComplaintClassification,
        *,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        llm_client: LLMClient | None = None,
        guardrails: GuardrailValidator | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.complaint = complaint
        self.classification = classification
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm_client = llm_client
        # Default to a pass-through until the Day 21-22 engine is wired in.
        self.guardrails: GuardrailValidator = guardrails or NullGuardrail()
        self.max_attempts = max_retries + 1

    async def run(self) -> AgentResult:
        """Execute the full pipeline and return a persistable result."""
        reasoning: list[str] = []
        precedents, regulations, company = await self._gather_context(reasoning)

        draft_input = DraftResponseInput(
            complaint_narrative=self.complaint.narrative,
            classification=self.classification,
            precedents=precedents,
            regulations=regulations,
            company_profile=company,
        )
        return await self._draft_validate_loop(draft_input, reasoning)

    async def _gather_context(
        self, reasoning: list[str]
    ) -> tuple[list[PrecedentResult], list[RegulationResult], CompanyHistoryResult | None]:
        """Run the three grounding tools concurrently; failures degrade to empty."""
        tasks: dict[str, Any] = {
            "precedents": search_precedents(
                SearchPrecedentsInput(
                    complaint_text=self.complaint.narrative, product=self.complaint.product
                ),
                vector_store=self.vector_store,
            )
        }
        # These tools need their key field; skip (don't fabricate) when it's absent.
        if self.complaint.product:
            tasks["regulations"] = lookup_regulations(
                LookupRegulationsInput(product=self.complaint.product, issue=self.complaint.issue),
                graph_store=self.graph_store,
            )
        if self.complaint.company:
            tasks["company"] = check_company_history(
                CompanyHistoryInput(company_name=self.complaint.company),
                graph_store=self.graph_store,
            )

        keys = list(tasks)
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        by_key = dict(zip(keys, results, strict=True))

        precedents = self._unwrap(
            by_key.get("precedents"), default=[], label="search_precedents", reasoning=reasoning
        )
        regulations = self._unwrap(
            by_key.get("regulations"), default=[], label="lookup_regulations", reasoning=reasoning
        )
        company = self._unwrap(
            by_key.get("company"), default=None, label="check_company_history", reasoning=reasoning
        )

        reasoning.append(
            f"context gathered: {len(precedents)} precedents, {len(regulations)} regulations, "
            f"company_profile={'yes' if company else 'no'}"
        )
        return precedents, regulations, company

    @staticmethod
    def _unwrap(value: Any, *, default: Any, label: str, reasoning: list[str]) -> Any:
        """Return a gathered tool result, or ``default`` if it raised / was absent."""
        if isinstance(value, Exception):
            logger.warning("tool %s failed; degrading: %s", label, value)
            reasoning.append(f"{label} failed ({type(value).__name__}); proceeding without it")
            return default
        if value is None:
            return default
        return value

    async def _draft_validate_loop(
        self, draft_input: DraftResponseInput, reasoning: list[str]
    ) -> AgentResult:
        """Draft, validate, and regenerate up to ``max_attempts`` times."""
        llm_calls: list[DraftOutcome] = []
        feedback = ""
        last_draft: DraftedResponse | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                outcome = await draft_response(
                    draft_input,
                    llm_client=self.llm_client,
                    feedback=feedback or None,
                    previous_draft=last_draft.response_text if last_draft else None,
                )
            except LLMUnavailableError as exc:
                logger.error("draft attempt %d: all providers down: %s", attempt, exc)
                reasoning.append(f"attempt {attempt}: LLM unavailable; flagged for human review")
                return AgentResult(
                    drafted=last_draft,
                    status=STATUS_UNAVAILABLE,
                    reasoning_steps=reasoning,
                    llm_calls=llm_calls,
                    attempts=attempt - 1,
                )

            llm_calls.append(outcome)
            last_draft = outcome.drafted
            reasoning.append(
                f"attempt {attempt}: drafted via {outcome.provider} "
                f"(confidence={outcome.drafted.confidence:.2f}, tone={outcome.drafted.tone})"
            )

            guardrail = await self.guardrails.validate(outcome.drafted, draft_input)
            if guardrail.passed:
                reasoning.append(f"attempt {attempt}: guardrails passed")
                return AgentResult(
                    drafted=last_draft,
                    status=STATUS_PASSED,
                    reasoning_steps=reasoning,
                    llm_calls=llm_calls,
                    attempts=attempt,
                )
            feedback = guardrail.feedback
            reasoning.append(f"attempt {attempt}: guardrails failed -> {feedback}")

        reasoning.append("max attempts reached; flagged for human review")
        return AgentResult(
            drafted=last_draft,
            status=STATUS_FAILED,
            reasoning_steps=reasoning,
            llm_calls=llm_calls,
            attempts=self.max_attempts,
            guardrail_feedback=feedback,
        )
