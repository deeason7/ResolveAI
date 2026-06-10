"""
Four-layer guardrail engine for agent-drafted resolutions.

Satisfies the ``GuardrailValidator`` Protocol the orchestrator defines (the
consumer owns the seam; this module implements it). The layers:

    1. structural          — length bounds, acknowledgment + next-steps present
    2. content safety      — forbidden patterns (legal advice, liability
                             admission, discriminatory framing) + PII redaction
    3. regulatory accuracy — every cited regulation must be grounded in the
                             regulations the graph supplied for this complaint
    4. tone                — LLM-as-judge scoring empathy / professionalism /
                             actionability, every score >= TONE_SCORE_THRESHOLD

Layers 1-3 are pure functions and always all run, so a failing draft collects
every programmatic violation in one pass and the regeneration prompt can fix
everything at once. Layer 4 costs a real LLM call, so it only runs once 1-3
are clean — scoring the tone of a draft that is being regenerated anyway is
money spent on a verdict nobody uses.

PII hits do not fail the draft (the policy is redact automatically), so they
produce a ``sanitized_draft`` instead of a violation. If the judge cannot be
reached at all we fail closed: a draft whose tone we could not verify is not
an approved draft, and the normal retry / human-review path takes over.
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.schemas.agent import DraftedResponse, DraftResponseInput, RegulationResult
from app.schemas.guardrails import (
    GuardrailOutcome,
    GuardrailViolation,
    JudgeCallMetadata,
    ToneValidation,
)
from app.services.llm_client import (
    LLMClient,
    LLMResponse,
    LLMUnavailableError,
    get_llm_client,
)

logger = logging.getLogger(__name__)

# --- Layer 1: structure ---

MIN_RESPONSE_CHARS = 200
MAX_RESPONSE_CHARS = 3000

# Cheap keyword heuristics. A false negative costs one regeneration with
# explicit feedback, not a bad response reaching a consumer.
ACKNOWLEDGMENT_KEYWORDS = (
    "thank you",
    "we understand",
    "i understand",
    "we apologize",
    "we're sorry",
    "we are sorry",
    "acknowledge",
    "appreciate",
    "frustrat",  # frustration / frustrating / frustrated
)
NEXT_STEPS_KEYWORDS = ("next steps", "recommend", "suggest", "steps you can take")

# --- Layer 2: content safety ---

_LEGAL_ADVICE = re.compile(
    r"\byou (?:should|could|must|may want to|might) (?:sue\b|file a lawsuit|take legal action)"
    r"|\bfile a lawsuit\b"
    r"|\bsue the\b",
    re.IGNORECASE,
)
_LIABILITY_ADMISSION = re.compile(
    r"\b(?:the company|we|they) (?:is|are|was|were) (?:legally )?"
    r"(?:liable|at fault|guilty|negligent)\b"
    r"|\badmits? (?:fault|liability|guilt)\b"
    r"|\baccepts? (?:full )?liability\b",
    re.IGNORECASE,
)
_DISCRIMINATORY = re.compile(
    r"\bbecause (?:you are|you're|of your) (?:race|gender|sex|religion|age|nationality"
    r"|national origin|disability|marital status)\b"
    r"|\bpeople like you\b",
    re.IGNORECASE,
)

# (code, message template, pattern) — hard violations that force regeneration.
FORBIDDEN_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "legal_advice",
        "Gives legal advice ({phrase!r}). Point to non-legal escalation paths "
        "(written dispute, CFPB complaint) instead.",
        _LEGAL_ADVICE,
    ),
    (
        "liability_admission",
        "Admits fault or liability on the company's behalf ({phrase!r}). Describe "
        "facts and process without conceding liability.",
        _LIABILITY_ADMISSION,
    ),
    (
        "discriminatory_language",
        "Contains discriminatory or biased framing ({phrase!r}). Remove any "
        "reference to protected characteristics.",
        _DISCRIMINATORY,
    ),
)

# PII gets redacted in place, never blocked. Order matters: longer shapes first
# so a card number isn't partially consumed by the SSN pattern.
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("card_number", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")),
    ("phone", re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]?\d{4}\b")),
    ("account_number", re.compile(r"\b\d{8,17}\b")),
)
REDACTION_TOKEN = "[REDACTED]"

# --- Layer 3: regulatory accuracy ---

_PARENTHETICAL_RE = re.compile(r"\(([^)]{2,40})\)")
_MIN_MATCH_CHARS = 4  # 'FCRA' matches; bare 'act' never grounds a citation

# --- Layer 4: tone ---

TONE_SCORE_THRESHOLD = 6  # spec: every score must be >= 6
JUDGE_TEMPERATURE = 0.1

JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality reviewer at a consumer financial protection agency. "
    "You score draft responses to consumer complaints on three 1-10 scales:\n"
    "- empathy_score: does it genuinely acknowledge the consumer's situation and "
    "frustration?\n"
    "- professionalism_score: is it respectful, measured, and free of blame or "
    "condescension?\n"
    "- actionability_score: are the next steps concrete and specific enough to act "
    "on today?\n"
    "A score of 6 is the minimum acceptable bar. Be critical: generic boilerplate "
    "should not score above 5. If any score is below 6, say exactly what to change "
    "in 'feedback'. Respond with JSON only, matching the requested schema."
)


def build_judge_prompt(draft_text: str, complaint_narrative: str) -> str:
    """Render the user turn for the tone-judge call."""
    return (
        f"ORIGINAL COMPLAINT:\n{complaint_narrative.strip()}\n\n"
        f"DRAFT RESPONSE UNDER REVIEW:\n{draft_text.strip()}\n\n"
        "Score the draft response."
    )


# --- layer implementations (module-level: pure and directly testable) ---


def validate_structure(text: str) -> list[GuardrailViolation]:
    """Layer 1: length bounds plus acknowledgment / next-steps presence."""
    violations: list[GuardrailViolation] = []
    n = len(text)
    if n < MIN_RESPONSE_CHARS:
        violations.append(
            GuardrailViolation(
                layer="structural",
                code="too_short",
                message=f"Response is {n} characters; minimum is {MIN_RESPONSE_CHARS}. "
                "Expand the acknowledgment and the resolution detail.",
            )
        )
    elif n > MAX_RESPONSE_CHARS:
        violations.append(
            GuardrailViolation(
                layer="structural",
                code="too_long",
                message=f"Response is {n} characters; maximum is {MAX_RESPONSE_CHARS}. "
                "Tighten it without dropping the acknowledgment or next steps.",
            )
        )
    lowered = text.lower()
    if not any(kw in lowered for kw in ACKNOWLEDGMENT_KEYWORDS):
        violations.append(
            GuardrailViolation(
                layer="structural",
                code="missing_acknowledgment",
                message="Response never acknowledges the consumer's situation; open by "
                "acknowledging the specific issue they raised.",
            )
        )
    if not any(kw in lowered for kw in NEXT_STEPS_KEYWORDS):
        violations.append(
            GuardrailViolation(
                layer="structural",
                code="missing_next_steps",
                message='Response lacks explicit next steps; add a "next steps" section '
                "with concrete recommendations.",
            )
        )
    return violations


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII matches with the redaction token.

    Returns the (possibly unchanged) text and the names of the patterns that
    hit, e.g. ``["ssn x1"]`` — used for logging, never as violations.
    """
    hits: list[str] = []
    for name, pattern in PII_PATTERNS:
        text, count = pattern.subn(REDACTION_TOKEN, text)
        if count:
            hits.append(f"{name} x{count}")
    return text, hits


def validate_content_safety(text: str) -> list[GuardrailViolation]:
    """Layer 2 (violation half): forbidden phrasing that must never ship."""
    violations: list[GuardrailViolation] = []
    for code, template, pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(
                GuardrailViolation(
                    layer="content_safety",
                    code=code,
                    message=template.format(phrase=match.group(0)),
                )
            )
    return violations


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.casefold()).split())


def _title_keys(title: str) -> set[str]:
    """Matchable keys for one canonical title: full form, parentheticals, base name.

    'Fair Credit Reporting Act (FCRA)' -> {'fair credit reporting act fcra',
    'fcra', 'fair credit reporting act'} so a draft may cite the act by full
    name or acronym and still ground.
    """
    keys = {_normalize(title)}
    for inner in _PARENTHETICAL_RE.findall(title):
        keys.add(_normalize(inner))
    base = _normalize(_PARENTHETICAL_RE.sub(" ", title))
    if base:
        keys.add(base)
    return {k for k in keys if k}


def _is_grounded(cited: str, allowed_keys: list[set[str]]) -> bool:
    cited_norm = _normalize(cited)
    if len(cited_norm) < _MIN_MATCH_CHARS:
        return False
    for keys in allowed_keys:
        for key in keys:
            if len(key) >= _MIN_MATCH_CHARS and (cited_norm in key or key in cited_norm):
                return True
    return False


def validate_regulatory_accuracy(
    cited_regulations: list[str], regulations: list[RegulationResult]
) -> list[GuardrailViolation]:
    """Layer 3: every citation must match a regulation the graph supplied.

    The drafting prompt only ever hands the model regulations that APPLIES_TO
    the complaint's product (Phase 3 guards that edge in Cypher), so grounding
    against ``context.regulations`` *is* the existence + applicability check —
    no second Neo4j round-trip needed. An empty context means the model was
    told "no regulations matched", so any citation at all is fabricated.
    """
    if not cited_regulations:
        return []
    allowed = [_title_keys(r.title) for r in regulations]
    ungrounded = [c for c in cited_regulations if not _is_grounded(c, allowed)]
    if not ungrounded:
        return []
    known = ", ".join(r.title for r in regulations) or "none were provided"
    listed = ", ".join(repr(c) for c in ungrounded)
    return [
        GuardrailViolation(
            layer="regulatory_accuracy",
            code="ungrounded_citation",
            message=f"Cited regulation(s) not in the provided context: {listed}. "
            f"Only cite regulations you were given ({known}).",
        )
    ]


def _render_feedback(violations: list[GuardrailViolation]) -> str:
    return "\n".join(f"- [{v.layer}] {v.message}" for v in violations)


class GuardrailEngine:
    """Validates agent drafts through all four layers.

    Stateless between calls; the optional injected ``llm_client`` exists for
    tests, while production resolves the process singleton lazily per call —
    the same convention the agent tools use.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        tone_threshold: int = TONE_SCORE_THRESHOLD,
    ) -> None:
        self._llm_client = llm_client
        self.tone_threshold = tone_threshold

    async def validate(
        self, draft: DraftedResponse, context: DraftResponseInput
    ) -> GuardrailOutcome:
        """Run layers 1-3 (always) and layer 4 (only when 1-3 are clean)."""
        redacted_text, redactions = redact_pii(draft.response_text)
        if redactions:
            logger.warning("guardrails: redacted PII in draft: %s", ", ".join(redactions))

        violations = [
            *validate_structure(redacted_text),
            *validate_content_safety(redacted_text),
            *validate_regulatory_accuracy(draft.cited_regulations, context.regulations),
        ]

        scores: dict[str, int] = {}
        judge_call: JudgeCallMetadata | None = None
        if not violations:
            tone_violations, scores, judge_call = await self._validate_tone(redacted_text, context)
            violations.extend(tone_violations)

        sanitized = (
            draft.model_copy(update={"response_text": redacted_text}) if redactions else None
        )
        return GuardrailOutcome(
            passed=not violations,
            feedback=_render_feedback(violations),
            violations=violations,
            scores=scores,
            judge_call=judge_call,
            sanitized_draft=sanitized,
        )

    async def _validate_tone(
        self, draft_text: str, context: DraftResponseInput
    ) -> tuple[list[GuardrailViolation], dict[str, int], JudgeCallMetadata | None]:
        """Layer 4: LLM-as-judge. Fails closed when the judge is unreachable."""
        client = self._llm_client or get_llm_client()
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_judge_prompt(draft_text, context.complaint_narrative),
            },
        ]
        try:
            # Sync client, same trampoline as the drafting tool.
            resp: LLMResponse[ToneValidation] = await asyncio.to_thread(
                client.structured, ToneValidation, messages, temperature=JUDGE_TEMPERATURE
            )
        except LLMUnavailableError as exc:
            logger.error("guardrails: tone judge unavailable: %s", exc)
            violation = GuardrailViolation(
                layer="tone",
                code="judge_unavailable",
                message="Tone validation could not run (judge LLM unavailable); "
                "the draft is unverified.",
            )
            return [violation], {}, None

        tone = resp.data
        scores = {
            "empathy": tone.empathy_score,
            "professionalism": tone.professionalism_score,
            "actionability": tone.actionability_score,
        }
        judge_call = JudgeCallMetadata(
            provider=resp.provider.value,
            model=resp.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
            is_fallback=resp.is_fallback,
        )

        low = {name: value for name, value in scores.items() if value < self.tone_threshold}
        if not low:
            # The deterministic threshold governs; the judge's own overall_pass
            # is recorded in the scores' company but never overrides it.
            return [], scores, judge_call

        detail = ", ".join(f"{name} {value}/10" for name, value in low.items())
        message = f"Tone scores below the minimum of {self.tone_threshold}: {detail}."
        if tone.feedback:
            message += f" Judge feedback: {tone.feedback}"
        violation = GuardrailViolation(layer="tone", code="tone_below_threshold", message=message)
        return [violation], scores, judge_call
