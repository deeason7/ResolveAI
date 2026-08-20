"""Tests for the four-layer guardrail engine.

Layers 1-3 are pure functions, so they are tested directly with crafted text.
The tone layer and the full ``validate`` flow run against a fake judge client
(no LLM), and the integration tests drive the real engine inside the real
``ResolutionAgent`` loop with the tool seams stubbed — proving the engine
satisfies the ``GuardrailValidator`` Protocol in practice, not just in shape.
"""

from __future__ import annotations

import pytest

from app.models.complaint import Complaint
from app.schemas.agent import DraftedResponse, DraftResponseInput, RegulationResult
from app.schemas.classification import ComplaintClassification
from app.schemas.guardrails import ToneValidation
from app.services import guardrails
from app.services.agent import orchestrator as orch
from app.services.agent.orchestrator import STATUS_FAILED, STATUS_PASSED, ResolutionAgent
from app.services.agent.tools import DraftOutcome
from app.services.guardrails import GuardrailEngine
from app.services.llm_client import LLMResponse, LLMUnavailableError, Provider

# Passes every programmatic layer: >200 chars, acknowledgment + next steps,
# no forbidden phrasing, no PII.
GOOD_TEXT = (
    "Thank you for bringing this to our attention — we understand how frustrating a "
    "duplicate mortgage payment charge is, especially when it affects your other "
    "obligations. Based on the details you provided and our review of similar cases, "
    "this appears eligible for a billing-error investigation under the regulations "
    "that apply to your account.\n\n"
    "Next steps: we recommend you submit a written billing-error notice to the company "
    "within 60 days of the statement date, keep copies of both transactions, and "
    "monitor your account for the provisional credit. If the company does not respond "
    "within two billing cycles, you can escalate the matter to the Consumer Financial "
    "Protection Bureau."
)


class FakeJudge:
    """Stand-in for ``LLMClient`` as the tone layer uses it (sync ``structured``)."""

    def __init__(self, tone: ToneValidation | None = None, *, raises: bool = False):
        self.tone = tone or _tone()
        self.raises = raises
        self.calls: list[dict] = []

    def structured(self, response_model, messages, *, max_retries=None, temperature=0.0):
        self.calls.append(
            {"response_model": response_model, "messages": messages, "temperature": temperature}
        )
        if self.raises:
            raise LLMUnavailableError("all providers failed")
        return LLMResponse(
            data=self.tone,
            provider=Provider.GROQ,
            model="llama-3.3-70b-versatile",
            prompt_tokens=400,
            completion_tokens=60,
            latency_ms=700,
            is_fallback=False,
            raw_json="{}",
        )


def _tone(empathy=8, professionalism=9, actionability=8, overall=True, feedback=""):
    return ToneValidation(
        empathy_score=empathy,
        professionalism_score=professionalism,
        actionability_score=actionability,
        overall_pass=overall,
        feedback=feedback,
    )


def _reg(title: str = "Fair Credit Reporting Act (FCRA)") -> RegulationResult:
    return RegulationResult(
        title=title,
        cfr_reference="12 CFR 1022",
        summary="Governs accuracy and dispute handling in consumer credit reporting.",
        key_provisions=["dispute rights"],
        relevance="Applies to 'Mortgage' complaints.",
    )


def _classification() -> ComplaintClassification:
    return ComplaintClassification(
        sentiment="extreme_negative",
        intent="dispute_resolution",
        urgency=5,
        key_entities=[],
        reasoning="Consumer reports a duplicate charge and wants it reversed.",
    )


def _draft(**overrides) -> DraftedResponse:
    base = {
        "response_text": GOOD_TEXT,
        "tone": "empathetic",
        "cited_regulations": ["FCRA"],
        "recommended_actions": ["Submit a written billing-error notice"],
        "confidence": 0.82,
    }
    base.update(overrides)
    return DraftedResponse(**base)


def _context(regs: list[RegulationResult] | None = None) -> DraftResponseInput:
    return DraftResponseInput(
        complaint_narrative="They charged me twice for the same mortgage payment.",
        classification=_classification(),
        regulations=[_reg()] if regs is None else regs,
    )


def _codes(violations) -> list[str]:
    return [v.code for v in violations]


# --- Layer 1: structure ---


class TestStructuralLayer:
    def test_too_short_flagged(self):
        out = guardrails.validate_structure("Too short to ship.")
        assert "too_short" in _codes(out)

    def test_too_long_flagged(self):
        out = guardrails.validate_structure(GOOD_TEXT + " padding" * 500)
        assert "too_long" in _codes(out)

    def test_missing_acknowledgment_flagged(self):
        text = (
            "Your account shows two charges for the same payment date. "
            "Next steps: we recommend disputing the duplicate in writing. " * 4
        )
        out = guardrails.validate_structure(text)
        assert _codes(out) == ["missing_acknowledgment"]

    def test_missing_next_steps_flagged(self):
        text = (
            "Thank you for raising this; we understand the situation with the duplicate "
            "charge on your mortgage account and appreciate your patience while it is "
            "reviewed in detail. " * 3
        )
        out = guardrails.validate_structure(text)
        assert _codes(out) == ["missing_next_steps"]

    def test_good_text_is_clean(self):
        assert guardrails.validate_structure(GOOD_TEXT) == []


# --- Layer 2: content safety ---


class TestContentSafetyLayer:
    @pytest.mark.parametrize(
        ("phrase", "expected_code"),
        [
            ("You should sue the company immediately.", "legal_advice"),
            ("You could file a lawsuit over this.", "legal_advice"),
            ("The company is liable for this error.", "liability_admission"),
            ("We admit fault in this matter.", "liability_admission"),
            ("They were negligent in handling your account.", "liability_admission"),
            ("This happened because of your age.", "discriminatory_language"),
        ],
    )
    def test_forbidden_phrasing_flagged(self, phrase, expected_code):
        out = guardrails.validate_content_safety(f"{GOOD_TEXT} {phrase}")
        assert expected_code in _codes(out)

    @pytest.mark.parametrize(
        "phrase",
        [
            # Qualifiers between verb and noun — the original pattern required
            # them to be adjacent, so every one of these shipped clean.
            "The bank admits full legal liability for the duplicate charge.",
            "We admit complete fault for the error on your account.",
            "The company accepts full responsibility for the delay.",
            "We accept responsibility for every resulting fee.",
            "They assume all liability for the mishandled dispute.",
            "The bank is fully responsible for the incorrect reporting.",
            "We are legally responsible for the damage to your credit.",
        ],
    )
    def test_qualified_admissions_flagged(self, phrase):
        """Regression: 'admits liability' was caught, 'admits full liability' wasn't."""
        out = guardrails.validate_content_safety(f"{GOOD_TEXT} {phrase}")
        assert "liability_admission" in _codes(out)

    @pytest.mark.parametrize(
        "phrase",
        [
            # The broadened pattern must not swallow ordinary process language.
            "It is your responsibility to submit the notice within 60 days.",
            "The representative responsible for your case will follow up.",
            "We take your concerns seriously and are reviewing the account.",
        ],
    )
    def test_ordinary_process_language_not_flagged(self, phrase):
        out = guardrails.validate_content_safety(f"{GOOD_TEXT} {phrase}")
        assert "liability_admission" not in _codes(out)

    def test_violation_message_quotes_the_phrase(self):
        out = guardrails.validate_content_safety("You should sue the company.")
        assert "You should sue" in out[0].message

    def test_clean_text_has_no_violations(self):
        # 'pursue' and 'issue' must not trip the \bsue\b alternation.
        text = GOOD_TEXT + " You may pursue the issue through the dispute process."
        assert guardrails.validate_content_safety(text) == []


class TestPiiRedaction:
    @pytest.mark.parametrize(
        "pii",
        [
            "123-45-6789",  # SSN
            "123456789",  # SSN without separators
            "4111 1111 1111 1111",  # card, spaced
            "4111-1111-1111-1111",  # card, dashed
            "4111111111111111",  # card, bare
            "(555) 123-4567",  # phone
            "555-123-4567",  # phone, dashed
            "123456789012",  # 12-digit account number
        ],
    )
    def test_pii_is_redacted(self, pii):
        text, hits = guardrails.redact_pii(f"Reference: {pii} on file.")
        assert guardrails.REDACTION_TOKEN in text
        assert pii not in text
        assert hits  # at least one pattern reported a hit

    def test_clean_text_untouched(self):
        text = "You may dispute $3,000.00 under 12 CFR 1026 within 60 days."
        out, hits = guardrails.redact_pii(text)
        assert out == text
        assert hits == []


# --- Layer 3: regulatory accuracy ---


class TestRegulatoryAccuracyLayer:
    @pytest.mark.parametrize(
        "cited",
        [
            "Fair Credit Reporting Act (FCRA)",  # exact title
            "Fair Credit Reporting Act",  # full name, no acronym
            "FCRA",  # acronym only
            "fcra",  # case-insensitive
        ],
    )
    def test_grounded_citation_forms_pass(self, cited):
        out = guardrails.validate_regulatory_accuracy([cited], [_reg()])
        assert out == []

    def test_fabricated_citation_flagged_by_name(self):
        out = guardrails.validate_regulatory_accuracy(
            ["FCRA", "Consumer Justice Act of 2031"], [_reg()]
        )
        assert _codes(out) == ["ungrounded_citation"]
        assert "Consumer Justice Act of 2031" in out[0].message
        assert "FCRA" not in out[0].message.split("Only cite")[0]  # grounded one not listed

    def test_any_citation_with_empty_context_is_fabricated(self):
        # The drafting prompt said "no regulations matched" — citing anything is invention.
        out = guardrails.validate_regulatory_accuracy(["FCRA"], [])
        assert _codes(out) == ["ungrounded_citation"]
        assert "none were provided" in out[0].message

    def test_no_citations_is_fine_even_with_no_context(self):
        assert guardrails.validate_regulatory_accuracy([], []) == []

    def test_trivially_short_citation_never_grounds(self):
        out = guardrails.validate_regulatory_accuracy(["Act"], [_reg()])
        assert _codes(out) == ["ungrounded_citation"]


# --- Layer 4: tone (via the engine, exercising the to_thread path) ---


class TestToneLayer:
    async def test_judge_runs_at_low_temperature_with_metadata_captured(self):
        judge = FakeJudge(_tone(8, 9, 8))
        out = await GuardrailEngine(llm_client=judge).validate(_draft(), _context())
        assert out.passed is True
        assert judge.calls[0]["temperature"] == guardrails.JUDGE_TEMPERATURE
        assert judge.calls[0]["response_model"] is ToneValidation
        assert out.scores == {"empathy": 8, "professionalism": 9, "actionability": 8}
        assert out.judge_call is not None
        assert out.judge_call.model == "llama-3.3-70b-versatile"
        assert out.judge_call.prompt_tokens == 400

    async def test_judge_sees_complaint_and_draft(self):
        judge = FakeJudge()
        await GuardrailEngine(llm_client=judge).validate(_draft(), _context())
        user_turn = judge.calls[0]["messages"][1]["content"]
        assert "charged me twice" in user_turn
        assert "Thank you for bringing this" in user_turn

    async def test_low_score_fails_with_judge_feedback(self):
        judge = FakeJudge(_tone(empathy=4, feedback="Open by acknowledging the consumer."))
        out = await GuardrailEngine(llm_client=judge).validate(_draft(), _context())
        assert out.passed is False
        assert _codes(out.violations) == ["tone_below_threshold"]
        assert "empathy 4/10" in out.violations[0].message
        assert "Open by acknowledging the consumer." in out.violations[0].message
        assert out.scores["empathy"] == 4  # scores recorded even on failure

    async def test_score_threshold_governs_not_judge_verdict(self):
        # All scores at the bar but the judge votes no: the deterministic rule wins.
        judge = FakeJudge(_tone(6, 6, 6, overall=False, feedback="just a feeling"))
        out = await GuardrailEngine(llm_client=judge).validate(_draft(), _context())
        assert out.passed is True

    async def test_judge_skipped_when_programmatic_layers_fail(self):
        judge = FakeJudge()
        bad = _draft(response_text=GOOD_TEXT + " You should sue the company.")
        out = await GuardrailEngine(llm_client=judge).validate(bad, _context())
        assert out.passed is False
        assert judge.calls == []  # no money spent judging a draft we're regenerating
        assert out.judge_call is None

    async def test_judge_unavailable_fails_closed(self):
        judge = FakeJudge(raises=True)
        out = await GuardrailEngine(llm_client=judge).validate(_draft(), _context())
        assert out.passed is False
        assert _codes(out.violations) == ["judge_unavailable"]
        assert out.judge_call is None
        assert out.scores == {}


# --- validate(): aggregation, feedback rendering, sanitization ---


class TestEngineOutcome:
    async def test_clean_draft_passes_without_sanitization(self):
        out = await GuardrailEngine(llm_client=FakeJudge()).validate(_draft(), _context())
        assert out.passed is True
        assert out.violations == []
        assert out.feedback == ""
        assert out.sanitized_draft is None

    async def test_pii_redaction_produces_sanitized_copy_not_a_violation(self):
        draft = _draft(response_text=GOOD_TEXT + " Reference: SSN 123-45-6789.")
        out = await GuardrailEngine(llm_client=FakeJudge()).validate(draft, _context())
        assert out.passed is True
        assert out.violations == []
        assert out.sanitized_draft is not None
        assert "123-45-6789" not in out.sanitized_draft.response_text
        assert guardrails.REDACTION_TOKEN in out.sanitized_draft.response_text
        assert "123-45-6789" in draft.response_text  # original draft not mutated

    async def test_programmatic_violations_aggregate_across_layers(self):
        bad = _draft(
            response_text="You should sue the company.",  # short + no ack/steps + legal advice
            cited_regulations=["Consumer Justice Act of 2031"],
        )
        out = await GuardrailEngine(llm_client=FakeJudge()).validate(bad, _context())
        codes = set(_codes(out.violations))
        assert {"too_short", "legal_advice", "ungrounded_citation"} <= codes
        # Feedback renders one actionable line per violation, tagged by layer.
        assert "- [structural]" in out.feedback
        assert "- [content_safety]" in out.feedback
        assert "- [regulatory_accuracy]" in out.feedback


# --- Integration: real engine inside the real ResolutionAgent loop ---


def _stub_context_tools(monkeypatch, regs: list[RegulationResult]):
    async def _precedents(inp, *, vector_store=None):
        return []

    async def _regs(inp, *, graph_store=None):
        return regs

    async def _company(inp, *, graph_store=None):
        return None

    monkeypatch.setattr(orch, "search_precedents", _precedents)
    monkeypatch.setattr(orch, "lookup_regulations", _regs)
    monkeypatch.setattr(orch, "check_company_history", _company)


def _seq_draft(drafts: list[DraftedResponse]):
    """Drafting stub that returns the queued drafts in order, recording feedback."""
    state = {"feedbacks": []}

    async def _f(inp, *, llm_client=None, feedback=None, previous_draft=None):
        state["feedbacks"].append(feedback)
        drafted = drafts.pop(0) if len(drafts) > 1 else drafts[0]
        return DraftOutcome(
            drafted=drafted,
            provider="groq",
            model="m",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1,
            is_fallback=False,
        )

    _f.state = state
    return _f


def _complaint() -> Complaint:
    return Complaint(
        narrative="They charged me twice for the same payment.",
        product="Mortgage",
        company="Wells Fargo",
        issue="Trouble during payment",
    )


class TestAgentWithRealEngine:
    async def test_happy_path_records_scores_and_judge_call(self, monkeypatch):
        _stub_context_tools(monkeypatch, regs=[_reg()])
        monkeypatch.setattr(orch, "draft_response", _seq_draft([_draft()]))
        engine = GuardrailEngine(llm_client=FakeJudge(_tone(8, 9, 8)))

        res = await ResolutionAgent(_complaint(), _classification(), guardrails=engine).run()

        assert res.status == STATUS_PASSED
        assert res.attempts == 1
        assert len(res.judge_calls) == 1
        assert res.guardrail_scores == {"empathy": 8, "professionalism": 9, "actionability": 8}
        assert any("tone scores" in step for step in res.reasoning_steps)

    async def test_bad_draft_regenerates_with_layer_feedback_then_passes(self, monkeypatch):
        _stub_context_tools(monkeypatch, regs=[_reg()])
        bad = _draft(response_text=GOOD_TEXT + " You should sue the company.")
        draft_stub = _seq_draft([bad, _draft()])
        monkeypatch.setattr(orch, "draft_response", draft_stub)
        judge = FakeJudge()
        engine = GuardrailEngine(llm_client=judge)

        res = await ResolutionAgent(_complaint(), _classification(), guardrails=engine).run()

        assert res.status == STATUS_PASSED
        assert res.attempts == 2
        # The retry carried the engine's rendered feedback for the failing layer.
        assert "[content_safety]" in draft_stub.state["feedbacks"][1]
        # Judge ran only for the clean second attempt.
        assert len(judge.calls) == 1
        assert len(res.judge_calls) == 1

    async def test_sanitized_draft_is_what_ships(self, monkeypatch):
        _stub_context_tools(monkeypatch, regs=[_reg()])
        leaky = _draft(response_text=GOOD_TEXT + " Reference: SSN 123-45-6789.")
        monkeypatch.setattr(orch, "draft_response", _seq_draft([leaky]))
        engine = GuardrailEngine(llm_client=FakeJudge())

        res = await ResolutionAgent(_complaint(), _classification(), guardrails=engine).run()

        assert res.status == STATUS_PASSED
        assert res.drafted is not None
        assert "123-45-6789" not in res.drafted.response_text
        assert guardrails.REDACTION_TOKEN in res.drafted.response_text

    async def test_exhausted_retries_surface_structured_violations(self, monkeypatch):
        _stub_context_tools(monkeypatch, regs=[_reg()])
        bad = _draft(response_text=GOOD_TEXT + " The company is liable for this.")
        monkeypatch.setattr(orch, "draft_response", _seq_draft([bad]))
        engine = GuardrailEngine(llm_client=FakeJudge())

        res = await ResolutionAgent(
            _complaint(), _classification(), guardrails=engine, max_retries=1
        ).run()

        assert res.status == STATUS_FAILED
        assert res.attempts == 2
        # The worker persists these into Resolution.guardrail_violations.
        assert "liability_admission" in [v.code for v in res.guardrail_violations]
        assert res.guardrail_feedback
