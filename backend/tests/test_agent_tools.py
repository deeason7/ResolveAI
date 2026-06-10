"""Tests for the resolution agent's tools and orchestration loop.

Strategy: every external service is faked, so these run with no Qdrant, Neo4j,
or LLM. Tool tests exercise the real tool functions against fakes; orchestrator
tests monkeypatch the tool functions in the orchestrator's namespace so the loop
logic (gather -> draft -> validate -> regenerate) is tested in isolation.
"""

from __future__ import annotations

from app.models.complaint import Complaint
from app.schemas.agent import (
    CompanyHistoryInput,
    DraftedResponse,
    DraftResponseInput,
    LookupRegulationsInput,
    SearchPrecedentsInput,
)
from app.schemas.classification import ComplaintClassification
from app.schemas.graph import CompanyProfile, ProductBreakdown, Regulation
from app.services.agent import orchestrator as orch
from app.services.agent import tools
from app.services.agent.orchestrator import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    GuardrailOutcome,
    ResolutionAgent,
)
from app.services.agent.tools import DraftOutcome
from app.services.llm_client import LLMResponse, LLMUnavailableError, Provider
from app.services.vector_store import SimilarComplaint

# --- fakes & factories ---


class FakeVectorStore:
    def __init__(self, hits: list[SimilarComplaint]):
        self.hits = hits
        self.last_filter: object = "UNSET"
        self.last_limit: int | None = None

    def search_similar(self, embedding, filters, limit):
        self.last_filter = filters
        self.last_limit = limit
        return self.hits


class FakeGraphStore:
    def __init__(self, *, regs=None, profile=None, raises=False):
        self._regs = regs or []
        self._profile = profile
        self.raises = raises
        self.regs_args: tuple | None = None
        self.profile_arg: str | None = None

    async def get_regulations(self, product, issue=None):
        if self.raises:
            raise RuntimeError("neo4j down")
        self.regs_args = (product, issue)
        return self._regs

    async def get_company_profile(self, name):
        if self.raises:
            raise RuntimeError("neo4j down")
        self.profile_arg = name
        return self._profile


class FakeLLMClient:
    """Stand-in for LLMClient.structured (sync, like the real one)."""

    def __init__(self, draft: DraftedResponse, *, raises: bool = False):
        self.draft = draft
        self.raises = raises
        self.calls: list[list[dict]] = []

    def structured(self, response_model, messages, *, max_retries=None):
        self.calls.append(messages)
        if self.raises:
            raise LLMUnavailableError("all providers failed")
        return LLMResponse(
            data=self.draft,
            provider=Provider.GROQ,
            model="llama-3.3-70b-versatile",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=900,
            is_fallback=False,
            raw_json="{}",
        )


def _draft() -> DraftedResponse:
    return DraftedResponse(
        response_text="Dear consumer, we acknowledge your concern and recommend the following.",
        tone="empathetic",
        cited_regulations=["Fair Credit Reporting Act"],
        recommended_actions=["File a dispute with the bureau"],
        confidence=0.82,
    )


def _classification() -> ComplaintClassification:
    return ComplaintClassification(
        sentiment="extreme_negative",
        intent="dispute_resolution",
        urgency=5,
        key_entities=[],
        reasoning="Consumer reports a duplicate charge and wants it reversed.",
    )


def _complaint(
    *,
    product: str | None = "Mortgage",
    company: str | None = "Wells Fargo",
    issue: str | None = "Trouble during payment",
) -> Complaint:
    return Complaint(
        narrative="They charged me twice for the same payment.",
        product=product,
        company=company,
        issue=issue,
    )


def _draft_input() -> DraftResponseInput:
    return DraftResponseInput(
        complaint_narrative="They charged me twice.",
        classification=_classification(),
    )


# Async stub factories for the orchestrator's tool seams.


def _stub_precedents(value):
    async def _f(inp, *, vector_store=None):
        return value

    return _f


def _stub_regs(value):
    async def _f(inp, *, graph_store=None):
        return value

    return _f


def _stub_company(value):
    async def _f(inp, *, graph_store=None):
        return value

    return _f


def _stub_draft(draft: DraftedResponse):
    state = {"n": 0, "feedbacks": []}

    async def _f(inp, *, llm_client=None, feedback=None, previous_draft=None):
        state["n"] += 1
        state["feedbacks"].append(feedback)
        return DraftOutcome(
            drafted=draft,
            provider="groq",
            model="m",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1,
            is_fallback=False,
        )

    _f.state = state
    return _f


def _stub_context(monkeypatch):
    """Stub all three context tools to benign empty results."""
    monkeypatch.setattr(orch, "search_precedents", _stub_precedents([]))
    monkeypatch.setattr(orch, "lookup_regulations", _stub_regs([]))
    monkeypatch.setattr(orch, "check_company_history", _stub_company(None))


class _Guard:
    """Guardrail stub that returns queued outcomes, then passes."""

    def __init__(self, results: list[GuardrailOutcome]):
        self._results = list(results)
        self.calls = 0

    async def validate(self, draft, context):
        self.calls += 1
        return self._results.pop(0) if self._results else GuardrailOutcome(passed=True)


# --- Tool 1: search_precedents ---


class TestSearchPrecedents:
    async def test_maps_hits_and_tolerates_missing_payload_keys(self, monkeypatch):
        monkeypatch.setattr(tools, "embed_text", lambda text: [0.1] * 384)
        store = FakeVectorStore(
            [
                SimilarComplaint(
                    "id1",
                    0.92,
                    {
                        "narrative_preview": "late fee never refunded",
                        "sentiment": "negative",
                        "company_response": "Closed with explanation",
                    },
                ),
                SimilarComplaint("id2", 0.81, {}),  # payload missing the display keys
            ]
        )

        out = await tools.search_precedents(
            SearchPrecedentsInput(complaint_text="charged twice", product="Mortgage"),
            vector_store=store,
        )

        assert [p.complaint_id for p in out] == ["id1", "id2"]
        assert out[0].narrative_preview == "late fee never refunded"
        assert out[0].company_response == "Closed with explanation"
        assert out[1].narrative_preview == ""  # tolerant default
        assert out[1].company_response is None
        assert store.last_filter == {"product": "Mortgage"}

    async def test_empty_text_short_circuits(self):
        out = await tools.search_precedents(
            SearchPrecedentsInput(complaint_text="   "), vector_store=FakeVectorStore([])
        )
        assert out == []


# --- Tool 2: lookup_regulations ---


class TestLookupRegulations:
    def _reg(self):
        return Regulation(
            id="FCRA",
            title="Fair Credit Reporting Act",
            cfr_reference="15 USC 1681",
            summary="Governs consumer credit reporting accuracy.",
            key_provisions=["dispute rights", "accuracy duties"],
        )

    async def test_maps_and_annotates_relevance_with_issue(self):
        store = FakeGraphStore(regs=[self._reg()])
        out = await tools.lookup_regulations(
            LookupRegulationsInput(product="Credit reporting", issue="Incorrect information"),
            graph_store=store,
        )
        assert out[0].title == "Fair Credit Reporting Act"
        assert "Incorrect information" in out[0].relevance
        assert store.regs_args == ("Credit reporting", "Incorrect information")

    async def test_relevance_without_issue(self):
        store = FakeGraphStore(regs=[self._reg()])
        out = await tools.lookup_regulations(
            LookupRegulationsInput(product="Mortgage"), graph_store=store
        )
        assert "Mortgage" in out[0].relevance


# --- Tool 3: check_company_history ---


class TestCheckCompanyHistory:
    async def test_maps_profile_and_flags_repeat_offender(self):
        profile = CompanyProfile(
            name="Equifax",
            total_complaints=46333,
            risk_score=0.5,
            violations=["FCRA", "GLBA", "UDAAP"],
            product_breakdown=[
                ProductBreakdown(product="Credit reporting", count=46000),
                ProductBreakdown(product="Debt collection", count=333),
            ],
        )
        out = await tools.check_company_history(
            CompanyHistoryInput(company_name="Equifax"), graph_store=FakeGraphStore(profile=profile)
        )
        assert out is not None
        assert out.top_products == ["Credit reporting", "Debt collection"]
        assert out.repeat_offender is True  # 3 violations >= threshold

    async def test_unknown_company_returns_none(self):
        out = await tools.check_company_history(
            CompanyHistoryInput(company_name="Nope"), graph_store=FakeGraphStore(profile=None)
        )
        assert out is None

    async def test_few_violations_is_not_repeat_offender(self):
        profile = CompanyProfile(
            name="Small Bank", total_complaints=5, risk_score=0.1, violations=["FCRA"]
        )
        out = await tools.check_company_history(
            CompanyHistoryInput(company_name="Small Bank"),
            graph_store=FakeGraphStore(profile=profile),
        )
        assert out is not None
        assert out.repeat_offender is False


# --- Tool 4: draft_response ---


class TestDraftResponse:
    async def test_returns_outcome_with_call_metadata(self):
        client = FakeLLMClient(_draft())
        out = await tools.draft_response(_draft_input(), llm_client=client)
        assert out.drafted.response_text.startswith("Dear consumer")
        assert out.provider == "groq"
        assert out.model == "llama-3.3-70b-versatile"
        assert out.prompt_tokens == 100
        assert out.is_fallback is False
        assert len(client.calls[0]) == 2  # system + user, no regeneration turn

    async def test_regeneration_appends_feedback_turn(self):
        client = FakeLLMClient(_draft())
        await tools.draft_response(
            _draft_input(),
            llm_client=client,
            feedback="add concrete next steps",
            previous_draft="old draft",
        )
        messages = client.calls[0]
        assert len(messages) == 3  # system + user + regeneration user turn
        assert "add concrete next steps" in messages[2]["content"]
        assert "old draft" in messages[2]["content"]


# --- Orchestrator ---


class TestResolutionAgent:
    async def test_happy_path_passes_first_try(self, monkeypatch):
        _stub_context(monkeypatch)
        monkeypatch.setattr(orch, "draft_response", _stub_draft(_draft()))
        res = await ResolutionAgent(_complaint(), _classification()).run()
        assert res.status == STATUS_PASSED
        assert res.attempts == 1
        assert len(res.llm_calls) == 1
        assert res.drafted is not None
        assert "guardrails passed" in res.reasoning_summary

    async def test_regenerates_with_feedback_then_passes(self, monkeypatch):
        _stub_context(monkeypatch)
        draft_stub = _stub_draft(_draft())
        monkeypatch.setattr(orch, "draft_response", draft_stub)
        guard = _Guard(
            [
                GuardrailOutcome(passed=False, feedback="add next steps"),
                GuardrailOutcome(passed=True),
            ]
        )
        res = await ResolutionAgent(_complaint(), _classification(), guardrails=guard).run()
        assert res.status == STATUS_PASSED
        assert res.attempts == 2
        assert draft_stub.state["n"] == 2
        # First attempt has no feedback; the retry carries the guardrail feedback.
        assert draft_stub.state["feedbacks"] == [None, "add next steps"]
        assert len(res.llm_calls) == 2

    async def test_exhausts_retries_then_flags_for_review(self, monkeypatch):
        _stub_context(monkeypatch)
        monkeypatch.setattr(orch, "draft_response", _stub_draft(_draft()))
        guard = _Guard([GuardrailOutcome(passed=False, feedback="still bad")] * 5)
        res = await ResolutionAgent(
            _complaint(), _classification(), guardrails=guard, max_retries=2
        ).run()
        assert res.status == STATUS_FAILED
        assert res.attempts == 3  # initial + 2 retries
        assert res.guardrail_feedback == "still bad"
        assert len(res.llm_calls) == 3

    async def test_llm_unavailable_flags_for_human(self, monkeypatch):
        _stub_context(monkeypatch)

        async def _boom(inp, *, llm_client=None, feedback=None, previous_draft=None):
            raise LLMUnavailableError("all providers down")

        monkeypatch.setattr(orch, "draft_response", _boom)
        res = await ResolutionAgent(_complaint(), _classification()).run()
        assert res.status == STATUS_UNAVAILABLE
        assert res.drafted is None
        assert res.llm_calls == []

    async def test_context_tool_failure_degrades_not_crashes(self, monkeypatch):
        monkeypatch.setattr(orch, "search_precedents", _stub_precedents([]))

        async def _regs_boom(inp, *, graph_store=None):
            raise RuntimeError("neo4j down")

        monkeypatch.setattr(orch, "lookup_regulations", _regs_boom)
        monkeypatch.setattr(orch, "check_company_history", _stub_company(None))

        captured = {}

        async def _capture(inp, *, llm_client=None, feedback=None, previous_draft=None):
            captured["regulations"] = inp.regulations
            return DraftOutcome(
                drafted=_draft(),
                provider="groq",
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1,
                is_fallback=False,
            )

        monkeypatch.setattr(orch, "draft_response", _capture)
        res = await ResolutionAgent(_complaint(), _classification()).run()
        assert res.status == STATUS_PASSED
        assert captured["regulations"] == []  # degraded to empty, drafting continued
        assert any("lookup_regulations failed" in s for s in res.reasoning_steps)

    async def test_skips_graph_tools_when_fields_absent(self, monkeypatch):
        called = {"regs": False, "company": False}
        monkeypatch.setattr(orch, "search_precedents", _stub_precedents([]))

        async def _regs(inp, *, graph_store=None):
            called["regs"] = True
            return []

        async def _comp(inp, *, graph_store=None):
            called["company"] = True
            return None

        monkeypatch.setattr(orch, "lookup_regulations", _regs)
        monkeypatch.setattr(orch, "check_company_history", _comp)
        monkeypatch.setattr(orch, "draft_response", _stub_draft(_draft()))

        complaint = _complaint(product=None, company=None, issue=None)
        res = await ResolutionAgent(complaint, _classification()).run()
        assert called["regs"] is False  # no product -> tool skipped
        assert called["company"] is False  # no company -> tool skipped
        assert res.status == STATUS_PASSED
