"""Tests for the complaint classifier: field mapping, graceful fallback, and
the train/inference prompt-parity invariant.

The parity tests load ``fine_tuning/02_format_training_data.py`` (the training
formatter) via importlib and assert the classifier's prompts still match it
byte-for-byte. The prompts are hand-copied across a module boundary the backend
can't import, so this is the guard that catches drift at CI time rather than as
a silent accuracy regression in production.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from app.schemas.classification import ComplaintClassification
from app.services import classifier as clf
from app.services.classifier import Classifier, build_user_prompt
from app.services.llm_client import LLMResponse, LLMUnavailableError, Provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FORMATTER_PATH = PROJECT_ROOT / "fine_tuning" / "02_format_training_data.py"


def _load_formatter():
    spec = importlib.util.spec_from_file_location("format_training_data", FORMATTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["format_training_data"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeLLMClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls: list = []

    def structured(self, response_model, messages, **kwargs):
        self.calls.append((response_model, messages))
        if self._exc is not None:
            raise self._exc
        return self._response


def _classification(**over) -> ComplaintClassification:
    base = {
        "sentiment": "negative",
        "intent": "dispute_resolution",
        "urgency": 3,
        "key_entities": [],
        "reasoning": "customer disputes a fee on their statement",
    }
    base.update(over)
    return ComplaintClassification(**base)


def _response(cls=None, provider=Provider.OLLAMA, is_fallback=False) -> LLMResponse:
    return LLMResponse(
        data=cls or _classification(),
        provider=provider,
        model="resolveai-sentiment",
        prompt_tokens=12,
        completion_tokens=8,
        latency_ms=321,
        is_fallback=is_fallback,
        raw_json="{}",
    )


class TestClassify:
    def test_success_maps_all_fields(self):
        client = _FakeLLMClient(response=_response(_classification(urgency=4)))
        outcome = Classifier(client=client).classify("I was charged twice")
        assert outcome.succeeded is True
        assert outcome.provider == "ollama"
        assert outcome.model == "resolveai-sentiment"
        assert outcome.prompt_tokens == 12
        assert outcome.completion_tokens == 8
        assert outcome.latency_ms == 321
        assert outcome.is_fallback is False
        assert outcome.classification.urgency == 4

    def test_fallback_when_all_providers_down(self):
        client = _FakeLLMClient(exc=LLMUnavailableError("everything down"))
        outcome = Classifier(client=client).classify("urgent help needed")
        assert outcome.succeeded is False
        assert outcome.provider == "none"
        assert outcome.model == "none"
        assert outcome.prompt_tokens == 0
        assert outcome.classification.sentiment == "negative"
        assert outcome.classification.urgency == 3
        assert outcome.classification.key_entities == []
        assert "review" in outcome.classification.reasoning.lower()

    def test_sends_system_and_user_messages(self):
        client = _FakeLLMClient(response=_response())
        Classifier(client=client).classify("a complaint", product="Mortgage")
        _model, messages = client.calls[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == clf.SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"].startswith("COMPLAINT: a complaint")
        assert "PRODUCT: Mortgage" in messages[1]["content"]


class TestUserPrompt:
    def test_narrative_only_is_stripped(self):
        assert build_user_prompt("  hello  ") == "COMPLAINT: hello"

    def test_full_metadata_in_fixed_order(self):
        out = build_user_prompt("n", product="P", issue="I", company="C")
        assert out == "COMPLAINT: n\nPRODUCT: P\nISSUE: I\nCOMPANY: C"


class TestTrainInferenceParity:
    def test_system_prompt_matches_training_formatter(self):
        fmt = _load_formatter()
        assert clf.SYSTEM_PROMPT == fmt.SYSTEM_PROMPT

    def test_user_prompt_matches_training_formatter(self):
        fmt = _load_formatter()
        from app.models.complaint import Complaint

        c = Complaint(
            narrative="disputed charge",
            product="Credit card",
            issue="Fees",
            company="Chase",
        )
        ours = build_user_prompt(c.narrative, product=c.product, issue=c.issue, company=c.company)
        assert ours == fmt._build_user_prompt(c)
