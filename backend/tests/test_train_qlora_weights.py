"""
Tests for the class-weight derivation helpers in fine_tuning/03_train_qlora.py.

These cover the pure-Python parts — sentiment extraction from a chat
record, frequency-to-weight conversion, and JSONL counting. The custom
SFTTrainer subclass and data collator need torch + transformers + trl,
which only live on the Colab side; we don't exercise those here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "fine_tuning" / "03_train_qlora.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_qlora", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_qlora"] = mod
    spec.loader.exec_module(mod)
    return mod


train_qlora = _load_script()


# ---------------------------------------------------------------------------
# _extract_sentiment_from_messages
# ---------------------------------------------------------------------------
class TestExtractSentiment:
    def test_pulls_sentiment_from_assistant_json(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "COMPLAINT: I am angry"},
            {
                "role": "assistant",
                "content": json.dumps({"sentiment": "negative", "intent": "x", "urgency": 3}),
            },
        ]
        assert train_qlora._extract_sentiment_from_messages(messages) == "negative"

    def test_returns_none_when_no_assistant_turn(self):
        messages = [{"role": "user", "content": "hi"}]
        assert train_qlora._extract_sentiment_from_messages(messages) is None

    def test_returns_none_when_assistant_content_isnt_json(self):
        messages = [{"role": "assistant", "content": "not json at all"}]
        assert train_qlora._extract_sentiment_from_messages(messages) is None

    def test_returns_none_when_sentiment_key_missing(self):
        messages = [
            {"role": "assistant", "content": json.dumps({"intent": "fraud_report"})}
        ]
        assert train_qlora._extract_sentiment_from_messages(messages) is None

    def test_returns_none_when_sentiment_isnt_string(self):
        messages = [
            {"role": "assistant", "content": json.dumps({"sentiment": 42})}
        ]
        assert train_qlora._extract_sentiment_from_messages(messages) is None

    def test_handles_empty_message_list(self):
        assert train_qlora._extract_sentiment_from_messages([]) is None
        assert train_qlora._extract_sentiment_from_messages(None) is None


# ---------------------------------------------------------------------------
# _derive_class_weights
# ---------------------------------------------------------------------------
class TestDeriveClassWeights:
    def test_sqrt_inverse_freq_matches_formula(self):
        counts = {"neutral": 100, "negative": 5000, "extreme_negative": 4900}
        weights = train_qlora._derive_class_weights(counts, "sqrt_inverse_freq")
        n_total = sum(counts.values())
        k = len(counts)
        for cls, count in counts.items():
            expected = math.sqrt(n_total / (k * count))
            assert weights[cls] == pytest.approx(expected)

    def test_inverse_freq_matches_formula(self):
        counts = {"a": 10, "b": 90}
        weights = train_qlora._derive_class_weights(counts, "inverse_freq")
        n_total = 100
        k = 2
        assert weights["a"] == pytest.approx(n_total / (k * 10))
        assert weights["b"] == pytest.approx(n_total / (k * 90))

    def test_minority_class_gets_higher_weight(self):
        counts = {"neutral": 200, "negative": 5300, "extreme_negative": 4500}
        w = train_qlora._derive_class_weights(counts, "sqrt_inverse_freq")
        assert w["neutral"] > w["negative"]
        assert w["neutral"] > w["extreme_negative"]

    def test_none_method_returns_uniform_weights(self):
        counts = {"a": 1, "b": 99}
        weights = train_qlora._derive_class_weights(counts, "none")
        assert weights == {"a": 1.0, "b": 1.0}

    def test_manual_scale_multiplies_after_method(self):
        counts = {"a": 50, "b": 50}
        base = train_qlora._derive_class_weights(counts, "sqrt_inverse_freq")
        scaled = train_qlora._derive_class_weights(
            counts, "sqrt_inverse_freq", manual_scale={"a": 2.0, "b": 0.5}
        )
        assert scaled["a"] == pytest.approx(base["a"] * 2.0)
        assert scaled["b"] == pytest.approx(base["b"] * 0.5)

    def test_classes_missing_from_manual_scale_default_to_one(self):
        counts = {"a": 50, "b": 50}
        base = train_qlora._derive_class_weights(counts, "sqrt_inverse_freq")
        scaled = train_qlora._derive_class_weights(
            counts, "sqrt_inverse_freq", manual_scale={"a": 3.0}
        )
        assert scaled["a"] == pytest.approx(base["a"] * 3.0)
        assert scaled["b"] == pytest.approx(base["b"])

    def test_zero_count_class_gets_zero_weight_without_dividing_by_zero(self):
        counts = {"a": 0, "b": 100}
        weights = train_qlora._derive_class_weights(counts, "sqrt_inverse_freq")
        assert weights["a"] == 0.0
        assert weights["b"] > 0

    def test_empty_counts_returns_empty(self):
        assert train_qlora._derive_class_weights({}, "sqrt_inverse_freq") == {}

    def test_unknown_method_falls_back_to_uniform_with_warning(self, caplog):
        counts = {"a": 10, "b": 90}
        with caplog.at_level("WARNING"):
            weights = train_qlora._derive_class_weights(counts, "no_such_method")
        assert weights == {"a": 1.0, "b": 1.0}
        assert any("unknown class_weights.method" in r.message for r in caplog.records)

    def test_realistic_resolveai_distribution(self):
        # Numbers from the actual audit run.
        counts = {"neutral": 206, "negative": 5315, "extreme_negative": 4447}
        w = train_qlora._derive_class_weights(counts, "sqrt_inverse_freq")
        # Neutral should be the largest weight by a wide margin; the two
        # majority classes should be close to 1.0 (~sqrt(N/K/N_class) ≈ 0.8).
        assert w["neutral"] > 3.0
        assert 0.5 < w["negative"] < 1.5
        assert 0.5 < w["extreme_negative"] < 1.5


# ---------------------------------------------------------------------------
# _count_sentiments_in_jsonl
# ---------------------------------------------------------------------------
class TestCountSentimentsInJsonl:
    def test_counts_each_class_correctly(self, tmp_path):
        path = tmp_path / "fake_train.jsonl"
        records = [
            {"messages": [{"role": "assistant", "content": json.dumps({"sentiment": "neutral"})}]},
            {"messages": [{"role": "assistant", "content": json.dumps({"sentiment": "negative"})}]},
            {"messages": [{"role": "assistant", "content": json.dumps({"sentiment": "negative"})}]},
            {
                "messages": [
                    {"role": "assistant", "content": json.dumps({"sentiment": "extreme_negative"})}
                ]
            },
        ]
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        counts = train_qlora._count_sentiments_in_jsonl(path)
        assert counts == {"neutral": 1, "negative": 2, "extreme_negative": 1}

    def test_skips_unparseable_lines(self, tmp_path):
        path = tmp_path / "messy.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {"messages": [{"role": "assistant", "content": json.dumps({"sentiment": "neutral"})}]}
                    ),
                    "not json",
                    "",
                    json.dumps(
                        {"messages": [{"role": "assistant", "content": json.dumps({"sentiment": "neutral"})}]}
                    ),
                ]
            ),
            encoding="utf-8",
        )
        counts = train_qlora._count_sentiments_in_jsonl(path)
        assert counts == {"neutral": 2}
