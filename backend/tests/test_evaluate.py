"""
Tests for the pure-Python helpers in fine_tuning/04_evaluate.py.

Covers: gold extraction from chat messages, narrative-length extraction,
length-bucket assignment, JSON-parse + tolerance for markdown fences,
per-class P/R/F1 (incl. macro/weighted averages), urgency MAE +
Spearman rank correlation (incl. tie-handling), confusion matrix,
length-bucketed accuracy.

torch/transformers/peft are not imported here — the harness loads
them lazily only when actually generating from a model.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "fine_tuning" / "04_evaluate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = mod
    spec.loader.exec_module(mod)
    return mod


evaluate = _load_script()


# gold + narrative extraction
class TestExtractGold:
    def test_pulls_dict_from_assistant_turn(self):
        msgs = [
            {"role": "user", "content": "COMPLAINT: hi"},
            {"role": "assistant", "content": json.dumps({"sentiment": "neutral", "urgency": 1})},
        ]
        assert evaluate._extract_gold(msgs) == {"sentiment": "neutral", "urgency": 1}

    def test_returns_none_when_no_assistant(self):
        assert evaluate._extract_gold([{"role": "user", "content": "hi"}]) is None

    def test_returns_none_when_content_isnt_json(self):
        msgs = [{"role": "assistant", "content": "not json"}]
        assert evaluate._extract_gold(msgs) is None


class TestExtractNarrative:
    def test_strips_field_prefix(self):
        msgs = [
            {
                "role": "user",
                "content": "COMPLAINT: I am very angry\nPRODUCT: Credit card\nISSUE: x\nCOMPANY: y",
            }
        ]
        assert evaluate._extract_narrative(msgs) == "I am very angry"

    def test_returns_empty_when_no_user_turn(self):
        assert evaluate._extract_narrative([{"role": "assistant", "content": "x"}]) == ""

    def test_handles_missing_complaint_marker(self):
        # Falls back to whole content when no marker
        msgs = [{"role": "user", "content": "raw text"}]
        assert evaluate._extract_narrative(msgs) == "raw text"

# length buckets
class TestAssignLengthBucket:
    def test_short_bucket(self):
        assert evaluate.assign_length_bucket(0) == "short"
        assert evaluate.assign_length_bucket(499) == "short"

    def test_medium_bucket(self):
        assert evaluate.assign_length_bucket(500) == "medium"
        assert evaluate.assign_length_bucket(1499) == "medium"

    def test_long_bucket(self):
        assert evaluate.assign_length_bucket(1500) == "long"
        assert evaluate.assign_length_bucket(50_000) == "long"

# parse_prediction
class TestParsePrediction:
    def test_clean_json(self):
        parsed, err = evaluate.parse_prediction('{"sentiment": "neutral"}')
        assert parsed == {"sentiment": "neutral"}
        assert err == "ok"

    def test_strips_markdown_fence(self):
        raw = '```json\n{"sentiment": "negative"}\n```'
        parsed, err = evaluate.parse_prediction(raw)
        assert parsed == {"sentiment": "negative"}
        assert err == "ok"

    def test_empty_returns_empty_error(self):
        parsed, err = evaluate.parse_prediction("")
        assert parsed is None and err == "empty"

    def test_invalid_json_returns_decode_error(self):
        parsed, err = evaluate.parse_prediction("not json {")
        assert parsed is None
        assert err.startswith("json_decode")

    def test_array_returns_not_an_object(self):
        parsed, err = evaluate.parse_prediction("[1, 2, 3]")
        assert parsed is None and err == "not_an_object"

# per_class_prf1
class TestPerClassPRF1:
    def test_perfect_predictions(self):
        truths = ["a", "b", "a", "b"]
        preds = ["a", "b", "a", "b"]
        out = evaluate.per_class_prf1(truths, preds, ("a", "b"))
        assert out["a"]["f1"] == 1.0
        assert out["b"]["f1"] == 1.0
        assert out["macro_avg"]["f1"] == 1.0
        assert out["weighted_avg"]["f1"] == 1.0

    def test_all_wrong_gives_zero(self):
        truths = ["a", "a"]
        preds = ["b", "b"]
        out = evaluate.per_class_prf1(truths, preds, ("a", "b"))
        assert out["a"]["recall"] == 0.0
        assert out["b"]["precision"] == 0.0

    def test_support_counts_truth_only(self):
        truths = ["a", "a", "b"]
        preds = ["a", "b", "b"]
        out = evaluate.per_class_prf1(truths, preds, ("a", "b"))
        assert out["a"]["support"] == 2
        assert out["b"]["support"] == 1

    def test_macro_treats_classes_equally(self):
        # 99 'a' correct + 1 'b' wrong → high weighted, lower macro
        truths = ["a"] * 99 + ["b"]
        preds = ["a"] * 99 + ["a"]
        out = evaluate.per_class_prf1(truths, preds, ("a", "b"))
        assert out["weighted_avg"]["f1"] > out["macro_avg"]["f1"]
        # b never predicted correctly → its f1 = 0; macro f1 = (a_f1 + 0) / 2
        assert out["macro_avg"]["f1"] < 0.6

    def test_invalid_prediction_sentinel_counts_as_miss(self):
        # The orchestrator substitutes "__INVALID__" for None preds so they
        # show up as misclassifications. Verify that flows through.
        truths = ["a", "b"]
        preds = ["a", "__INVALID__"]
        out = evaluate.per_class_prf1(truths, preds, ("a", "b"))
        assert out["b"]["recall"] == 0.0
        assert out["a"]["recall"] == 1.0

# urgency_metrics + Spearman
class TestUrgencyMetrics:
    def test_perfect_prediction(self):
        truths = [1, 2, 3, 4, 5]
        preds = [1, 2, 3, 4, 5]
        out = evaluate.urgency_metrics(truths, preds)
        assert out["mae"] == 0.0
        assert out["spearman"] == pytest.approx(1.0)

    def test_constant_predictions_yield_none_spearman(self):
        # No variance in preds → Spearman is undefined → returns None
        out = evaluate.urgency_metrics([1, 2, 3, 4, 5], [3, 3, 3, 3, 3])
        assert out["mae"] == pytest.approx(1.2)
        assert out["spearman"] is None

    def test_perfectly_anticorrelated(self):
        out = evaluate.urgency_metrics([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert out["spearman"] == pytest.approx(-1.0)

    def test_empty_inputs(self):
        out = evaluate.urgency_metrics([], [])
        assert out["n"] == 0 and out["mae"] is None


class TestRanks:
    def test_strictly_increasing(self):
        assert evaluate._ranks([10, 20, 30]) == [1.0, 2.0, 3.0]

    def test_ties_get_average_rank(self):
        # 10 occurs twice at positions 0,1 (ranks 1,2) → average 1.5
        assert evaluate._ranks([10, 10, 20]) == [1.5, 1.5, 3.0]

    def test_unordered_input(self):
        ranks = evaluate._ranks([30, 10, 20])
        assert ranks == [3.0, 1.0, 2.0]

# confusion_matrix
class TestConfusionMatrix:
    def test_diagonal_for_perfect_predictions(self):
        truths = ["a", "b", "a"]
        preds = ["a", "b", "a"]
        cm = evaluate.confusion_matrix(truths, preds, ("a", "b"))
        assert cm == [[2, 0], [0, 1]]

    def test_off_diagonal_captures_confusion(self):
        # All 'a' predicted as 'b'
        cm = evaluate.confusion_matrix(["a", "a"], ["b", "b"], ("a", "b"))
        assert cm == [[0, 2], [0, 0]]

    def test_ignores_classes_not_in_vocab(self):
        cm = evaluate.confusion_matrix(["a", "z"], ["a", "a"], ("a", "b"))
        # The 'z' truth doesn't fall into either bucket → dropped
        assert cm == [[1, 0], [0, 0]]

# length_bucketed_accuracy
class TestLengthBucketedAccuracy:
    def _ex(self, sentiment: str, length: int):
        return evaluate.TestExample(messages=[], gold={"sentiment": sentiment}, narrative_length=length)

    def test_per_class_per_bucket_accuracy(self):
        examples = [
            self._ex("neutral", 200),  # short
            self._ex("neutral", 250),  # short
            self._ex("extreme_negative", 1800),  # long
        ]
        preds = ["neutral", "negative", "extreme_negative"]
        out = evaluate.length_bucketed_accuracy(examples, preds)
        # Short bucket: 1/2 neutral correct
        assert out["short"]["per_class"]["neutral"]["n"] == 2
        assert out["short"]["per_class"]["neutral"]["correct"] == 1
        assert out["short"]["per_class"]["neutral"]["accuracy"] == 0.5
        # Long bucket: 1/1 extreme_negative correct
        assert out["long"]["per_class"]["extreme_negative"]["n"] == 1
        assert out["long"]["per_class"]["extreme_negative"]["accuracy"] == 1.0
        # Buckets without that class get accuracy=None (no examples to score)
        assert out["short"]["per_class"]["extreme_negative"]["accuracy"] is None

    def test_overall_accuracy_per_bucket(self):
        examples = [
            self._ex("neutral", 100),
            self._ex("negative", 100),
            self._ex("negative", 1000),
        ]
        preds = ["neutral", "negative", "extreme_negative"]
        out = evaluate.length_bucketed_accuracy(examples, preds)
        assert out["short"]["overall_accuracy"] == 1.0  # 2/2
        assert out["medium"]["overall_accuracy"] == 0.0  # 0/1

# load_test_examples (file IO)
class TestLoadTestExamples:
    def test_loads_well_formed_records(self, tmp_path):
        path = tmp_path / "test.jsonl"
        rec = {
            "messages": [
                {"role": "user", "content": "COMPLAINT: short narrative\nPRODUCT: x"},
                {"role": "assistant", "content": json.dumps({"sentiment": "neutral"})},
            ]
        }
        path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        out = evaluate.load_test_examples(path)
        assert len(out) == 1
        assert out[0].gold == {"sentiment": "neutral"}
        assert out[0].narrative_length == len("short narrative")

    def test_skips_records_without_gold(self, tmp_path, caplog):
        path = tmp_path / "test.jsonl"
        rec_ok = {
            "messages": [
                {"role": "user", "content": "COMPLAINT: ok"},
                {"role": "assistant", "content": json.dumps({"sentiment": "neutral"})},
            ]
        }
        rec_bad = {
            "messages": [{"role": "user", "content": "COMPLAINT: missing assistant"}]
        }
        path.write_text(json.dumps(rec_ok) + "\n" + json.dumps(rec_bad) + "\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            out = evaluate.load_test_examples(path)
        assert len(out) == 1

# Spearman behaviour vs known reference values
class TestSpearmanReference:
    def test_classic_textbook_example(self):
        # Simple example with a known Spearman of about 0.886
        # Pairs: (1, 2), (2, 3), (3, 1), (4, 5), (5, 4)
        rho = evaluate._spearman([1, 2, 3, 4, 5], [2, 3, 1, 5, 4])
        assert rho is not None
        assert math.isclose(rho, 0.6, abs_tol=0.05)