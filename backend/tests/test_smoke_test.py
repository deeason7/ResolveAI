"""Tests for fine_tuning/06_smoke_test.py — pure-Python helpers only.

The actual Ollama call requires a running instance and is exercised by
the user when running the script post-deploy; here we test prompt
building, test-set sampling, and result formatting — everything that
runs before the network call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load 06_smoke_test.py via importlib because digit-prefixed filenames
# aren't valid Python identifiers. Same pattern as test_export_gguf and
# test_train_qlora_weights.
_SCRIPT = Path(__file__).resolve().parents[2] / "fine_tuning" / "06_smoke_test.py"
_spec = importlib.util.spec_from_file_location("smoke_test", _SCRIPT)
smoke_test = importlib.util.module_from_spec(_spec)
sys.modules["smoke_test"] = smoke_test
_spec.loader.exec_module(smoke_test)


class TestBuildUserPrompt:
    """Prompt format MUST match 02_format_training_data._build_user_prompt."""

    def test_complaint_only(self):
        prompt = smoke_test.build_user_prompt("the charge is wrong")
        assert prompt == "COMPLAINT: the charge is wrong"

    def test_with_product_only(self):
        prompt = smoke_test.build_user_prompt("the charge is wrong", product="Credit card")
        assert prompt == "COMPLAINT: the charge is wrong\nPRODUCT: Credit card"

    def test_full_record(self):
        prompt = smoke_test.build_user_prompt(
            "the charge is wrong",
            product="Credit card",
            issue="Billing disputes",
            company="Chase",
        )
        assert prompt == (
            "COMPLAINT: the charge is wrong\n"
            "PRODUCT: Credit card\n"
            "ISSUE: Billing disputes\n"
            "COMPANY: Chase"
        )

    def test_skips_empty_optional_fields(self):
        # Trainer skips empty optionals — we must do the same so distribution
        # at inference matches training.
        prompt = smoke_test.build_user_prompt("x", product="", issue=None, company="Chase")
        assert prompt == "COMPLAINT: x\nCOMPANY: Chase"

    def test_strips_leading_trailing_whitespace_in_complaint(self):
        prompt = smoke_test.build_user_prompt("  the charge is wrong  ")
        assert prompt == "COMPLAINT: the charge is wrong"

    def test_field_order_is_fixed(self):
        # Order in the trained model's training data is COMPLAINT, then
        # PRODUCT, ISSUE, COMPANY. Tests pin the order so a future "let's
        # alphabetize the fields" refactor breaks at lint time, not at
        # inference where it'd silently degrade accuracy.
        prompt = smoke_test.build_user_prompt("x", company="C", issue="I", product="P")
        lines = prompt.split("\n")
        assert lines[0].startswith("COMPLAINT:")
        assert lines[1].startswith("PRODUCT:")
        assert lines[2].startswith("ISSUE:")
        assert lines[3].startswith("COMPANY:")

    def test_matches_format_training_data_signature(self):
        # Locked invariant: this script's build_user_prompt must produce the
        # same string as fine_tuning/02_format_training_data._build_user_prompt
        # for the same inputs. A drift would mean inference distribution
        # diverges from training distribution.
        fmt_path = (
            Path(__file__).resolve().parents[2] / "fine_tuning" / "02_format_training_data.py"
        )
        text = fmt_path.read_text()
        # The format script builds the same prefix strings. Verify the
        # field labels are byte-identical.
        for label in ("COMPLAINT:", "PRODUCT:", "ISSUE:", "COMPANY:"):
            assert label in text, f"label {label!r} not in 02_format_training_data.py"


class TestLoadTestSample:
    """Reservoir sampling of one record from the test JSONL."""

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def _make_record(self, complaint: str, sentiment: str = "negative") -> dict:
        return {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"COMPLAINT: {complaint}"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "sentiment": sentiment,
                            "intent": "dispute_resolution",
                            "urgency": 3,
                            "key_entities": [],
                            "reasoning": "ok",
                        }
                    ),
                },
            ]
        }

    def test_returns_record_from_nonempty_file(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        self._write_jsonl(path, [self._make_record("c1"), self._make_record("c2")])
        sample = smoke_test.load_test_sample(path, seed=0)
        assert "user_prompt" in sample
        assert "gold" in sample
        assert sample["user_prompt"].startswith("COMPLAINT: ")
        assert sample["gold"]["sentiment"] == "negative"

    def test_raises_when_file_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            smoke_test.load_test_sample(tmp_path / "nope.jsonl", seed=0)

    def test_raises_on_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            smoke_test.load_test_sample(path, seed=0)

    def test_seed_is_deterministic(self, tmp_path: Path):
        # With the same seed and same input file, repeated runs should
        # pick the same record. Critical for reproducible smoke-test runs.
        records = [self._make_record(f"c{i}") for i in range(20)]
        path = tmp_path / "test.jsonl"
        self._write_jsonl(path, records)
        s1 = smoke_test.load_test_sample(path, seed=42)
        s2 = smoke_test.load_test_sample(path, seed=42)
        assert s1["user_prompt"] == s2["user_prompt"]


class TestFormatResult:
    @staticmethod
    def _prediction() -> dict:
        return {
            "sentiment": "negative",
            "intent": "dispute_resolution",
            "urgency": 3,
            "key_entities": [],
            "reasoning": "x",
        }

    def test_includes_latency_in_header(self):
        out = smoke_test.format_result(self._prediction(), gold=None, latency_ms=123.456)
        assert "123 ms" in out  # rounded to int

    def test_emits_valid_pretty_json(self):
        out = smoke_test.format_result(self._prediction(), gold=None, latency_ms=10.0)
        # Strip the header line, parse the rest as JSON
        body_start = out.index("\n") + 1
        # The output is header + json (may have trailing context lines if
        # gold is given; here gold=None so the body is just JSON).
        assert json.loads(out[body_start:])["sentiment"] == "negative"

    def test_no_gold_diff_when_gold_is_none(self):
        out = smoke_test.format_result(self._prediction(), gold=None, latency_ms=10.0)
        assert "vs gold" not in out

    def test_gold_diff_emits_match_markers(self):
        pred = self._prediction()
        out = smoke_test.format_result(pred, gold=pred, latency_ms=10.0)
        assert "vs gold" in out
        assert "sentiment=✓" in out
        assert "intent=✓" in out
        assert "urgency_Δ=0" in out

    def test_gold_diff_marks_mismatch(self):
        pred = self._prediction()
        gold = {**pred, "sentiment": "extreme_negative", "urgency": 5}
        out = smoke_test.format_result(pred, gold=gold, latency_ms=10.0)
        assert "sentiment=✗" in out
        assert "urgency_Δ=2" in out  # 5 - 3 = 2


class TestModuleConstants:
    """Lock the constants that must agree with sibling scripts."""

    def test_system_prompt_matches_format_script(self):
        fmt_path = (
            Path(__file__).resolve().parents[2] / "fine_tuning" / "02_format_training_data.py"
        )
        text = fmt_path.read_text()
        assert "You are a financial complaint classifier." in text
        assert "You are a financial complaint classifier." in smoke_test.TRAINING_SYSTEM_PROMPT
        assert (
            "Analyze the complaint and output a structured JSON classification."
            in smoke_test.TRAINING_SYSTEM_PROMPT
        )

    def test_default_model_name_matches_export_default(self):
        # 05_export_gguf creates the GGUF + Modelfile with --model-name
        # "resolveai-sentiment" by default. The smoke test must point at
        # the same name so ollama can find it.
        export_path = Path(__file__).resolve().parents[2] / "fine_tuning" / "05_export_gguf.py"
        text = export_path.read_text()
        assert 'DEFAULT_MODEL_NAME = "resolveai-sentiment"' in text
        assert smoke_test.DEFAULT_MODEL == "resolveai-sentiment"

    def test_valid_json_threshold_matches_spec(self):
        # The 95% target is from spec line 1524: "Structured output
        # reliability: % valid JSON (target: >95%)". Pinning the constant
        # means tweaking the threshold requires touching the spec context.
        assert smoke_test.DEFAULT_VALID_JSON_THRESHOLD == 0.95
