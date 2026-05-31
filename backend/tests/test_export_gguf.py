"""Tests for fine_tuning/05_export_gguf.py — pure-Python helpers only.

The actual merge/convert/quantize stages require torch + an installed
llama.cpp and are exercised at export time on Colab or locally; here we
test the routing, path resolution, and Modelfile rendering — everything
the user can hit before any subprocess fires.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# 05_export_gguf is not importable by `import` because filenames can't start
# with a digit in Python identifiers. Same pattern as test_train_qlora_weights
# and test_evaluate.
_SCRIPT = Path(__file__).resolve().parents[2] / "fine_tuning" / "05_export_gguf.py"
_spec = importlib.util.spec_from_file_location("export_gguf", _SCRIPT)
export_gguf = importlib.util.module_from_spec(_spec)
sys.modules["export_gguf"] = export_gguf
_spec.loader.exec_module(export_gguf)


class TestResolveAdapterDir:
    """Top-level adapter takes precedence; otherwise newest checkpoint wins."""

    def test_returns_top_level_when_adapter_config_present(self, tmp_path: Path):
        (tmp_path / "adapter_config.json").write_text("{}")
        assert export_gguf._resolve_adapter_dir(tmp_path) == tmp_path

    def test_falls_back_to_latest_checkpoint(self, tmp_path: Path):
        # No top-level config; three checkpoint dirs, only the latest has a config
        for step in (100, 200, 300):
            cp = tmp_path / f"checkpoint-{step}"
            cp.mkdir()
        (tmp_path / "checkpoint-300" / "adapter_config.json").write_text("{}")

        assert export_gguf._resolve_adapter_dir(tmp_path) == tmp_path / "checkpoint-300"

    def test_picks_highest_numbered_checkpoint(self, tmp_path: Path):
        # Lexicographic vs numeric sort distinction — "checkpoint-200" < "checkpoint-50"
        # lexicographically but 200 > 50 numerically.
        for step in (50, 100, 200):
            cp = tmp_path / f"checkpoint-{step}"
            cp.mkdir()
            (cp / "adapter_config.json").write_text("{}")
        assert export_gguf._resolve_adapter_dir(tmp_path) == tmp_path / "checkpoint-200"

    def test_returns_input_when_nothing_found(self, tmp_path: Path):
        # No top-level, no checkpoints — caller is expected to detect + error.
        assert export_gguf._resolve_adapter_dir(tmp_path) == tmp_path


class TestFindLlamaCpp:
    def test_explicit_hint_wins(self, tmp_path: Path):
        (tmp_path / "convert_hf_to_gguf.py").write_text("# stub")
        assert export_gguf._find_llama_cpp(str(tmp_path)) == tmp_path

    def test_explicit_hint_ignored_if_invalid(self, tmp_path: Path, monkeypatch):
        # Hint dir exists but has no convert script; env + home dirs also empty.
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        assert export_gguf._find_llama_cpp(str(tmp_path)) is None

    def test_env_fallback(self, tmp_path: Path, monkeypatch):
        env_dir = tmp_path / "env_llama"
        env_dir.mkdir()
        (env_dir / "convert_hf_to_gguf.py").write_text("# stub")
        monkeypatch.setenv("LLAMA_CPP_DIR", str(env_dir))
        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        assert export_gguf._find_llama_cpp(None) == env_dir

    def test_home_code_fallback(self, tmp_path: Path, monkeypatch):
        fake_home = tmp_path / "home"
        clone = fake_home / "code" / "llama.cpp"
        clone.mkdir(parents=True)
        (clone / "convert_hf_to_gguf.py").write_text("# stub")
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setenv("HOME", str(fake_home))
        assert export_gguf._find_llama_cpp(None) == clone

    def test_returns_none_when_not_found(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert export_gguf._find_llama_cpp(None) is None


class TestQuantizeBinary:
    def test_modern_build_bin_path(self, tmp_path: Path):
        bin_path = tmp_path / "build" / "bin" / "llama-quantize"
        bin_path.parent.mkdir(parents=True)
        bin_path.write_text("#!/bin/sh")
        bin_path.chmod(0o755)
        assert export_gguf._quantize_binary(tmp_path) == bin_path

    def test_top_level_llama_quantize(self, tmp_path: Path):
        bin_path = tmp_path / "llama-quantize"
        bin_path.write_text("#!/bin/sh")
        bin_path.chmod(0o755)
        assert export_gguf._quantize_binary(tmp_path) == bin_path

    def test_legacy_quantize_name(self, tmp_path: Path):
        bin_path = tmp_path / "quantize"
        bin_path.write_text("#!/bin/sh")
        bin_path.chmod(0o755)
        assert export_gguf._quantize_binary(tmp_path) == bin_path

    def test_returns_none_when_not_executable(self, tmp_path: Path, monkeypatch):
        # File exists but is not +x — must not be returned.
        bin_path = tmp_path / "llama-quantize"
        bin_path.write_text("#!/bin/sh")
        bin_path.chmod(0o644)
        # Also clear PATH so shutil.which can't find a system one.
        monkeypatch.setenv("PATH", "")
        assert export_gguf._quantize_binary(tmp_path) is None


class TestRenderModelfile:
    def test_from_directive_uses_relative_path(self, tmp_path: Path):
        gguf = tmp_path / "subdir" / "resolveai-sentiment-q4_k_m.gguf"
        rendered = export_gguf._render_modelfile(gguf, "system prompt")
        # FROM must use ./<filename>, not absolute path, so `ollama create -f`
        # can be run from the Modelfile's directory without leaking absolute
        # paths into the Modelfile.
        assert "FROM ./resolveai-sentiment-q4_k_m.gguf" in rendered
        assert str(gguf) not in rendered  # full absolute path should not leak

    def test_system_prompt_embedded(self):
        rendered = export_gguf._render_modelfile(Path("foo.gguf"), "you are a classifier")
        assert 'SYSTEM "you are a classifier"' in rendered

    def test_stop_token_is_chatml_im_end(self):
        rendered = export_gguf._render_modelfile(Path("foo.gguf"), "x")
        assert 'PARAMETER stop "<|im_end|>"' in rendered

    def test_template_uses_chatml_markers(self):
        rendered = export_gguf._render_modelfile(Path("foo.gguf"), "x")
        # The exact ChatML scaffolding the Qwen tokenizer uses.
        assert "<|im_start|>system" in rendered
        assert "{{ .System }}" in rendered
        assert "<|im_start|>user" in rendered
        assert "{{ .Prompt }}" in rendered
        assert "<|im_start|>assistant" in rendered

    def test_training_system_prompt_constant_matches_format_script(self):
        # Locked invariant: TRAINING_SYSTEM_PROMPT in 05_export_gguf must equal
        # SYSTEM_PROMPT in 02_format_training_data — drift breaks inference.
        fmt_script = (
            Path(__file__).resolve().parents[2]
            / "fine_tuning"
            / "02_format_training_data.py"
        )
        text = fmt_script.read_text()
        # Extract the SYSTEM_PROMPT literal robustly: it's the multi-line
        # concatenated string on adjacent lines, so we look for both halves.
        assert "You are a financial complaint classifier." in text
        assert "Analyze the complaint and output a structured JSON classification." in text
        assert "You are a financial complaint classifier." in export_gguf.TRAINING_SYSTEM_PROMPT
        assert (
            "Analyze the complaint and output a structured JSON classification."
            in export_gguf.TRAINING_SYSTEM_PROMPT
        )


class TestBuildPaths:
    def test_output_naming_convention(self, tmp_path: Path):
        import argparse

        args = argparse.Namespace(
            adapter_dir=str(tmp_path / "adapter"),
            output_dir=str(tmp_path / "out"),
            base_model="some/base",
            model_name="resolveai-sentiment",
            llama_cpp_dir=None,
        )
        paths = export_gguf._build_paths(args)
        # Outputs follow {model-name}-{stage} convention so multiple models
        # can coexist in the same output dir.
        assert paths.merged_hf_dir.name == "resolveai-sentiment-merged"
        assert paths.fp16_gguf.name == "resolveai-sentiment-fp16.gguf"
        assert paths.quant_gguf.name == "resolveai-sentiment-q4_k_m.gguf"
        assert paths.modelfile.name == "Modelfile"
        # All under the same output dir.
        assert paths.merged_hf_dir.parent == paths.modelfile.parent

    def test_paths_are_absolute(self, tmp_path: Path, monkeypatch):
        import argparse

        # Use a relative path and verify _build_paths resolves it absolute,
        # which matters because the merge step writes from cwd-independent code.
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(
            adapter_dir="rel/adapter",
            output_dir="rel/out",
            base_model="some/base",
            model_name="foo",
            llama_cpp_dir=None,
        )
        paths = export_gguf._build_paths(args)
        assert paths.adapter_dir.is_absolute()
        assert paths.merged_hf_dir.is_absolute()
        assert paths.quant_gguf.is_absolute()


class TestDryRun:
    def test_dry_run_returns_2_when_adapter_missing(self, tmp_path: Path):
        paths = export_gguf.ExportPaths(
            adapter_dir=tmp_path / "nope",
            merged_hf_dir=tmp_path / "out" / "merged",
            fp16_gguf=tmp_path / "out" / "fp16.gguf",
            quant_gguf=tmp_path / "out" / "q4.gguf",
            modelfile=tmp_path / "out" / "Modelfile",
            llama_cpp_dir=None,
        )
        assert export_gguf._dry_run(paths) == 2

    def test_dry_run_returns_0_when_everything_present(self, tmp_path: Path):
        # Construct a fully valid fake setup
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")

        llama_dir = tmp_path / "llama.cpp"
        (llama_dir / "build" / "bin").mkdir(parents=True)
        (llama_dir / "convert_hf_to_gguf.py").write_text("# stub")
        quant = llama_dir / "build" / "bin" / "llama-quantize"
        quant.write_text("#!/bin/sh")
        quant.chmod(0o755)

        out = tmp_path / "out"
        out.mkdir()

        paths = export_gguf.ExportPaths(
            adapter_dir=adapter,
            merged_hf_dir=out / "merged",
            fp16_gguf=out / "fp16.gguf",
            quant_gguf=out / "q4.gguf",
            modelfile=out / "Modelfile",
            llama_cpp_dir=llama_dir,
        )
        assert export_gguf._dry_run(paths) == 0
