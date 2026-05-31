"""
Export the fine-tuned LoRA adapter as a GGUF model for Ollama.

Pipeline:
    1.  Load base Qwen2.5-3B-Instruct in fp16
    2.  Load the LoRA adapter and merge it into the base via
        ``peft.PeftModel.merge_and_unload()``
    3.  Save the merged HF model to a temp directory
    4.  Shell out to llama.cpp's ``convert_hf_to_gguf.py`` → fp16 GGUF
    5.  Shell out to llama.cpp's ``llama-quantize`` → Q4_K_M GGUF
    6.  Render the Ollama ``Modelfile`` pointing at the Q4_K_M GGUF
    7.  Delete the fp16 GGUF intermediate (we keep only Q4_K_M for Ollama)

Why this shape:
    Stages 4-5 live in llama.cpp because that's the canonical path
    Ollama itself uses — going through llama.cpp guarantees the GGUF
    will load in Ollama without surprises. ``optimum`` / pure-python
    GGUF emitters exist but lag llama.cpp on new architectures.

Why fp16 → quantize as two steps instead of one:
    ``convert_hf_to_gguf.py`` only emits fp16 or fp32 by design; the
    quantization to Q4_K_M is a separate llama.cpp binary that reads
    the fp16 GGUF and writes the quantized one. Keeping them split
    also lets us inspect the fp16 with a fp16-only tool if we need to
    debug the merge.

Usage:
    # Dry-run — validate inputs without doing the actual conversion
    python fine_tuning/05_export_gguf.py --dry-run

    # Real export (after training has produced the adapter)
    python fine_tuning/05_export_gguf.py \\
        --adapter-dir fine_tuning/models/resolveai-sentiment-lora \\
        --llama-cpp-dir ~/code/llama.cpp \\
        --output-dir fine_tuning/models

Exit codes:
    0 — finished cleanly
    2 — fatal misconfiguration (missing adapter, missing llama.cpp, etc.)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_gguf")

SCRIPT_DIR = Path(__file__).resolve().parent

# Defaults match the Colab notebook's checkpoint location after `cp -R` to
# the local repo. Adjust via CLI flags if you keep them elsewhere.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_DIR = SCRIPT_DIR / "models" / "resolveai-sentiment-lora"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models"
DEFAULT_MODEL_NAME = "resolveai-sentiment"
DEFAULT_QUANT = "Q4_K_M"

# The system prompt used at training-time. Must match 02_format_training_data
# SYSTEM_PROMPT verbatim — Ollama prepends this on every inference call, so a
# drift here would mean the deployed model sees a different system message
# than it was trained on.
TRAINING_SYSTEM_PROMPT = (
    "You are a financial complaint classifier. "
    "Analyze the complaint and output a structured JSON classification."
)

# Qwen2.5's ChatML template — same delimiters used by the tokenizer at
# training time. Ollama substitutes {{ .System }} and {{ .Prompt }} at
# inference time.
QWEN_CHATML_TEMPLATE = """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""


@dataclass
class ExportPaths:
    """Resolved absolute paths for every stage of the pipeline.

    Computed once at startup so every stage references the same files
    even if cwd changes (e.g., subprocess.run inside the llama.cpp dir).
    """

    adapter_dir: Path
    merged_hf_dir: Path
    fp16_gguf: Path
    quant_gguf: Path
    modelfile: Path
    llama_cpp_dir: Path | None


def _resolve_adapter_dir(adapter_dir: Path) -> Path:
    """Return the directory that actually contains ``adapter_config.json``.

    The Colab notebook may save the final adapter at the top level of
    the output dir (after ``load_best_model_at_end`` restores it), but
    intermediate ``checkpoint-N/`` subdirs also contain a full adapter.
    If the top level has no config, fall back to the latest checkpoint.
    Mirrors the same fallback that 04_evaluate.py uses.
    """
    if (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir

    checkpoints = sorted(
        (p for p in adapter_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints and (checkpoints[-1] / "adapter_config.json").is_file():
        log.info("top-level adapter missing; using %s", checkpoints[-1].name)
        return checkpoints[-1]

    return adapter_dir  # let the caller error with a clear message


def _find_llama_cpp(user_hint: str | None) -> Path | None:
    """Locate a llama.cpp checkout.

    Search order:
        1. ``--llama-cpp-dir`` CLI flag (or LLAMA_CPP_DIR env)
        2. ``~/code/llama.cpp``
        3. ``~/llama.cpp``
        4. ``~/git/llama.cpp``

    Validates by checking for the conversion script at the expected path.
    """
    candidates: list[Path] = []
    if user_hint:
        candidates.append(Path(user_hint).expanduser())
    if (env := os.environ.get("LLAMA_CPP_DIR")):
        candidates.append(Path(env).expanduser())
    home = Path.home()
    candidates += [home / "code" / "llama.cpp", home / "llama.cpp", home / "git" / "llama.cpp"]

    for path in candidates:
        if (path / "convert_hf_to_gguf.py").is_file():
            return path
    return None


def _quantize_binary(llama_cpp_dir: Path) -> Path | None:
    """Find the llama-quantize binary in a llama.cpp checkout.

    The binary moved across llama.cpp versions:
        - Old (pre-2024-09): ``./quantize``
        - Newer (~2024-09+): ``./build/bin/llama-quantize`` or ``./llama-quantize``
        - When installed via brew: ``llama-quantize`` on PATH
    """
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",  # legacy
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c

    # Fall back to PATH (brew install llama.cpp lands here).
    path_bin = shutil.which("llama-quantize")
    if path_bin:
        return Path(path_bin)

    return None


def _render_modelfile(quant_gguf_path: Path, system_prompt: str) -> str:
    """Render the Ollama Modelfile body.

    The ``FROM`` directive uses a path relative to the Modelfile's
    location so ``ollama create -f Modelfile`` works regardless of cwd.

    Parameter choices:
        - temperature=0.1 → near-greedy. Classification doesn't need
          creative sampling and high temperature breaks JSON validity.
        - top_p=0.9 → tightens the tail to suppress rare-token JSON errors.
        - num_predict=512 → caps assistant output. JSON outputs average
          ~150 tokens; 512 leaves headroom for long reasoning fields.
        - stop="<|im_end|>" → matches the ChatML end-of-turn token the
          model was trained to emit.
    """
    return f"""FROM ./{quant_gguf_path.name}

TEMPLATE \"\"\"{QWEN_CHATML_TEMPLATE}\"\"\"

SYSTEM \"{system_prompt}\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 512
PARAMETER stop \"<|im_end|>\"
"""


def _dry_run(paths: ExportPaths) -> int:
    """Validate paths without loading torch or running subprocesses."""
    ok = True

    actual_adapter = _resolve_adapter_dir(paths.adapter_dir)
    if (actual_adapter / "adapter_config.json").is_file():
        log.info("adapter: %s ✓", actual_adapter)
    else:
        log.critical(
            "missing adapter_config.json under %s (checked top-level + checkpoint-*)",
            paths.adapter_dir,
        )
        ok = False

    if paths.llama_cpp_dir and (paths.llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
        log.info("llama.cpp: %s ✓", paths.llama_cpp_dir)
        quant = _quantize_binary(paths.llama_cpp_dir)
        if quant:
            log.info("llama-quantize: %s ✓", quant)
        else:
            log.critical(
                "llama-quantize binary not found. Build it with `make llama-quantize` "
                "inside the llama.cpp checkout, or `brew install llama.cpp`.",
            )
            ok = False
    else:
        log.critical(
            "llama.cpp not found. Clone https://github.com/ggerganov/llama.cpp and "
            "either pass --llama-cpp-dir or set LLAMA_CPP_DIR.",
        )
        ok = False

    out = paths.modelfile.parent
    if out.is_dir() and os.access(out, os.W_OK):
        log.info("output dir: %s ✓ (writable)", out)
    else:
        try:
            out.mkdir(parents=True, exist_ok=True)
            log.info("output dir: %s ✓ (created)", out)
        except OSError as exc:
            log.critical("output dir %s not writable: %s", out, exc)
            ok = False

    # Render the Modelfile body so the user can eyeball it without committing
    # to a real export.
    log.info("Modelfile preview:\n%s", _render_modelfile(paths.quant_gguf, TRAINING_SYSTEM_PROMPT))
    return 0 if ok else 2


def _merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    """Load base + adapter, merge into base weights, save as HF format.

    Lazy-imports torch/transformers/peft so --dry-run stays stdlib-only.
    Merges in fp16 — higher precision than the 4-bit training-time base,
    necessary because we'll later quantize to Q4_K_M and any extra
    precision lost in the merge can't be recovered downstream.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading base model %s in fp16...", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # merge on CPU — uses ~6GB RAM but doesn't require a GPU
        low_cpu_mem_usage=True,
    )

    log.info("loading adapter from %s and merging...", adapter_dir)
    peft_model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float16)
    merged = peft_model.merge_and_unload()

    log.info("saving merged model to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    # Persist the tokenizer alongside so convert_hf_to_gguf can find it.
    # The adapter dir already has the tokenizer files (saved by SFTTrainer),
    # but copying them next to the merged weights is what HF tooling expects.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)


def _convert_to_fp16_gguf(
    merged_hf_dir: Path, fp16_gguf: Path, llama_cpp_dir: Path
) -> None:
    """Run llama.cpp's convert_hf_to_gguf.py to emit a fp16 GGUF."""
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    cmd = [
        sys.executable,
        str(script),
        str(merged_hf_dir),
        "--outfile",
        str(fp16_gguf),
        "--outtype",
        "f16",
    ]
    log.info("converting to fp16 GGUF: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _quantize_to_q4_k_m(fp16_gguf: Path, quant_gguf: Path, quant_bin: Path) -> None:
    """Quantize fp16 GGUF → Q4_K_M GGUF.

    Q4_K_M is the standard balance for 3B-class models: ~2.0 GB on disk,
    almost-imperceptible quality loss vs fp16. Other options for the curious:
        - Q5_K_M: ~2.4 GB, marginally better quality
        - Q3_K_M: ~1.5 GB, visible quality loss
        - Q8_0:   ~3.3 GB, near-lossless but loses the point of quantizing
    """
    cmd = [str(quant_bin), str(fp16_gguf), str(quant_gguf), DEFAULT_QUANT]
    log.info("quantizing to %s: %s", DEFAULT_QUANT, " ".join(cmd))
    subprocess.run(cmd, check=True)


def _write_modelfile(modelfile_path: Path, quant_gguf_path: Path) -> None:
    content = _render_modelfile(quant_gguf_path, TRAINING_SYSTEM_PROMPT)
    modelfile_path.write_text(content, encoding="utf-8")
    log.info("wrote Modelfile → %s", modelfile_path)


def _export(paths: ExportPaths, base_model: str, keep_fp16: bool) -> int:
    """Run the full merge → convert → quantize → Modelfile pipeline."""
    actual_adapter = _resolve_adapter_dir(paths.adapter_dir)
    if not (actual_adapter / "adapter_config.json").is_file():
        log.critical("no adapter_config.json under %s", paths.adapter_dir)
        return 2
    if paths.llama_cpp_dir is None:
        log.critical("llama.cpp not found — run with --dry-run for diagnostic hints.")
        return 2
    quant_bin = _quantize_binary(paths.llama_cpp_dir)
    if quant_bin is None:
        log.critical("llama-quantize binary not found in %s", paths.llama_cpp_dir)
        return 2

    _merge_adapter(base_model, actual_adapter, paths.merged_hf_dir)
    _convert_to_fp16_gguf(paths.merged_hf_dir, paths.fp16_gguf, paths.llama_cpp_dir)
    _quantize_to_q4_k_m(paths.fp16_gguf, paths.quant_gguf, quant_bin)
    _write_modelfile(paths.modelfile, paths.quant_gguf)

    if not keep_fp16 and paths.fp16_gguf.exists():
        paths.fp16_gguf.unlink()
        log.info("removed fp16 intermediate: %s", paths.fp16_gguf)

    log.info("DONE: %s (%.1f MB)", paths.quant_gguf, paths.quant_gguf.stat().st_size / 1e6)
    log.info("Deploy with: ollama create %s -f %s", DEFAULT_MODEL_NAME, paths.modelfile)
    return 0


def _build_paths(args: argparse.Namespace) -> ExportPaths:
    out = Path(args.output_dir).expanduser().resolve()
    adapter = Path(args.adapter_dir).expanduser().resolve()
    name = args.model_name
    return ExportPaths(
        adapter_dir=adapter,
        merged_hf_dir=out / f"{name}-merged",
        fp16_gguf=out / f"{name}-fp16.gguf",
        quant_gguf=out / f"{name}-q4_k_m.gguf",
        modelfile=out / "Modelfile",
        llama_cpp_dir=_find_llama_cpp(args.llama_cpp_dir),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter and export as GGUF for Ollama.")
    p.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Output filename prefix.")
    p.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Path to a llama.cpp checkout. Falls back to LLAMA_CPP_DIR env or ~/code/llama.cpp.",
    )
    p.add_argument("--keep-fp16", action="store_true", help="Keep the fp16 GGUF intermediate.")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs without exporting.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _build_paths(args)
    code = _dry_run(paths) if args.dry_run else _export(paths, args.base_model, args.keep_fp16)
    sys.exit(code)


if __name__ == "__main__":
    main()
