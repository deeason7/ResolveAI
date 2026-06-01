"""
Evaluate the fine-tuned QLoRA adapter (or the base model alone) on the
held-out test split.

Standard SFT-classifier eval bundle:

  * Per-class precision / recall / F1 for sentiment and intent
  * Urgency MAE + Spearman rank correlation
  * Structured-output reliability — % of generations that parse cleanly
    against the ComplaintClassification schema
  * Confusion matrices (PNG via matplotlib)
  * Latency benchmarking — tokens/sec, p50, p95
  * **Length-bucketed accuracy** — short / medium / long narratives,
    catches the length-leak risk flagged by audit_labels.py
  * Baseline-vs-fine-tuned delta — pass --no-adapter to run the same
    harness on the bare base model and produce a comparable JSON, so
    we can quote "fine-tuning lifted macro-F1 from X to Y"

Designed to run on Colab T4 after `03_train_qlora.py` finishes, but
the `--dry-run` path works in the CPU-only api container (validates
test data shape, runs JSON-parsing audit, exercises metric math on the
gold labels alone) so the harness can be sanity-checked without GPU.

------------------------------------------------------------------------
Quickstart on Colab
------------------------------------------------------------------------

    # 1. Confirm GPU runtime is on (T4)
    # 2. Upload (or git-clone) the fine_tuning/ folder
    # 3. Make sure ./resolveai-sentiment-lora/ exists (output of 03)
    #    and fine_tuning/data/formatted/test.jsonl is present
    # 4. !pip install -q transformers peft bitsandbytes accelerate \\
    #              datasets matplotlib scikit-learn pyyaml
    # 5. !python 04_evaluate.py \\
    #        --adapter-dir ./resolveai-sentiment-lora \\
    #        --test-path data/formatted/test.jsonl \\
    #        --output-dir results

Outputs land in `results/`:
    metrics.json          machine-readable summary (diffable across runs)
    eval_report.md        human-readable digest with red flags up top
    predictions.jsonl     one record per test example (raw + parsed + truth)
    confusion_sentiment.png
    confusion_intent.png
    length_buckets.json   per-bucket per-class accuracy

Exit codes:
    0 — finished, metrics written
    2 — fatal misconfiguration (missing files, no CUDA when not --dry-run)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "training_config.yaml"
DEFAULT_TEST = SCRIPT_DIR / "data" / "formatted" / "test.jsonl"
DEFAULT_ADAPTER = SCRIPT_DIR / "resolveai-sentiment-lora"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"

# Closed vocabularies — duplicated from app.schemas.classification so this
# script stays standalone (same reasoning as audit_labels.py).
SENTIMENTS = ("neutral", "negative", "extreme_negative")
INTENTS = (
    "information_request",
    "dispute_resolution",
    "account_action",
    "fraud_report",
    "regulatory_complaint",
)

# Bucket boundaries (char length) come from the audit findings:
# neutral median 280, negative median 477, extreme_negative median 951.
# These cuts split the dataset into ~thirds while keeping each class
# represented in every bucket.
LENGTH_BUCKETS: list[tuple[str, int, int]] = [
    ("short", 0, 500),
    ("medium", 500, 1500),
    ("long", 1500, 10_000_000),  # effectively unbounded
]

MAX_NEW_TOKENS = 256  # output JSON is typically <200 tokens, headroom for reasoning


# ---------------------------------------------------------------------------
# Pure-Python helpers — testable without torch
# ---------------------------------------------------------------------------
@dataclass
class TestExample:
    """One row from the test JSONL, with the gold label parsed out."""

    messages: list[dict]
    gold: dict
    narrative_length: int


def _extract_gold(messages: list[dict]) -> dict | None:
    """Pull the parsed assistant-turn JSON (= gold label)."""
    for msg in messages or []:
        if msg.get("role") != "assistant":
            continue
        try:
            payload = json.loads(msg.get("content", ""))
        except (TypeError, ValueError):
            return None
        if isinstance(payload, dict):
            return payload
    return None


def _extract_narrative(messages: list[dict]) -> str:
    """Pull the narrative substring from the user turn (after 'COMPLAINT: ')."""
    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        marker = "COMPLAINT:"
        if marker in content:
            tail = content.split(marker, 1)[1]
            # Stop at the next field marker so we measure narrative only
            for stop in ("\nPRODUCT:", "\nISSUE:", "\nCOMPANY:"):
                if stop in tail:
                    tail = tail.split(stop, 1)[0]
            return tail.strip()
        return content
    return ""


def load_test_examples(path: Path) -> list[TestExample]:
    """Read the test JSONL into typed examples."""
    out: list[TestExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                log.warning("skipping unparseable line %d in %s", i, path)
                continue
            messages = rec.get("messages", [])
            gold = _extract_gold(messages)
            if gold is None:
                log.warning("no gold label on line %d in %s", i, path)
                continue
            narrative = _extract_narrative(messages)
            out.append(
                TestExample(
                    messages=messages,
                    gold=gold,
                    narrative_length=len(narrative),
                )
            )
    return out


def assign_length_bucket(narrative_length: int) -> str:
    """Map a narrative length to its bucket label."""
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= narrative_length < hi:
            return name
    return LENGTH_BUCKETS[-1][0]


def parse_prediction(raw_text: str) -> tuple[dict | None, str]:
    """Try to parse a model output as JSON.

    Returns (parsed_dict_or_none, error_reason).
    Strict — no LLM-driven repair, because the % valid metric is what
    we want to measure.
    """
    if not raw_text:
        return None, "empty"
    # Strip common preludes/suffixes the model might emit even at
    # temperature 0 (Qwen sometimes prefixes a newline).
    text = raw_text.strip()
    # If the model wraps in a markdown fence, peel it.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except ValueError as e:
        return None, f"json_decode: {e.__class__.__name__}"
    if not isinstance(parsed, dict):
        return None, "not_an_object"
    return parsed, "ok"


def per_class_prf1(
    truths: list[str], preds: list[str], classes: tuple[str, ...]
) -> dict[str, dict]:
    """Per-class precision, recall, F1 + macro/weighted averages.

    Implemented in pure Python so the test harness doesn't have to
    pull in sklearn. Matches sklearn.classification_report semantics.
    """
    n_total = len(truths)
    out: dict[str, dict] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(truths, preds, strict=False) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truths, preds, strict=False) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truths, preds, strict=False) if t == cls and p != cls)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro: unweighted mean across classes. Treats each class equally
    # — the metric that matters most for imbalanced datasets.
    macro_p = statistics.fmean(out[c]["precision"] for c in classes)
    macro_r = statistics.fmean(out[c]["recall"] for c in classes)
    macro_f1 = statistics.fmean(out[c]["f1"] for c in classes)

    # Weighted: average weighted by class support. Closer to overall
    # accuracy on imbalanced data; the metric majority-class collapse
    # would score well on.
    if n_total:
        weighted_p = sum(out[c]["precision"] * out[c]["support"] for c in classes) / n_total
        weighted_r = sum(out[c]["recall"] * out[c]["support"] for c in classes) / n_total
        weighted_f1 = sum(out[c]["f1"] * out[c]["support"] for c in classes) / n_total
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0

    out["macro_avg"] = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
        "support": n_total,
    }
    out["weighted_avg"] = {
        "precision": round(weighted_p, 4),
        "recall": round(weighted_r, 4),
        "f1": round(weighted_f1, 4),
        "support": n_total,
    }
    return out


def urgency_metrics(truths: list[int], preds: list[int]) -> dict:
    """MAE + Spearman rank correlation for ordinal urgency.

    Urgency is 1-5 ordinal — MAE captures "how far off on average",
    Spearman captures "does the ranking agree even if absolute values
    drift". A model with MAE=0.5 and Spearman=0.9 is better than one
    with MAE=0.3 and Spearman=0.4 (the second one is randomly close
    but doesn't track the gradient).
    """
    if not truths:
        return {"n": 0, "mae": None, "spearman": None}
    n = len(truths)
    mae = sum(abs(t - p) for t, p in zip(truths, preds, strict=False)) / n
    spearman = _spearman(truths, preds) if n >= 2 else None
    return {
        "n": n,
        "mae": round(mae, 4),
        "spearman": round(spearman, 4) if spearman is not None else None,
    }


def _spearman(x: list[int], y: list[int]) -> float | None:
    """Spearman ρ via Pearson on ranks. No scipy needed."""
    rx = _ranks(x)
    ry = _ranks(y)
    mean_x = statistics.fmean(rx)
    mean_y = statistics.fmean(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=False))
    den_x = sum((a - mean_x) ** 2 for a in rx)
    den_y = sum((b - mean_y) ** 2 for b in ry)
    if den_x == 0 or den_y == 0:
        return None
    return num / ((den_x * den_y) ** 0.5)


def _ranks(values: list[int]) -> list[float]:
    """Average-rank for ties (the standard Spearman convention)."""
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j + 1 < len(sorted_idx) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # ranks are 1-indexed by convention
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def confusion_matrix(
    truths: list[str], preds: list[str], classes: tuple[str, ...]
) -> list[list[int]]:
    """K×K count matrix. rows=truth, cols=pred, in `classes` order."""
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0] * len(classes) for _ in classes]
    for t, p in zip(truths, preds, strict=False):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    return mat


def length_bucketed_accuracy(
    examples: list[TestExample],
    pred_sentiments: list[str | None],
) -> dict:
    """Per-bucket per-class accuracy for sentiment.

    Catches the length-leak risk surfaced by audit_labels.py: if the
    model is great at extreme_negative on long narratives but bad on
    short ones, the model learned length as a shortcut.
    """
    bucket_truths: dict[str, list[str]] = defaultdict(list)
    bucket_preds: dict[str, list[str | None]] = defaultdict(list)
    for ex, pred in zip(examples, pred_sentiments, strict=False):
        bucket = assign_length_bucket(ex.narrative_length)
        bucket_truths[bucket].append(ex.gold["sentiment"])
        bucket_preds[bucket].append(pred)

    out: dict[str, Any] = {}
    for bucket_name, lo, hi in LENGTH_BUCKETS:
        truths = bucket_truths.get(bucket_name, [])
        preds = bucket_preds.get(bucket_name, [])
        n = len(truths)
        per_class: dict[str, dict] = {}
        for cls in SENTIMENTS:
            cls_idx = [i for i, t in enumerate(truths) if t == cls]
            cls_n = len(cls_idx)
            cls_correct = sum(1 for i in cls_idx if preds[i] == cls)
            per_class[cls] = {
                "n": cls_n,
                "correct": cls_correct,
                "accuracy": round(cls_correct / cls_n, 4) if cls_n else None,
            }
        overall_correct = sum(1 for t, p in zip(truths, preds, strict=False) if t == p)
        out[bucket_name] = {
            "range": [lo, hi if hi < 10_000_000 else None],
            "n": n,
            "overall_accuracy": round(overall_correct / n, 4) if n else None,
            "per_class": per_class,
        }
    return out


# ---------------------------------------------------------------------------
# Heavy-import helpers (torch / transformers / peft) — deferred so --dry-run
# stays on the CPU container.
# ---------------------------------------------------------------------------
def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        log.critical("config file not found: %s", path)
        sys.exit(2)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_adapter_path(adapter_dir: Path) -> Path:
    """Find the actual adapter directory.

    Depending on TRL/PEFT version and whether training finished cleanly,
    the final adapter may live at the top of `adapter_dir` or only inside
    intermediate `checkpoint-N/` subdirectories. Be tolerant: if the
    top-level doesn't have `adapter_config.json`, fall back to the
    highest-numbered checkpoint subdir that does.
    """
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir
    candidates: list[tuple[int, Path]] = []
    for sub in adapter_dir.glob("checkpoint-*"):
        if not sub.is_dir():
            continue
        if not (sub / "adapter_config.json").exists():
            continue
        try:
            step = int(sub.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, sub))
    if not candidates:
        raise FileNotFoundError(
            f"No adapter_config.json found at {adapter_dir} or in any "
            "checkpoint-N/ subdirectory. Did training complete and save?"
        )
    candidates.sort()
    step, path = candidates[-1]
    log.info(
        "no adapter_config.json at top level; using latest checkpoint: %s (step %d)",
        path.name,
        step,
    )
    return path


def _load_model_and_tokenizer(base_model: str, adapter_dir: Path | None):
    """Load the base model (4-bit) and optionally apply the LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("loading tokenizer for %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("loading base model in 4-bit NF4")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        device_map="auto",
    )
    if adapter_dir is not None:
        from peft import PeftModel

        resolved = _resolve_adapter_path(adapter_dir)
        log.info("applying LoRA adapter from %s", resolved)
        model = PeftModel.from_pretrained(model, str(resolved))
    model.eval()
    return model, tokenizer


def _generate_one(model, tokenizer, messages: list[dict], max_new_tokens: int) -> tuple[str, dict]:
    """Run a single generation. Returns (raw_text, timing_dict)."""
    import torch

    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_input = int(inputs["input_ids"].shape[1])

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - start

    n_generated = int(out.shape[1]) - n_input
    text = tokenizer.decode(out[0, n_input:], skip_special_tokens=True)
    return text, {
        "input_tokens": n_input,
        "output_tokens": n_generated,
        "latency_s": elapsed,
        "tokens_per_sec": (n_generated / elapsed) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    n_total: int
    n_valid_json: int
    pct_valid_json: float
    sentiment_prf1: dict
    intent_prf1: dict
    urgency: dict
    confusion_sentiment: list[list[int]]
    confusion_intent: list[list[int]]
    length_buckets: dict
    latency: dict
    parse_errors: dict[str, int] = field(default_factory=dict)


def _render_confusion_table(matrix: list[list[int]], classes: tuple[str, ...]) -> list[str]:
    """ASCII confusion table for the markdown report."""
    header = "| truth \\ pred | " + " | ".join(classes) + " |"
    sep = "|---" * (len(classes) + 1) + "|"
    lines = [header, sep]
    for i, cls in enumerate(classes):
        row = "| " + cls + " | " + " | ".join(str(matrix[i][j]) for j in range(len(classes))) + " |"
        lines.append(row)
    return lines


def render_report(label: str, result: EvalResult) -> str:
    red: list[str] = []
    sent_macro = result.sentiment_prf1.get("macro_avg", {}).get("f1", 0)
    if sent_macro < 0.70:
        red.append(f"⚠️  Sentiment macro-F1 is **{sent_macro}** — below the 0.70 quality bar.")
    neutral_recall = result.sentiment_prf1.get("neutral", {}).get("recall", 0)
    if neutral_recall < 0.50:
        red.append(
            f"⚠️  Neutral recall is **{neutral_recall}** — minority class is under-learned. "
            "Consider bumping `class_weights.manual_scale.neutral` or switching method "
            "to `inverse_freq`."
        )
    if result.pct_valid_json < 95.0:
        red.append(
            f"⚠️  Only **{result.pct_valid_json}%** of generations parsed as valid JSON — "
            "target is >95%. Check the parse-error breakdown below."
        )
    # Length-leak signal: short-bucket accuracy on extreme_negative
    # significantly worse than long-bucket accuracy suggests the model
    # learned length as a shortcut.
    sa = (
        result.length_buckets.get("short", {})
        .get("per_class", {})
        .get("extreme_negative", {})
        .get("accuracy")
    )
    la = (
        result.length_buckets.get("long", {})
        .get("per_class", {})
        .get("extreme_negative", {})
        .get("accuracy")
    )
    if sa is not None and la is not None and la - sa > 0.20:
        red.append(
            f"⚠️  Length-leak suspected: extreme_negative accuracy is **{sa}** on short "
            f"narratives vs **{la}** on long ones (Δ {round(la - sa, 2)}). Model may be "
            "using length as a shortcut."
        )

    lines: list[str] = []
    lines.append(f"# Evaluation Report — {label}")
    lines.append("")
    lines.append(f"_Total test examples: **{result.n_total:,}**_  ")
    lines.append(f"_Valid JSON: **{result.n_valid_json:,}** ({result.pct_valid_json}%)_  ")
    lines.append("")

    lines.append("## Red Flags")
    if red:
        lines.extend(f"- {rf}" for rf in red)
    else:
        lines.append("- _None detected at the configured thresholds._")
    lines.append("")

    lines.append("## Sentiment Classification")
    lines.append("")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in list(SENTIMENTS) + ["macro_avg", "weighted_avg"]:
        d = result.sentiment_prf1.get(cls, {})
        lines.append(
            f"| {cls} | {d.get('precision', 0)} | {d.get('recall', 0)} | "
            f"{d.get('f1', 0)} | {d.get('support', 0)} |"
        )
    lines.append("")
    lines.append("### Sentiment confusion matrix")
    lines.append("")
    lines.extend(_render_confusion_table(result.confusion_sentiment, SENTIMENTS))
    lines.append("")

    lines.append("## Intent Classification")
    lines.append("")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in list(INTENTS) + ["macro_avg", "weighted_avg"]:
        d = result.intent_prf1.get(cls, {})
        lines.append(
            f"| {cls} | {d.get('precision', 0)} | {d.get('recall', 0)} | "
            f"{d.get('f1', 0)} | {d.get('support', 0)} |"
        )
    lines.append("")
    lines.append("### Intent confusion matrix")
    lines.append("")
    lines.extend(_render_confusion_table(result.confusion_intent, INTENTS))
    lines.append("")

    lines.append("## Urgency (ordinal 1-5)")
    u = result.urgency
    lines.append(f"- MAE: **{u.get('mae')}**")
    lines.append(f"- Spearman ρ: **{u.get('spearman')}**")
    lines.append(f"- n: {u.get('n')}")
    lines.append("")

    lines.append("## Length-Bucketed Accuracy (sentiment)")
    lines.append("")
    lines.append(
        "_Catches the length-leak risk flagged by audit_labels.py. If "
        "accuracy collapses on short narratives but stays high on long "
        "ones, the model is using length as a shortcut._"
    )
    lines.append("")
    lines.append(
        "| bucket | range (chars) | n | overall acc | neutral acc | negative acc | extreme acc |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, _, _ in LENGTH_BUCKETS:
        b = result.length_buckets.get(name, {})
        rng = b.get("range", [None, None])
        rng_str = f"[{rng[0]}, {rng[1] if rng[1] is not None else '∞'})"
        pc = b.get("per_class", {})
        lines.append(
            f"| {name} | {rng_str} | {b.get('n', 0)} | "
            f"{b.get('overall_accuracy')} | "
            f"{pc.get('neutral', {}).get('accuracy')} | "
            f"{pc.get('negative', {}).get('accuracy')} | "
            f"{pc.get('extreme_negative', {}).get('accuracy')} |"
        )
    lines.append("")

    lines.append("## Structured-Output Reliability")
    lines.append(
        f"- Valid JSON: **{result.n_valid_json}** / {result.n_total} ({result.pct_valid_json}%)"
    )
    if result.parse_errors:
        lines.append("- Parse-error breakdown:")
        for err, c in sorted(result.parse_errors.items(), key=lambda x: -x[1]):
            lines.append(f"  - `{err}`: {c}")
    lines.append("")

    lines.append("## Latency Benchmark")
    lat = result.latency
    lines.append(f"- n: {lat.get('n', 0)}")
    lines.append(f"- p50 latency: **{lat.get('p50_s')} s**")
    lines.append(f"- p95 latency: **{lat.get('p95_s')} s**")
    lines.append(f"- median tokens/sec: **{lat.get('median_tokens_per_sec')}**")
    lines.append(
        f"- input tokens median/p95: {lat.get('input_tokens_median')} / {lat.get('input_tokens_p95')}"
    )
    lines.append(
        f"- output tokens median/p95: {lat.get('output_tokens_median')} / {lat.get('output_tokens_p95')}"
    )
    return "\n".join(lines) + "\n"


def _summary_stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return s[idx]

    return {
        "n": n,
        "p50": round(q(0.5), 4),
        "p95": round(q(0.95), 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "mean": round(statistics.fmean(s), 4),
    }


def _save_confusion_png(
    matrix: list[list[int]], classes: tuple[str, ...], title: str, path: Path
) -> None:
    """Save a confusion matrix as a PNG. Matplotlib only — no seaborn."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend for Colab/Docker
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping %s", path)
        return

    fig, ax = plt.subplots(figsize=(0.8 * len(classes) + 3, 0.8 * len(classes) + 3))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                color="white" if matrix[i][j] > max(max(r) for r in matrix) / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _run_inference(
    model, tokenizer, examples: list[TestExample], max_new_tokens: int
) -> list[dict]:
    """Generate predictions for every test example. Returns per-example dicts."""
    results: list[dict] = []
    for i, ex in enumerate(examples):
        raw, timing = _generate_one(model, tokenizer, ex.messages, max_new_tokens)
        parsed, err = parse_prediction(raw)
        results.append(
            {
                "gold": ex.gold,
                "narrative_length": ex.narrative_length,
                "raw": raw,
                "parsed": parsed,
                "parse_error": err,
                "timing": timing,
            }
        )
        if (i + 1) % 25 == 0:
            log.info("inference progress: %d/%d", i + 1, len(examples))
    return results


def _aggregate(examples: list[TestExample], preds: list[dict]) -> EvalResult:
    """Roll up per-example results into the EvalResult bundle."""
    n_total = len(preds)
    parse_errors: Counter[str] = Counter()
    pred_sent: list[str | None] = []
    pred_int: list[str | None] = []
    pred_urg: list[int | None] = []
    truth_sent: list[str] = []
    truth_int: list[str] = []
    truth_urg: list[int] = []
    latencies: list[float] = []
    tps: list[float] = []
    input_tokens: list[float] = []
    output_tokens: list[float] = []

    for ex, p in zip(examples, preds, strict=False):
        truth_sent.append(ex.gold["sentiment"])
        truth_int.append(ex.gold["intent"])
        truth_urg.append(int(ex.gold["urgency"]))
        parsed = p["parsed"]
        if parsed is None:
            parse_errors[p["parse_error"]] += 1
            pred_sent.append(None)
            pred_int.append(None)
            pred_urg.append(None)
        else:
            parse_errors["ok"] += 1
            ps = parsed.get("sentiment") if isinstance(parsed.get("sentiment"), str) else None
            pi = parsed.get("intent") if isinstance(parsed.get("intent"), str) else None
            pu = parsed.get("urgency")
            pred_sent.append(ps)
            pred_int.append(pi)
            try:
                pred_urg.append(int(pu))
            except (TypeError, ValueError):
                pred_urg.append(None)
        t = p["timing"]
        latencies.append(t["latency_s"])
        tps.append(t["tokens_per_sec"])
        input_tokens.append(t["input_tokens"])
        output_tokens.append(t["output_tokens"])

    n_valid = parse_errors.get("ok", 0)
    pct_valid = round(100.0 * n_valid / n_total, 2) if n_total else 0.0

    # PRF1 — substitute a sentinel for None predictions so they count as
    # misclassifications instead of being silently dropped.
    pred_sent_filled = [p or "__INVALID__" for p in pred_sent]
    pred_int_filled = [p or "__INVALID__" for p in pred_int]

    sentiment_prf1 = per_class_prf1(truth_sent, pred_sent_filled, SENTIMENTS)
    intent_prf1 = per_class_prf1(truth_int, pred_int_filled, INTENTS)

    # Urgency: only score rows where the model produced a valid 1-5 integer
    urg_pairs = [
        (t, p)
        for t, p in zip(truth_urg, pred_urg, strict=False)
        if isinstance(p, int) and 1 <= p <= 5
    ]
    urgency = urgency_metrics(
        [t for t, _ in urg_pairs],
        [p for _, p in urg_pairs],
    )

    cm_sent = confusion_matrix(truth_sent, pred_sent_filled, SENTIMENTS)
    cm_int = confusion_matrix(truth_int, pred_int_filled, INTENTS)

    length_buckets = length_bucketed_accuracy(examples, pred_sent_filled)

    lat = _summary_stats(latencies)
    lat["p50_s"] = lat.pop("p50", None)
    lat["p95_s"] = lat.pop("p95", None)
    tps_summary = _summary_stats(tps)
    lat["median_tokens_per_sec"] = tps_summary.get("p50")
    in_tok = _summary_stats(input_tokens)
    out_tok = _summary_stats(output_tokens)
    lat["input_tokens_median"] = in_tok.get("p50")
    lat["input_tokens_p95"] = in_tok.get("p95")
    lat["output_tokens_median"] = out_tok.get("p50")
    lat["output_tokens_p95"] = out_tok.get("p95")

    return EvalResult(
        n_total=n_total,
        n_valid_json=n_valid,
        pct_valid_json=pct_valid,
        sentiment_prf1=sentiment_prf1,
        intent_prf1=intent_prf1,
        urgency=urgency,
        confusion_sentiment=cm_sent,
        confusion_intent=cm_int,
        length_buckets=length_buckets,
        latency=lat,
        parse_errors=dict(parse_errors),
    )


def _evaluate(args: argparse.Namespace) -> int:
    test_path = Path(args.test_path)
    if not test_path.exists():
        log.critical("test file missing: %s", test_path)
        return 2

    log.info("loading test examples from %s", test_path)
    examples = load_test_examples(test_path)
    if not examples:
        log.critical("no usable test examples found")
        return 2
    if args.limit:
        examples = examples[: args.limit]
    log.info("test set: n=%d (limit applied: %s)", len(examples), args.limit or "no")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Validate JSON parsing on the GOLD labels themselves — a sanity
        # check that the test file is well-formed and the schema-extraction
        # helpers work. Also exercise the metric math by "predicting" the
        # gold labels (should give perfect scores).
        log.info("DRY RUN: validating test data and exercising metric math")
        truth_sent = [ex.gold["sentiment"] for ex in examples]
        truth_int = [ex.gold["intent"] for ex in examples]
        sentiment_prf1 = per_class_prf1(truth_sent, truth_sent, SENTIMENTS)
        intent_prf1 = per_class_prf1(truth_int, truth_int, INTENTS)
        buckets = length_bucketed_accuracy(examples, truth_sent)
        log.info(
            "DRY RUN: sentiment macro-F1 (gold-vs-gold) = %s (expect 1.0)",
            sentiment_prf1["macro_avg"]["f1"],
        )
        log.info(
            "DRY RUN: intent macro-F1 (gold-vs-gold) = %s (expect 1.0)",
            intent_prf1["macro_avg"]["f1"],
        )
        for name in (b[0] for b in LENGTH_BUCKETS):
            b = buckets[name]
            log.info(
                "DRY RUN: bucket=%s n=%d (neutral=%d / negative=%d / extreme_negative=%d)",
                name,
                b["n"],
                b["per_class"]["neutral"]["n"],
                b["per_class"]["negative"]["n"],
                b["per_class"]["extreme_negative"]["n"],
            )
        log.info("DRY RUN: model load and inference skipped; harness verified.")
        return 0

    # ---- GPU path ----
    import torch

    if not torch.cuda.is_available():
        log.critical("CUDA is not available — pass --dry-run to validate the harness on CPU.")
        return 2

    cfg = _load_yaml_config(Path(args.config))
    base_model = cfg.get("model", {}).get("base_model")
    if not base_model:
        log.critical("config missing model.base_model")
        return 2

    adapter_dir = None if args.no_adapter else Path(args.adapter_dir)
    if adapter_dir is not None and not adapter_dir.exists():
        log.critical("adapter directory missing: %s", adapter_dir)
        return 2

    label = "baseline" if args.no_adapter else f"adapter@{adapter_dir.name}"
    log.info("evaluating: %s", label)

    model, tokenizer = _load_model_and_tokenizer(base_model, adapter_dir)
    preds = _run_inference(model, tokenizer, examples, args.max_new_tokens)
    result = _aggregate(examples, preds)

    # Write artifacts
    suffix = "_baseline" if args.no_adapter else ""
    metrics_path = output_dir / f"metrics{suffix}.json"
    report_path = output_dir / f"eval_report{suffix}.md"
    preds_path = output_dir / f"predictions{suffix}.jsonl"
    cm_sent_path = output_dir / f"confusion_sentiment{suffix}.png"
    cm_int_path = output_dir / f"confusion_intent{suffix}.png"
    length_path = output_dir / f"length_buckets{suffix}.json"

    payload = {
        "label": label,
        "n_total": result.n_total,
        "n_valid_json": result.n_valid_json,
        "pct_valid_json": result.pct_valid_json,
        "sentiment_prf1": result.sentiment_prf1,
        "intent_prf1": result.intent_prf1,
        "urgency": result.urgency,
        "confusion_sentiment": result.confusion_sentiment,
        "confusion_intent": result.confusion_intent,
        "length_buckets": result.length_buckets,
        "latency": result.latency,
        "parse_errors": result.parse_errors,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", metrics_path)

    with preds_path.open("w", encoding="utf-8") as fh:
        for ex, p in zip(examples, preds, strict=False):
            fh.write(
                json.dumps(
                    {
                        "gold": ex.gold,
                        "narrative_length": ex.narrative_length,
                        "raw": p["raw"],
                        "parsed": p["parsed"],
                        "parse_error": p["parse_error"],
                        "timing": p["timing"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log.info("wrote %s", preds_path)

    length_path.write_text(json.dumps(result.length_buckets, indent=2), encoding="utf-8")
    log.info("wrote %s", length_path)

    _save_confusion_png(
        result.confusion_sentiment, SENTIMENTS, f"Sentiment ({label})", cm_sent_path
    )
    _save_confusion_png(result.confusion_intent, INTENTS, f"Intent ({label})", cm_int_path)

    report_path.write_text(render_report(label, result), encoding="utf-8")
    log.info("wrote %s", report_path)

    log.info("DONE — see %s", report_path)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the QLoRA adapter on the test set.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to training_config.yaml.")
    p.add_argument("--test-path", default=str(DEFAULT_TEST), help="Path to test.jsonl.")
    p.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_ADAPTER),
        help="Path to the trained LoRA adapter directory.",
    )
    p.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run the base model alone (baseline) — produces a comparable metrics file.",
    )
    p.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT), help="Where to write metrics + plots."
    )
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of test examples (for smoke tests; full set otherwise).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate test data + metric math without loading the model.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sys.exit(_evaluate(args))


if __name__ == "__main__":
    main()
