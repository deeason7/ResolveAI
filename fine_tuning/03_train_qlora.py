"""
QLoRA fine-tuning script for the ResolveAI complaint classifier.

Loads Qwen2.5-3B-Instruct in 4-bit NF4 quantization, attaches LoRA
adapters on all linear projections, and trains for 3 epochs on the
ChatML-formatted data emitted by ``02_format_training_data.py``.

Designed for a free Google Colab T4 GPU (16 GB VRAM). All hyperparameters
live in ``configs/training_config.yaml`` so experiments can be tracked
by config diffs rather than ad-hoc CLI flag combinations.

------------------------------------------------------------------------
Quickstart on Colab
------------------------------------------------------------------------

    # 1. Enable GPU: Runtime → Change runtime type → T4 GPU
    # 2. Upload (or git-clone) the fine_tuning/ folder
    # 3. Install dependencies (one-shot; persistent on Colab Pro):
    #
    #    !pip install -q -U transformers peft trl bitsandbytes accelerate \\
    #                       datasets pyyaml huggingface_hub
    #
    # 4. (Optional) Login if you want to push the adapter to HF Hub later:
    #
    #    from huggingface_hub import notebook_login; notebook_login()
    #
    # 5. Run the trainer:
    #
    #    !python 03_train_qlora.py --config configs/training_config.yaml
    #
    # The trained adapter (~70 MB) lands in ``./resolveai-sentiment-lora/``
    # alongside the tokenizer — ready for ``04_evaluate.py`` and
    # ``05_export_gguf.py`` in the next phase.

------------------------------------------------------------------------
Local invocation (no training, just config + data sanity check)
------------------------------------------------------------------------

    python fine_tuning/03_train_qlora.py --dry-run

Exit codes
----------
0  finished cleanly, best checkpoint saved
2  fatal misconfiguration (missing config, missing data, no CUDA, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_qlora")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "training_config.yaml"


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path against ``SCRIPT_DIR`` if relative.

    Lets the config use ``data/formatted/train.jsonl`` regardless of cwd,
    while still honouring absolute overrides.
    """
    p = Path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    """Typed wrapper around the YAML config.

    Validation is deliberately shallow — we only check the top-level
    sections exist. The downstream builders (`_build_*`) will KeyError
    early if specific fields are missing, which is the behaviour we want:
    a typo in the YAML should crash before we load a 3B model into VRAM.
    """

    model: dict[str, Any]
    quantization: dict[str, Any]
    lora: dict[str, Any]
    training: dict[str, Any]
    data: dict[str, Any]
    # Optional — defaults to disabled so older configs without this block
    # keep working unchanged.
    class_weights: dict[str, Any] = field(default_factory=dict)


def _load_config(path: Path) -> TrainingConfig:
    if not path.exists():
        log.critical("config file not found: %s", path)
        sys.exit(2)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    required = ("model", "quantization", "lora", "training", "data")
    missing = [k for k in required if k not in raw]
    if missing:
        log.critical("config missing required sections: %s", missing)
        sys.exit(2)
    return TrainingConfig(
        model=raw["model"],
        quantization=raw["quantization"],
        lora=raw["lora"],
        training=raw["training"],
        data=raw["data"],
        class_weights=raw.get("class_weights", {}),
    )


# ---------------------------------------------------------------------------
# Helpers — heavy imports are deferred so `--help` works without CUDA libs.
# ---------------------------------------------------------------------------
def _resolve_dtype(name: str):
    """Map a config string to a torch dtype."""
    import torch

    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in table:
        log.critical("unsupported dtype: %s (choose from %s)", name, list(table))
        sys.exit(2)
    return table[name]


def _resolve_data_paths(cfg: TrainingConfig) -> tuple[Path, Path]:
    train_path = _resolve_path(cfg.data["train_path"])
    val_path = _resolve_path(cfg.data["val_path"])
    for p in (train_path, val_path):
        if not p.exists():
            log.critical("data file missing: %s", p)
            sys.exit(2)
    return train_path, val_path


def _peek_jsonl(path: Path, n: int = 1) -> list[dict]:
    """Read the first ``n`` records of a JSONL file using stdlib only.

    Used by --dry-run to validate the data files without pulling in the
    heavy ``datasets`` library.
    """
    import json

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _load_datasets(cfg: TrainingConfig):
    """Stream the train/val JSONL files into a HuggingFace ``DatasetDict``.

    SFTTrainer auto-detects the ``messages`` column and applies the
    tokenizer's chat template — no manual formatting needed here.
    """
    from datasets import load_dataset

    train_path, val_path = _resolve_data_paths(cfg)
    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    log.info(
        "loaded train=%d val=%d",
        len(ds["train"]),
        len(ds["validation"]),
    )
    return ds


def _build_quant_config(cfg: TrainingConfig):
    from transformers import BitsAndBytesConfig

    q = cfg.quantization
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_resolve_dtype(q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def _build_lora_config(cfg: TrainingConfig):
    from peft import LoraConfig

    lcfg = cfg.lora
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        target_modules=lcfg["target_modules"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"],
        task_type=lcfg["task_type"],
    )


# ---------------------------------------------------------------------------
# Class-imbalance helpers (pure Python — testable without GPU libs)
# ---------------------------------------------------------------------------
WEIGHT_COLUMN = "weight"


def _extract_sentiment_from_messages(messages: list[dict]) -> str | None:
    """Pull the `sentiment` field from the assistant turn's JSON content.

    Returns None if the turn isn't found or isn't valid JSON. Used both
    when computing class weights from the train set and when tagging each
    example with its own weight.
    """
    for msg in messages or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return None
        if isinstance(payload, dict):
            s = payload.get("sentiment")
            return s if isinstance(s, str) else None
    return None


def _derive_class_weights(
    class_counts: dict[str, int],
    method: str,
    manual_scale: dict[str, float] | None = None,
) -> dict[str, float]:
    """Translate class counts into per-class loss multipliers.

    method=sqrt_inverse_freq → weight = sqrt(N / (K * N_class))
    method=inverse_freq      → weight = N / (K * N_class)
    method=none              → uniform 1.0

    The manual_scale dict (if provided) multiplies each class's weight at
    the end, letting you nudge classes without re-deriving from frequencies.
    """
    manual_scale = manual_scale or {}
    if not class_counts:
        return {}
    if method == "none":
        return {k: 1.0 * manual_scale.get(k, 1.0) for k in class_counts}
    n_total = sum(class_counts.values())
    k = len(class_counts)
    weights: dict[str, float] = {}
    for cls, count in class_counts.items():
        if count <= 0:
            weights[cls] = 0.0
            continue
        raw = n_total / (k * count)
        if method == "sqrt_inverse_freq":
            raw = math.sqrt(raw)
        elif method != "inverse_freq":
            log.warning("unknown class_weights.method=%s — falling back to uniform", method)
            raw = 1.0
        weights[cls] = raw * manual_scale.get(cls, 1.0)
    return weights


def _count_sentiments_in_jsonl(path: Path) -> dict[str, int]:
    """Count sentiment classes by reading the training JSONL once with stdlib."""
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            sentiment = _extract_sentiment_from_messages(rec.get("messages", []))
            if sentiment:
                counts[sentiment] += 1
    return dict(counts)


def _filter_sft_kwargs(candidate: dict, accepted_params: set[str]) -> dict:
    """Keep only kwargs this TRL version's SFTConfig accepts.

    TRL renames things between minor versions. Most notably 0.12+ removed
    ``max_seq_length`` from SFTConfig in favor of ``max_length`` (and may
    remove it entirely in a later version). Rather than pin TRL or guess,
    we introspect ``SFTConfig.__init__`` and:
      * drop kwargs not in the signature (with a warning, so a silently
        ignored config change is visible in the log),
      * rename ``max_seq_length`` to ``max_length`` if the former is
        rejected and the latter is accepted.
    """
    out = {k: v for k, v in candidate.items() if k in accepted_params}
    dropped = [k for k in candidate if k not in accepted_params]
    if "max_seq_length" in dropped and "max_length" in accepted_params:
        out["max_length"] = candidate["max_seq_length"]
        dropped.remove("max_seq_length")
    if dropped:
        log.warning(
            "dropping SFTConfig kwargs not accepted by this TRL version: %s",
            dropped,
        )
    return out


def _build_sft_config(cfg: TrainingConfig):
    import inspect

    from trl import SFTConfig

    t = cfg.training
    candidate = {
        "output_dir": t["output_dir"],
        "num_train_epochs": t["num_train_epochs"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "per_device_eval_batch_size": t.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "learning_rate": t["learning_rate"],
        "lr_scheduler_type": t["lr_scheduler_type"],
        "warmup_ratio": t["warmup_ratio"],
        "weight_decay": t["weight_decay"],
        "optim": t["optim"],
        "fp16": t["fp16"],
        "bf16": t["bf16"],
        "max_grad_norm": t["max_grad_norm"],
        "max_seq_length": cfg.model["max_seq_length"],
        "gradient_checkpointing": t["gradient_checkpointing"],
        "logging_steps": t["logging_steps"],
        "eval_strategy": t["eval_strategy"],
        "eval_steps": t["eval_steps"],
        "save_strategy": t["save_strategy"],
        "save_steps": t["save_steps"],
        "save_total_limit": t["save_total_limit"],
        "load_best_model_at_end": t["load_best_model_at_end"],
        "metric_for_best_model": t["metric_for_best_model"],
        "greater_is_better": t["greater_is_better"],
        "report_to": t["report_to"],
        "seed": t["seed"],
    }
    accepted_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    kwargs = _filter_sft_kwargs(candidate, accepted_params)
    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def _train(cfg: TrainingConfig, dry_run: bool) -> int:
    """Run the full QLoRA training loop.

    With ``dry_run=True`` we stop after a stdlib JSONL sanity check — no
    GPU libraries are imported, so it works inside the CPU-only api
    container.
    """
    if dry_run:
        train_path, val_path = _resolve_data_paths(cfg)
        n_train = _count_jsonl(train_path)
        n_val = _count_jsonl(val_path)
        sample = _peek_jsonl(train_path, n=1)[0]
        if "messages" not in sample:
            log.critical("expected 'messages' key in JSONL record; got keys=%s", list(sample))
            return 2
        log.info("DRY RUN: train=%d val=%d", n_train, n_val)
        log.info("sample record (truncated): %s", str(sample)[:300])
        cw_cfg = cfg.class_weights or {}
        if cw_cfg.get("enabled"):
            counts = _count_sentiments_in_jsonl(train_path)
            weights = _derive_class_weights(
                counts,
                cw_cfg.get("method", "sqrt_inverse_freq"),
                cw_cfg.get("manual_scale", {}) or {},
            )
            log.info(
                "DRY RUN: class_weights enabled (method=%s) counts=%s weights=%s",
                cw_cfg.get("method"),
                counts,
                {k: round(v, 3) for k, v in weights.items()},
            )
        else:
            log.info("DRY RUN: class_weights disabled in config")
        log.info("DRY RUN: model load and training skipped; data path verified.")
        return 0

    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from torch.nn import CrossEntropyLoss
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        log.critical("CUDA is not available — QLoRA training requires a GPU.")
        return 2

    # ---------------------------------------------------------------------
    # WeightedSFTTrainer + collator are nested inside _train so that torch
    # only loads when we're actually training. Keeps --dry-run and --help
    # importable on CPU-only environments.
    # ---------------------------------------------------------------------
    class WeightedDataCollator:
        """Wrap the trainer's inner collator and lift our per-example
        weight column into a batch tensor.

        The default TRL collator only knows about tokenized fields; any
        extra column (like `weight`) gets dropped. We pop it ourselves,
        delegate the rest, and re-attach the resulting (B,) tensor.
        """

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, features):
            weights = [float(f.pop(WEIGHT_COLUMN, 1.0)) for f in features]
            batch = self.inner(features)
            batch[WEIGHT_COLUMN] = torch.tensor(weights, dtype=torch.float32)
            return batch

    class WeightedSFTTrainer(SFTTrainer):
        """SFTTrainer with per-example loss weighting.

        Standard SFT produces one mean-reduced scalar loss per batch. To
        apply a different weight to each example we have to recompute the
        per-token CE from logits, mean over valid tokens to get
        per-example loss, multiply by the per-example weight, then mean.
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop(WEIGHT_COLUMN, None)
            labels = inputs.get("labels")
            outputs = model(**inputs)

            # If weights weren't threaded through (e.g., eval batches built
            # without our collator), fall back to vanilla loss.
            if weights is None or labels is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            # Standard causal-LM shift: predict token t+1 from positions
            # [:-1]; compare against labels [1:].
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
            flat_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            # Per-example mean over only the supervised positions (assistant
            # turn); user/system tokens are -100 in labels and excluded.
            valid = (shift_labels != -100).float()
            denom = valid.sum(dim=-1).clamp(min=1.0)
            per_example = (flat_loss * valid).sum(dim=-1) / denom

            w = weights.to(per_example.device).to(per_example.dtype)
            loss = (per_example * w).mean()
            return (loss, outputs) if return_outputs else loss

    log.info("loading datasets...")
    ds = _load_datasets(cfg)

    model_name = cfg.model["base_model"]
    log.info("loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Qwen2.5 ships with an eos_token; reusing it as pad is the standard
        # pattern for causal LMs with no dedicated pad token.
        tokenizer.pad_token = tokenizer.eos_token

    log.info("loading base model with 4-bit NF4 quantization...")
    quant_config = _build_quant_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    # Wraps the quantized model so gradients can flow through LoRA layers
    # without re-quantizing the frozen base on every step.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.training["gradient_checkpointing"],
    )

    log.info(
        "attaching LoRA adapter (r=%d, alpha=%d, targets=%s)",
        cfg.lora["r"],
        cfg.lora["alpha"],
        cfg.lora["target_modules"],
    )
    lora_config = _build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------------
    # Optionally derive + attach per-example class weights.
    # ---------------------------------------------------------------------
    cw_cfg = cfg.class_weights or {}
    weights_enabled = bool(cw_cfg.get("enabled", False))
    class_weights: dict[str, float] = {}
    train_ds = ds["train"]
    eval_ds = ds["validation"]
    if weights_enabled:
        method = cw_cfg.get("method", "sqrt_inverse_freq")
        manual_scale = cw_cfg.get("manual_scale", {}) or {}
        train_counts: Counter[str] = Counter()
        for ex in train_ds:
            s = _extract_sentiment_from_messages(ex.get("messages", []))
            if s:
                train_counts[s] += 1
        class_weights = _derive_class_weights(dict(train_counts), method, manual_scale)
        log.info(
            "class weights enabled: method=%s counts=%s weights=%s",
            method,
            dict(train_counts),
            {k: round(v, 3) for k, v in class_weights.items()},
        )

        def _attach_weight(ex):
            s = _extract_sentiment_from_messages(ex.get("messages", []))
            ex[WEIGHT_COLUMN] = float(class_weights.get(s, 1.0))
            return ex

        train_ds = train_ds.map(_attach_weight)
        eval_ds = eval_ds.map(_attach_weight)
    else:
        log.info("class weights disabled — vanilla SFT loss")

    log.info("building SFTTrainer and starting training...")
    sft_config = _build_sft_config(cfg)
    trainer_cls = WeightedSFTTrainer if weights_enabled else SFTTrainer
    trainer = trainer_cls(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    if weights_enabled:
        # Replace the trainer's data collator with our wrapper so the
        # `weight` column survives batching as a tensor instead of being
        # silently dropped.
        trainer.data_collator = WeightedDataCollator(trainer.data_collator)

    train_result = trainer.train()
    log.info("training metrics: %s", train_result.metrics)

    out_dir = sft_config.output_dir
    log.info("saving final adapter + tokenizer to %s", out_dir)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    final_eval = trainer.evaluate()
    log.info("final eval metrics: %s", final_eval)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning for the ResolveAI complaint classifier.",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to training_config.yaml (default: {DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and data but skip model load and training.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config))
    log.info("loaded config from %s", args.config)
    log.info(
        "base_model=%s | LoRA r=%d alpha=%d | epochs=%d | lr=%g | batch=%d x %d",
        cfg.model["base_model"],
        cfg.lora["r"],
        cfg.lora["alpha"],
        cfg.training["num_train_epochs"],
        cfg.training["learning_rate"],
        cfg.training["per_device_train_batch_size"],
        cfg.training["gradient_accumulation_steps"],
    )
    sys.exit(_train(cfg, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
