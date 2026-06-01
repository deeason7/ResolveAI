"""
Smoke test the deployed Ollama model.

Sends a complaint to the local Ollama instance via the OpenAI-compatible
API, asks for structured output through ``instructor``, validates against
the shared :class:`ComplaintClassification` schema, and prints the result
with timing. Catches the "the model loads but never produces valid JSON"
failure mode before you wire it into the backend.

The user prompt matches the training-time format exactly (COMPLAINT /
PRODUCT / ISSUE / COMPANY block — see ``02_format_training_data.py``)
so this script measures the same distribution the model was trained on.

Usage:
    # Single complaint via stdin
    echo "my credit card was charged twice for the same purchase" | \\
        python fine_tuning/06_smoke_test.py

    # Single complaint via flag
    python fine_tuning/06_smoke_test.py \\
        --complaint "my mortgage company won't fix an escrow error" \\
        --product "Mortgage" \\
        --issue "Loan servicing, payments, escrow account" \\
        --company "Wells Fargo"

    # Pull a random complaint from the held-out test split
    python fine_tuning/06_smoke_test.py --from-test-set

    # Batch: run N samples from the test split, report % valid JSON +
    # mean/p95 latency
    python fine_tuning/06_smoke_test.py --batch 25

    # Point at a different model name (default: resolveai-sentiment)
    python fine_tuning/06_smoke_test.py --model resolveai-sentiment-v2

Exit codes:
    0 — at least one complaint classified successfully
    2 — fatal config error (Ollama unreachable, schema import fails, etc.)
    3 — batch mode: success rate fell below threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Locate the backend so we can import the shared classification schema.
# Mirrors the path-resolution pattern used by 02_format_training_data.py
# so this works inside the api container, from repo root, or from any
# cwd.
_BACKEND_CONTAINER = Path("/app")
_BACKEND_LOCAL = SCRIPT_DIR.parent / "backend"
for _candidate in (_BACKEND_CONTAINER, _BACKEND_LOCAL):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "resolveai-sentiment"
DEFAULT_TEST_PATH = SCRIPT_DIR / "data" / "formatted" / "test.jsonl"
DEFAULT_BATCH_SAMPLE_SEED = 17
# Match the training-time system prompt verbatim. Lives in two places —
# 02_format_training_data.SYSTEM_PROMPT and 05_export_gguf.TRAINING_SYSTEM_PROMPT.
# Don't drift this string without updating both, then re-running this script.
TRAINING_SYSTEM_PROMPT = (
    "You are a financial complaint classifier. "
    "Analyze the complaint and output a structured JSON classification."
)
# Threshold: in batch mode, fall below this fraction of valid-JSON and
# exit code 3. 95% matches the spec's structured-output reliability target.
DEFAULT_VALID_JSON_THRESHOLD = 0.95


def build_user_prompt(
    complaint: str,
    product: str | None = None,
    issue: str | None = None,
    company: str | None = None,
) -> str:
    """Build the same COMPLAINT/PRODUCT/ISSUE/COMPANY block used at training.

    Order and labels MUST match 02_format_training_data._build_user_prompt
    so the deployed model sees the distribution it was trained on. Fields
    other than COMPLAINT are optional — empty values are skipped, not
    blank-filled, because the trainer also skipped them.
    """
    parts = [f"COMPLAINT: {complaint.strip()}"]
    if product:
        parts.append(f"PRODUCT: {product}")
    if issue:
        parts.append(f"ISSUE: {issue}")
    if company:
        parts.append(f"COMPANY: {company}")
    return "\n".join(parts)


def load_test_sample(test_path: Path, seed: int) -> dict:
    """Return one random (complaint, classification) pair from the test split.

    The JSONL records are in HuggingFace messages format — we extract the
    user content (the COMPLAINT/PRODUCT/... block) and the assistant
    content (the gold JSON) for comparison.
    """
    if not test_path.is_file():
        raise FileNotFoundError(f"test split not found at {test_path}")
    rng = random.Random(seed)
    with test_path.open("r", encoding="utf-8") as fh:
        # Reservoir sample of size 1 — read once, no need to materialize
        # all 995 records.
        chosen = None
        for i, line in enumerate(fh):
            if rng.randrange(i + 1) == 0:
                chosen = json.loads(line)
    if chosen is None:
        raise ValueError(f"test split is empty: {test_path}")

    messages = chosen["messages"]
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
    if user_msg is None or asst_msg is None:
        raise ValueError("test record missing user or assistant message")

    return {
        "user_prompt": user_msg["content"],
        "gold": json.loads(asst_msg["content"]),
    }


def format_result(prediction: dict, gold: dict | None, latency_ms: float) -> str:
    """Pretty-print the classification with optional gold-vs-prediction diff.

    When ``gold`` is provided (from --from-test-set or --batch), we show
    a side-by-side summary so you can eyeball whether the deployed model
    matches the trained model's expected output.
    """
    out: list[str] = []
    out.append(f"Classification (latency: {latency_ms:.0f} ms):")
    out.append(json.dumps(prediction, indent=2, ensure_ascii=False))
    if gold is not None:
        sentiment_match = prediction.get("sentiment") == gold.get("sentiment")
        intent_match = prediction.get("intent") == gold.get("intent")
        urgency_diff = abs(int(prediction.get("urgency", 0)) - int(gold.get("urgency", 0)))
        out.append("")
        out.append(
            f"vs gold: sentiment={'✓' if sentiment_match else '✗'} "
            f"intent={'✓' if intent_match else '✗'} urgency_Δ={urgency_diff}"
        )
    return "\n".join(out)


def _classify_one(client, model_name: str, user_prompt: str, schema_cls):
    """Single Ollama call returning (parsed_model, latency_ms).

    Wrapping instructor's call here so the batch loop and the single-shot
    path share retry / timing semantics.
    """
    t0 = time.perf_counter()
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_model=schema_cls,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


def _read_stdin_complaint() -> str | None:
    """Return stdin contents stripped, or None if stdin is a tty."""
    if sys.stdin.isatty():
        return None
    data = sys.stdin.read().strip()
    return data or None


def _resolve_complaint(args: argparse.Namespace) -> tuple[str, dict | None]:
    """Resolve the user prompt + (optional) gold classification from CLI args.

    Priority:
      1. --complaint (with optional --product / --issue / --company)
      2. --from-test-set (random pick from test split)
      3. stdin pipe
    """
    if args.complaint:
        prompt = build_user_prompt(args.complaint, args.product, args.issue, args.company)
        return prompt, None
    if args.from_test_set:
        sample = load_test_sample(Path(args.test_path), args.seed)
        return sample["user_prompt"], sample["gold"]
    piped = _read_stdin_complaint()
    if piped:
        return build_user_prompt(piped), None
    return "", None


def _run_single(args: argparse.Namespace) -> int:
    """Single-complaint mode. Calls Ollama once, prints the result."""
    user_prompt, gold = _resolve_complaint(args)
    if not user_prompt:
        log.critical(
            "no complaint provided. Use --complaint TEXT, --from-test-set, "
            "or pipe text via stdin.",
        )
        return 2

    client, schema_cls = _make_client(args)
    try:
        result, latency_ms = _classify_one(client, args.model, user_prompt, schema_cls)
    except Exception as exc:  # noqa: BLE001 — surface any failure cleanly
        log.critical("classification failed: %s: %s", type(exc).__name__, exc)
        return 2

    prediction = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    print(format_result(prediction, gold, latency_ms))
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    """Batch mode. Pulls N samples from the test split, reports aggregates."""
    test_path = Path(args.test_path)
    if not test_path.is_file():
        log.critical("test split not found at %s", test_path)
        return 2

    rng = random.Random(args.seed)
    with test_path.open("r", encoding="utf-8") as fh:
        all_records = [json.loads(line) for line in fh]
    if len(all_records) < args.batch:
        log.warning(
            "test split has %d records but --batch=%d requested; using all",
            len(all_records),
            args.batch,
        )
    sample = rng.sample(all_records, k=min(args.batch, len(all_records)))

    client, schema_cls = _make_client(args)

    n_valid = 0
    sentiment_correct = 0
    intent_correct = 0
    latencies: list[float] = []

    for i, record in enumerate(sample, 1):
        messages = record["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        gold = json.loads(next(m for m in messages if m["role"] == "assistant")["content"])
        try:
            result, latency_ms = _classify_one(
                client, args.model, user_msg["content"], schema_cls
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%d/%d] failed: %s", i, len(sample), exc)
            continue
        n_valid += 1
        latencies.append(latency_ms)
        pred = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        if pred.get("sentiment") == gold.get("sentiment"):
            sentiment_correct += 1
        if pred.get("intent") == gold.get("intent"):
            intent_correct += 1
        if i % 5 == 0 or i == len(sample):
            log.info(
                "[%d/%d] valid=%d sent=%d intent=%d",
                i,
                len(sample),
                n_valid,
                sentiment_correct,
                intent_correct,
            )

    total = len(sample)
    valid_rate = n_valid / total if total else 0.0
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    print("")
    print(f"Smoke-test summary ({total} samples):")
    print(f"  valid JSON: {n_valid}/{total} ({valid_rate * 100:.1f}%)")
    if n_valid:
        print(
            f"  sentiment top-1: {sentiment_correct}/{n_valid} "
            f"({sentiment_correct / n_valid * 100:.1f}% on valid)"
        )
        print(
            f"  intent top-1:    {intent_correct}/{n_valid} "
            f"({intent_correct / n_valid * 100:.1f}% on valid)"
        )
        print(f"  latency p50: {p50:.0f} ms / p95: {p95:.0f} ms")

    if valid_rate < args.valid_json_threshold:
        log.error(
            "valid JSON rate %.1f%% < threshold %.0f%% — investigate before integrating",
            valid_rate * 100,
            args.valid_json_threshold * 100,
        )
        return 3
    return 0


def _make_client(args: argparse.Namespace):
    """Construct an instructor-wrapped OpenAI-compatible client.

    Returns (client, schema_class). Lazily imports so --help and the
    import-error path don't require these deps installed.
    """
    try:
        import instructor
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency: {exc.name}. "
            f"Install with: pip install instructor openai"
        ) from exc

    try:
        from app.schemas.classification import ComplaintClassification
    except ImportError as exc:
        raise SystemExit(
            f"can't import ComplaintClassification: {exc}. "
            f"Run from repo root or set PYTHONPATH=/app inside the api container."
        ) from exc

    base = OpenAI(base_url=args.base_url, api_key="ollama")  # noqa: S106 — Ollama ignores the key
    return instructor.from_openai(base, mode=instructor.Mode.JSON), ComplaintClassification


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test the deployed Ollama classifier.")
    p.add_argument("--complaint", help="Complaint narrative.")
    p.add_argument("--product", help="Optional product field (e.g., 'Mortgage').")
    p.add_argument("--issue", help="Optional issue field.")
    p.add_argument("--company", help="Optional company field.")
    p.add_argument(
        "--from-test-set",
        action="store_true",
        help="Pick a random complaint from the held-out test split.",
    )
    p.add_argument("--batch", type=int, default=0, help="Batch mode: run N samples from test split.")
    p.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL)
    p.add_argument(
        "--valid-json-threshold",
        type=float,
        default=DEFAULT_VALID_JSON_THRESHOLD,
        help="Batch mode: minimum valid-JSON fraction before exit-3 (default: 0.95).",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_BATCH_SAMPLE_SEED)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch > 0:
        sys.exit(_run_batch(args))
    sys.exit(_run_single(args))


if __name__ == "__main__":
    main()
