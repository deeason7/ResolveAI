"""
Format teacher labels into TRL-ready SFT conversations.

Reads `complaint_labels` joined with `complaints`, emits one JSONL record
per training example in HuggingFace `messages` format, and writes three
files — train/val/test — stratified by sentiment so every split sees
all three classes proportionally.

Output schema (one record per line):

    {
      "messages": [
        {"role": "system", "content": "<short system prompt>"},
        {"role": "user", "content": "COMPLAINT: ...\\nPRODUCT: ...\\n..."},
        {"role": "assistant", "content": "{\\"sentiment\\": ...}"}
      ]
    }

Usage (inside the api container):

    PYTHONPATH=/app python /fine_tuning/02_format_training_data.py
    PYTHONPATH=/app python /fine_tuning/02_format_training_data.py --label-source groq:llama-3.3-70b-versatile

Exit codes:
    0  — finished cleanly
    2  — fatal misconfiguration (no labels found, invalid ratios)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND_DIR = Path("/app")
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
else:
    _local = Path(__file__).resolve().parent.parent / "backend"
    if _local.exists():
        sys.path.insert(0, str(_local))

from sqlmodel import select  # noqa: E402

from app.database import AsyncSessionLocal  # noqa: E402
from app.models.complaint import Complaint  # noqa: E402
from app.models.complaint_label import ComplaintLabel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("format_training_data")

DEFAULT_LABEL_SOURCE = "groq:llama-3.3-70b-versatile"
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "formatted"

SYSTEM_PROMPT = (
    "You are a financial complaint classifier. "
    "Analyze the complaint and output a structured JSON classification."
)


def _build_user_prompt(c: Complaint) -> str:
    parts = [f"COMPLAINT: {c.narrative.strip()}"]
    if c.product:
        parts.append(f"PRODUCT: {c.product}")
    if c.issue:
        parts.append(f"ISSUE: {c.issue}")
    if c.company:
        parts.append(f"COMPANY: {c.company}")
    return "\n".join(parts)


def _build_assistant_message(label: ComplaintLabel) -> str:
    """Compact JSON; keys ordered to match the schema for readability."""
    payload = {
        "sentiment": label.sentiment,
        "intent": label.intent,
        "urgency": label.urgency,
        "key_entities": label.key_entities,
        "reasoning": label.reasoning,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_record(complaint: Complaint, label: ComplaintLabel) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(complaint)},
            {"role": "assistant", "content": _build_assistant_message(label)},
        ]
    }


async def _fetch_pairs(label_source: str) -> list[tuple[Complaint, ComplaintLabel]]:
    """Pull every (complaint, label) pair for the given source."""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Complaint, ComplaintLabel)
            .join(ComplaintLabel, ComplaintLabel.complaint_id == Complaint.id)
            .where(ComplaintLabel.label_source == label_source)
            .order_by(Complaint.id)
        )
        result = await session.execute(stmt)
        return list(result.all())


def _stratified_split(
    pairs: list[tuple[Complaint, ComplaintLabel]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Split by sentiment so each class is represented proportionally.

    Within each sentiment bucket we shuffle once with the seed, then take
    the first floor(N * test_ratio) for test, next floor(N * val_ratio)
    for val, the rest for train. Deterministic across runs given the
    same input ordering + seed.
    """
    buckets: dict[str, list] = defaultdict(list)
    for c, l in pairs:
        buckets[l.sentiment].append((c, l))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for sentiment, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
        log.info(
            "sentiment=%s total=%d → train=%d val=%d test=%d",
            sentiment,
            n,
            n - n_test - n_val,
            n_val,
            n_test,
        )

    # Reshuffle the concatenated splits so sentiment isn't clustered in
    # the file (the trainer benefits from interleaved classes per batch).
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _write_jsonl(
    pairs: list[tuple[Complaint, ComplaintLabel]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c, l in pairs:
            fh.write(json.dumps(_build_record(c, l), ensure_ascii=False) + "\n")
    log.info("wrote %d records → %s", len(pairs), path)


async def _run(args: argparse.Namespace) -> int:
    if not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1):
        log.critical("val_ratio and test_ratio must be in (0, 1)")
        return 2
    if args.val_ratio + args.test_ratio >= 1:
        log.critical("val_ratio + test_ratio must leave room for train")
        return 2

    pairs = await _fetch_pairs(args.label_source)
    if not pairs:
        log.critical("no labels found for source=%s", args.label_source)
        return 2
    log.info("loaded %d (complaint, label) pairs", len(pairs))

    train, val, test = _stratified_split(
        pairs,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val, out_dir / "val.jsonl")
    _write_jsonl(test, out_dir / "test.jsonl")

    log.info(
        "DONE: train=%d val=%d test=%d (total=%d)",
        len(train),
        len(val),
        len(test),
        len(train) + len(val) + len(test),
    )
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Format teacher labels as ChatML training data.")
    p.add_argument(
        "--label-source",
        default=DEFAULT_LABEL_SOURCE,
        help=f"Which labeler's rows to pull (default: {DEFAULT_LABEL_SOURCE}).",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"RNG seed (default: {DEFAULT_SEED}).")
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where to write the JSONL splits (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    code = asyncio.run(_run(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
