"""
Generate teacher labels for the QLoRA fine-tuning dataset.

For each candidate complaint we ask a strong teacher LLM (Llama 3.3 70B
via Groq) to produce a structured `ComplaintClassification`, validated by
instructor against the shared Pydantic schema, and persist both to the
`complaint_labels` table (the source of truth) and to a JSONL artifact
under `fine_tuning/data/labeled/` (for Colab consumption later).

The script is fully idempotent and resumable: at startup we query the
set of complaint_ids already labeled by this source and skip them.
Re-running the script after a crash, a daily-cap exit, or a planned
restart simply continues where the previous run stopped.

Usage (inside the api container):

    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py
    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py --limit 100
    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py --concurrency 3

Exit codes:
    0  — finished cleanly (either hit --limit or labeled every candidate)
    1  — exhausted retries on rate limit (likely Groq daily cap reached)
    2  — fatal misconfiguration (no API key, no rubric file, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Bootstrap so `app.*` is importable both inside the api container
# (where /app is the backend root) and from a host venv (where backend/
# sits next to fine_tuning/).
_BACKEND_DIR = Path("/app")
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
else:
    _local = Path(__file__).resolve().parent.parent / "backend"
    if _local.exists():
        sys.path.insert(0, str(_local))

import instructor  # noqa: E402
from openai import AsyncOpenAI, RateLimitError  # noqa: E402
from sqlmodel import select  # noqa: E402
from tenacity import (  # noqa: E402
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings  # noqa: E402
from app.database import AsyncSessionLocal  # noqa: E402
from app.models.complaint import Complaint  # noqa: E402
from app.models.complaint_label import ComplaintLabel  # noqa: E402
from app.schemas.classification import ComplaintClassification  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_labels")

# Keep the run log readable — SQLAlchemy's engine echo is meant for app
# debugging, not a multi-day batch job.
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_CONCURRENCY = 3  # well under Groq's 30 rpm to leave headroom for retries
DEFAULT_LIMIT = 10_000
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MIN_NARRATIVE_CHARS = 50
# Groq surfaces two distinct 429s: per-minute (recoverable via backoff) and
# per-day (won't recover today — exit and let cron retry tomorrow).
DAILY_CAP_PHRASES = ("tokens per day", "(tpd)", "rpd")
CHUNK_MULTIPLIER = 4  # micro-batch size = concurrency × this; controls how
#                       often we check the cap flag between gather calls


class DailyCapReached(Exception):
    """Raised when Groq returns a 'tokens per day' 429 — don't retry today."""


def _is_daily_cap_error(err: BaseException) -> bool:
    """Walk the cause/context chain looking for the TPD signature.

    instructor and the openai client both wrap the original RateLimitError
    into their own exception types, so an `isinstance(err, RateLimitError)`
    check fires on a fraction of cases. The TPD wording is invariant
    across all wrappers — that's what we key on.
    """
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).lower()
        if any(phrase in msg for phrase in DAILY_CAP_PHRASES):
            return True
        cur = cur.__cause__ or cur.__context__
    return False

RUBRIC_PATH = Path(__file__).parent / "prompts" / "labeling_prompt.md"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "labeled" / "gold_labels.jsonl"


def _build_user_prompt(c: Complaint) -> str:
    parts = [f"COMPLAINT: {c.narrative.strip()}"]
    if c.product:
        parts.append(f"PRODUCT: {c.product}")
    if c.issue:
        parts.append(f"ISSUE: {c.issue}")
    if c.company:
        parts.append(f"COMPANY: {c.company}")
    return "\n".join(parts)


async def _load_already_labeled(label_source: str) -> set[str]:
    """Return UUIDs (as strings) already labeled by this source."""
    async with AsyncSessionLocal() as session:
        stmt = select(ComplaintLabel.complaint_id).where(
            ComplaintLabel.label_source == label_source
        )
        result = await session.execute(stmt)
        return {str(row) for row in result.scalars().all()}


async def _fetch_candidates(
    limit: int,
    exclude: set[str],
) -> list[Complaint]:
    """Pull up to `limit` complaints not yet labeled by this source.

    UUID v4 primary keys are uniformly distributed so `ORDER BY id` gives
    a deterministic, fair sample — re-running pulls the same complaints
    in the same order.
    """
    # Over-fetch to compensate for exclusions, then trim. Capped to avoid
    # pulling the whole table for an unlucky run.
    fetch_size = min(limit + len(exclude), limit * 3 + 100)
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Complaint)
            .where(Complaint.narrative.is_not(None))
            .order_by(Complaint.id)
            .limit(fetch_size)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    picked: list[Complaint] = []
    for c in rows:
        if str(c.id) in exclude:
            continue
        if not c.narrative or len(c.narrative) < MIN_NARRATIVE_CHARS:
            continue
        picked.append(c)
        if len(picked) >= limit:
            break
    return picked


def _make_client() -> instructor.AsyncInstructor:
    if not settings.groq_api_key:
        log.critical("GROQ_API_KEY is empty. Set it in .env and restart.")
        sys.exit(2)
    raw = AsyncOpenAI(api_key=settings.groq_api_key, base_url=GROQ_BASE_URL)
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def _label_one(
    client: instructor.AsyncInstructor,
    model: str,
    rubric: str,
    complaint: Complaint,
) -> tuple[ComplaintClassification, dict]:
    """Single Groq call. Returns the parsed result + operational metadata.

    Raises DailyCapReached (NOT RateLimitError) when the 429 is the
    tokens-per-day variant, so tenacity stops retrying immediately and
    the orchestrator can short-circuit the remaining work.
    """
    started = time.monotonic()
    try:
        result, raw = await client.chat.completions.create_with_completion(
            model=model,
            response_model=ComplaintClassification,
            max_retries=2,  # instructor's own JSON-repair loop, separate from tenacity
            messages=[
                {"role": "system", "content": rubric},
                {"role": "user", "content": _build_user_prompt(complaint)},
            ],
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 — type-agnostic TPD detection
        if _is_daily_cap_error(e):
            raise DailyCapReached(str(e)) from e
        raise
    latency_ms = int((time.monotonic() - started) * 1000)
    usage = getattr(raw, "usage", None)
    meta = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "latency_ms": latency_ms,
    }
    return result, meta


_jsonl_lock = asyncio.Lock()


async def _persist(
    complaint: Complaint,
    label_source: str,
    classification: ComplaintClassification,
    meta: dict,
    jsonl_path: Path,
) -> None:
    """Write to complaint_labels (source of truth) and append JSONL.

    The two writes are NOT in one transaction. The DB is authoritative;
    if the JSONL append fails the row still exists and a future
    `--rebuild-jsonl-from-db` (not implemented yet) can reconstruct.
    """
    row = ComplaintLabel(
        complaint_id=complaint.id,
        label_source=label_source,
        sentiment=classification.sentiment,
        intent=classification.intent,
        urgency=classification.urgency,
        key_entities=[e.model_dump() for e in classification.key_entities],
        reasoning=classification.reasoning,
        input_tokens=meta.get("input_tokens"),
        output_tokens=meta.get("output_tokens"),
        latency_ms=meta.get("latency_ms"),
    )
    async with AsyncSessionLocal() as session:
        session.add(row)
        await session.commit()

    record = {
        "complaint_id": str(complaint.id),
        "cfpb_complaint_id": complaint.cfpb_complaint_id,
        "narrative": complaint.narrative,
        "product": complaint.product,
        "issue": complaint.issue,
        "company": complaint.company,
        "label_source": label_source,
        "classification": classification.model_dump(),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "latency_ms": meta.get("latency_ms"),
        "labeled_at": datetime.utcnow().isoformat(),
    }
    # Lock serializes concurrent writers — POSIX O_APPEND atomicity only
    # holds for writes <= PIPE_BUF (~4 KB), and complaint narratives can
    # exceed that.
    line = json.dumps(record, ensure_ascii=False) + "\n"
    async with _jsonl_lock:
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line)


async def _run(args: argparse.Namespace) -> int:
    if not RUBRIC_PATH.exists():
        log.critical("Rubric not found at %s", RUBRIC_PATH)
        return 2
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    log.info("loaded rubric (%d chars) from %s", len(rubric), RUBRIC_PATH)

    label_source = f"groq:{args.model}"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already = await _load_already_labeled(label_source)
    log.info(
        "label_source=%s already_labeled=%d target_new=%d",
        label_source,
        len(already),
        args.limit,
    )

    candidates = await _fetch_candidates(args.limit, already)
    if not candidates:
        log.info("nothing to label — exiting")
        return 0
    log.info("fetched %d candidates", len(candidates))

    client = _make_client()
    sem = asyncio.Semaphore(args.concurrency)
    cap_event = asyncio.Event()
    counters = {"ok": 0, "rate_limited": 0, "other": 0, "tokens_in": 0, "tokens_out": 0}
    started = time.monotonic()

    async def process(c: Complaint) -> None:
        if cap_event.is_set():
            return
        async with sem:
            if cap_event.is_set():
                return
            try:
                result, meta = await _label_one(client, args.model, rubric, c)
            except DailyCapReached as e:
                cap_event.set()
                log.warning("daily token cap reached on %s: %s", c.id, str(e)[:200])
                return
            except Exception as e:  # noqa: BLE001 — single funnel, decide by content
                if _is_daily_cap_error(e):
                    cap_event.set()
                    log.warning("daily token cap reached on %s (wrapped)", c.id)
                    return
                if isinstance(e, RateLimitError):
                    counters["rate_limited"] += 1
                    log.warning("rate-limit on %s: %s", c.id, str(e)[:200])
                    return
                counters["other"] += 1
                log.warning("labeling failed for %s: %s", c.id, str(e)[:200])
                return
            try:
                await _persist(c, label_source, result, meta, output_path)
            except Exception as e:  # noqa: BLE001
                counters["other"] += 1
                log.exception("persistence failed for %s: %s", c.id, e)
                return
            counters["ok"] += 1
            counters["tokens_in"] += meta.get("input_tokens") or 0
            counters["tokens_out"] += meta.get("output_tokens") or 0
            if counters["ok"] % 10 == 0 or counters["ok"] == 1:
                elapsed = time.monotonic() - started
                rate = counters["ok"] / elapsed if elapsed else 0
                log.info(
                    "progress: ok=%d rate_limited=%d other_errors=%d "
                    "tokens_in=%d tokens_out=%d elapsed=%.1fs rate=%.1f/s",
                    counters["ok"],
                    counters["rate_limited"],
                    counters["other"],
                    counters["tokens_in"],
                    counters["tokens_out"],
                    elapsed,
                    rate,
                )

    # Process in micro-batches so the cap flag is checked frequently,
    # not just after all 10K tasks have either succeeded or burned through
    # their retry budgets.
    chunk_size = args.concurrency * CHUNK_MULTIPLIER
    for i in range(0, len(candidates), chunk_size):
        if cap_event.is_set():
            break
        chunk = candidates[i : i + chunk_size]
        await asyncio.gather(*(process(c) for c in chunk))

    elapsed = time.monotonic() - started
    log.info(
        "DONE: ok=%d rate_limited=%d other_errors=%d "
        "tokens_in=%d tokens_out=%d in %.1fs",
        counters["ok"],
        counters["rate_limited"],
        counters["other"],
        counters["tokens_in"],
        counters["tokens_out"],
        elapsed,
    )

    if cap_event.is_set():
        log.warning(
            "exited early due to Groq daily token cap — re-run after the "
            "cap resets (Groq surfaces a UTC reset window in the 429 message)"
        )
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label complaints with Groq teacher LLM.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max new labels to attempt this run (default: {DEFAULT_LIMIT}).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"In-flight Groq requests (default: {DEFAULT_CONCURRENCY}).",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSONL output path (default: {DEFAULT_OUTPUT}).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    code = asyncio.run(_run(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
