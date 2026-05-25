"""
Generate teacher labels for the QLoRA fine-tuning dataset.

For each candidate complaint we ask a strong teacher LLM to produce a
structured `ComplaintClassification`, validated by instructor against the
shared Pydantic schema, and persist both to the `complaint_labels` table
(the source of truth) and to a JSONL artifact under
`fine_tuning/data/labeled/` (for Colab consumption later).

Two teacher providers are supported (selected with --provider):
    bedrock — AWS Bedrock Converse API, default model
              us.meta.llama3-3-70b-instruct-v1:0 (cross-region inference
              profile, on-demand serverless, ~$0.72/M tokens in/out).
    groq    — Groq's free tier, default model llama-3.3-70b-versatile.
              Subject to a 100K-tokens/day cap; the run fast-exits with
              code 1 when the cap is hit so cron can resume tomorrow.

The script is fully idempotent and resumable: at startup we query the
set of complaint_ids already labeled by this source and skip them.
Re-running the script after a crash, a daily-cap exit, or a planned
restart simply continues where the previous run stopped.

Usage (inside the api container):

    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py
    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py --limit 100
    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py --provider bedrock --concurrency 5
    PYTHONPATH=/app python /fine_tuning/01_prepare_labels.py --provider groq --concurrency 3

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
    retry_if_exception,
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

# Keep the run log readable — SQLAlchemy's `echo=True` (set when
# ENVIRONMENT=development) explicitly forces `sqlalchemy.engine.Engine`
# to INFO. That explicit level on the child wins over any level we set
# on the parent `sqlalchemy` logger, so we have to silence the leaf
# loggers by name. We also flip echo off on the engine itself for
# belt-and-braces — same effect, different vector.
for _name in (
    "sqlalchemy",
    "sqlalchemy.engine",
    "sqlalchemy.engine.Engine",
    "sqlalchemy.pool",
    "sqlalchemy.dialects",
    "openai",
    "httpx",
    "botocore",
    "boto3",
    "urllib3",
):
    logging.getLogger(_name).setLevel(logging.WARNING)
try:
    from app.database import engine as _db_engine

    _db_engine.echo = False
except Exception:  # noqa: BLE001 — best-effort, the loggers above already cover it
    pass

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
BEDROCK_DEFAULT_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"
DEFAULT_PROVIDER = "bedrock"
# Groq throttles aggressively (30 rpm free tier), Bedrock on-demand starts
# at hundreds of rpm — so the safe concurrency floor differs by provider.
DEFAULT_CONCURRENCY_BY_PROVIDER = {"groq": 3, "bedrock": 5}
DEFAULT_LIMIT = 10_000
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MIN_NARRATIVE_CHARS = 50
# Groq surfaces two distinct 429s: per-minute (recoverable via backoff) and
# per-day (won't recover today — exit and let cron retry tomorrow). Bedrock
# on-demand has no equivalent daily cap, so this only fires for Groq.
DAILY_CAP_PHRASES = ("tokens per day", "(tpd)", "rpd")
# botocore error codes that indicate "back off and retry" rather than "give up"
BEDROCK_THROTTLE_CODES = (
    "ThrottlingException",
    "ServiceQuotaExceededException",
    "TooManyRequestsException",
    "ModelTimeoutException",
)
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


def _make_client(provider: str):
    """Build an instructor client for the chosen provider.

    Returns a tuple (client, is_async). The Groq path uses AsyncOpenAI under
    the hood (native async). The Bedrock path wraps a sync boto3 client —
    we'll trampoline calls through asyncio.to_thread to keep the event loop
    free during the HTTP round-trip.
    """
    if provider == "groq":
        if not settings.groq_api_key:
            log.critical("GROQ_API_KEY is empty. Set it in .env and restart.")
            sys.exit(2)
        raw = AsyncOpenAI(api_key=settings.groq_api_key, base_url=GROQ_BASE_URL)
        return instructor.from_openai(raw, mode=instructor.Mode.JSON), True
    if provider == "bedrock":
        try:
            import boto3
        except ImportError:
            log.critical(
                "boto3 not installed. Rebuild the api image after the latest "
                "pyproject.toml change (`docker compose build api`)."
            )
            sys.exit(2)
        # boto3 picks creds from ~/.aws/credentials (mounted into the
        # container) or AWS_* env vars; we only specify the region.
        bedrock = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        # BEDROCK_JSON over BEDROCK_TOOLS: Llama 3.x on Bedrock supports
        # tool use but rejects `toolConfig.toolChoice.tool` (forced selection),
        # which is what instructor's BEDROCK_TOOLS mode emits. JSON mode
        # injects the schema into the prompt and parses the text response —
        # less rigorous but works on every Converse-API model.
        mode = getattr(instructor.Mode, "BEDROCK_JSON", None)
        if mode is None:
            log.critical(
                "instructor build lacks Bedrock JSON mode. Upgrade `instructor` "
                "to a version with Bedrock support (>= 1.5)."
            )
            sys.exit(2)
        return instructor.from_bedrock(bedrock, mode=mode), False
    log.critical("unknown provider: %s", provider)
    sys.exit(2)


def _is_bedrock_throttle(err: BaseException) -> bool:
    """Recognize botocore client errors that mean 'slow down, try again'."""
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        return False
    if not isinstance(err, ClientError):
        return False
    code = err.response.get("Error", {}).get("Code", "")
    return code in BEDROCK_THROTTLE_CODES


def _is_retryable(err: BaseException) -> bool:
    """Tenacity predicate covering both providers' 'retry-me' shapes."""
    if isinstance(err, RateLimitError):
        return True
    return _is_bedrock_throttle(err)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def _label_one(
    client,
    is_async: bool,
    provider: str,
    model: str,
    rubric: str,
    complaint: Complaint,
) -> tuple[ComplaintClassification, dict]:
    """Single teacher call. Returns the parsed result + operational metadata.

    Raises DailyCapReached (NOT RateLimitError) when the 429 is the
    tokens-per-day variant, so tenacity stops retrying immediately and
    the orchestrator can short-circuit the remaining work.
    """
    user_text = _build_user_prompt(complaint)
    started = time.monotonic()
    try:
        if is_async:
            # OpenAI-compatible async path (Groq)
            result, raw = await client.chat.completions.create_with_completion(
                model=model,
                response_model=ComplaintClassification,
                max_retries=2,
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
            )
        else:
            # Bedrock Converse path — sync client trampolined off the loop.
            # Converse accepts top-level `system=[{"text": ...}]` separately
            # from `messages`; instructor's adapter handles the translation
            # when we pass a system-role message in the OpenAI shape.
            result, raw = await asyncio.to_thread(
                client.chat.completions.create_with_completion,
                modelId=model,
                response_model=ComplaintClassification,
                max_retries=2,
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": user_text},
                ],
                inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
            )
    except Exception as e:  # noqa: BLE001 — type-agnostic TPD detection
        if _is_daily_cap_error(e):
            raise DailyCapReached(str(e)) from e
        raise
    latency_ms = int((time.monotonic() - started) * 1000)
    meta = {
        "input_tokens": _extract_input_tokens(raw, provider),
        "output_tokens": _extract_output_tokens(raw, provider),
        "latency_ms": latency_ms,
    }
    return result, meta


def _extract_input_tokens(raw, provider: str) -> int | None:
    """Token-usage field names differ across providers; normalize here."""
    if raw is None:
        return None
    if provider == "groq":
        usage = getattr(raw, "usage", None)
        return getattr(usage, "prompt_tokens", None) if usage else None
    # Bedrock Converse returns {"usage": {"inputTokens": N, "outputTokens": M, ...}}
    if isinstance(raw, dict):
        return raw.get("usage", {}).get("inputTokens")
    # instructor may return an object with .usage attribute regardless
    usage = getattr(raw, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("inputTokens") or usage.get("prompt_tokens")
    return getattr(usage, "inputTokens", None) or getattr(usage, "prompt_tokens", None)


def _extract_output_tokens(raw, provider: str) -> int | None:
    if raw is None:
        return None
    if provider == "groq":
        usage = getattr(raw, "usage", None)
        return getattr(usage, "completion_tokens", None) if usage else None
    if isinstance(raw, dict):
        return raw.get("usage", {}).get("outputTokens")
    usage = getattr(raw, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("outputTokens") or usage.get("completion_tokens")
    return getattr(usage, "outputTokens", None) or getattr(usage, "completion_tokens", None)


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

    label_source = f"{args.provider}:{args.model}"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already = await _load_already_labeled(label_source)
    log.info(
        "provider=%s model=%s label_source=%s already_labeled=%d target_new=%d concurrency=%d",
        args.provider,
        args.model,
        label_source,
        len(already),
        args.limit,
        args.concurrency,
    )

    candidates = await _fetch_candidates(args.limit, already)
    if not candidates:
        log.info("nothing to label — exiting")
        return 0
    log.info("fetched %d candidates", len(candidates))

    client, is_async = _make_client(args.provider)
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
                result, meta = await _label_one(
                    client, is_async, args.provider, args.model, rubric, c
                )
            except DailyCapReached as e:
                cap_event.set()
                log.warning("daily token cap reached on %s: %s", c.id, str(e)[:200])
                return
            except Exception as e:  # noqa: BLE001 — single funnel, decide by content
                if _is_daily_cap_error(e):
                    cap_event.set()
                    log.warning("daily token cap reached on %s (wrapped)", c.id)
                    return
                if _is_retryable(e):
                    counters["rate_limited"] += 1
                    log.warning("throttle on %s: %s", c.id, str(e)[:200])
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
        "DONE: ok=%d rate_limited=%d other_errors=%d tokens_in=%d tokens_out=%d in %.1fs",
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
    p = argparse.ArgumentParser(description="Label complaints with a teacher LLM.")
    p.add_argument(
        "--provider",
        choices=("groq", "bedrock"),
        default=DEFAULT_PROVIDER,
        help=f"Teacher provider (default: {DEFAULT_PROVIDER}).",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier. Defaults: groq → "
            f"{GROQ_DEFAULT_MODEL}; bedrock → {BEDROCK_DEFAULT_MODEL}."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max new labels to attempt this run (default: {DEFAULT_LIMIT}).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "In-flight requests. Defaults: groq → "
            f"{DEFAULT_CONCURRENCY_BY_PROVIDER['groq']}; bedrock → "
            f"{DEFAULT_CONCURRENCY_BY_PROVIDER['bedrock']}."
        ),
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSONL output path (default: {DEFAULT_OUTPUT}).",
    )
    args = p.parse_args()
    if args.model is None:
        args.model = BEDROCK_DEFAULT_MODEL if args.provider == "bedrock" else GROQ_DEFAULT_MODEL
    if args.concurrency is None:
        args.concurrency = DEFAULT_CONCURRENCY_BY_PROVIDER[args.provider]
    return args


def main() -> None:
    args = _parse_args()
    code = asyncio.run(_run(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
