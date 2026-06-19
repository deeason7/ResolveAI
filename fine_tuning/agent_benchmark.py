"""
Standalone benchmark for the resolution agent + four-layer guardrail engine.

Runs the *real* ``ResolutionAgent`` and ``GuardrailEngine`` over a batch of
escalation-worthy CFPB complaints and measures what the offline classifier eval
never produced: guardrail pass rate, regeneration behavior, per-layer violation
breakdown, LLM-as-judge tone scores, and cost / latency per resolution.

Why standalone (no Docker, no DB, no live stack): the agent does no database
I/O — it pulls grounding context through injected stores and returns an
``AgentResult`` the worker would persist. So the entire draft -> validate ->
regenerate loop runs with:

  * stubbed vector / graph stores (an 8 GB laptop can't host Qdrant + Neo4j +
    Postgres + the rest at once), with the graph still serving the real seed
    regulations so Layer 3 (citation grounding) is a genuine check, not a no-op,
  * the real ``GuardrailEngine`` (all four layers, including the live LLM judge),
  * Groq as the sole provider (``LLM_SKIP_LOCAL=true``) for both drafting and the
    judge — the fine-tuned model is a *classifier*, not a drafting model, so the
    agent's generation path is cloud-served in production regardless of hardware.

Classification input defaults to the held-out gold labels: teacher-quality,
reproducible, and it keeps the run under Groq's free-tier daily token cap (the
agent's own draft + judge calls are what we're measuring here). Pass
``--classify-via-groq`` to exercise the full live classify -> agent handoff.

Usage (from the repo root, with the backend venv):
    backend/.venv/bin/python fine_tuning/agent_benchmark.py --dry-run
    backend/.venv/bin/python fine_tuning/agent_benchmark.py --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# --- environment bootstrap: must run before any `app.*` import ---------------
# app.config instantiates Settings() at import time and requires DB / Redis /
# Neo4j / JWT values. We never dial real infrastructure (no DB I/O; the stores
# are stubbed), so dummy connection strings satisfy validation without touching
# anything. Only the Groq key is real. os.environ wins over the repo .env in
# pydantic-settings, so these assignments are authoritative.

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"


def _groq_key_from_env_file() -> str:
    """Pull GROQ_API_KEY out of the repo .env without importing dotenv."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("GROQ_API_KEY=") and not stripped.startswith("#"):
            return stripped.split("=", 1)[1].strip().strip("'\"")
    return ""


_DUMMY_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://benchmark:benchmark@localhost:5432/benchmark",
    "REDIS_URL": "redis://localhost:6379/0",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "benchmark",
    "JWT_SECRET_KEY": "benchmark-only-not-a-secret",
    "JWT_REFRESH_SECRET_KEY": "benchmark-only-not-a-secret",
    "LLM_SKIP_LOCAL": "true",  # drop the local Ollama provider -> Groq is primary
}
os.environ.update(_DUMMY_ENV)
_GROQ_KEY = os.environ.get("GROQ_API_KEY") or _groq_key_from_env_file()
if _GROQ_KEY:
    os.environ["GROQ_API_KEY"] = _GROQ_KEY

sys.path.insert(0, str(BACKEND_DIR))

from app.models.complaint import Complaint  # noqa: E402
from app.schemas.agent import DraftedResponse, DraftResponseInput  # noqa: E402
from app.schemas.classification import ComplaintClassification  # noqa: E402
from app.schemas.guardrails import GuardrailOutcome  # noqa: E402
from app.services.agent import tools as agent_tools  # noqa: E402
from app.services.agent.orchestrator import ResolutionAgent  # noqa: E402
from app.services.classifier import Classifier  # noqa: E402
from app.services.guardrails import GuardrailEngine  # noqa: E402
from app.services.llmops_tracker import estimate_cost_usd  # noqa: E402


def _dummy_embed(text: str) -> list[float]:
    """Stand in for the sentence-transformer so no 90 MB model loads.

    ``search_precedents`` embeds the query before it ever reaches the (stubbed)
    vector store, so patching the embed at the seam keeps the run hermetic.
    """
    return [0.0] * 384


agent_tools.embed_text = _dummy_embed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "openai", "instructor", "sqlalchemy.engine.Engine"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("agent_benchmark")

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_LIMIT = 30
SHUFFLE_SEED = 42
# Groq free tier caps at 12K tokens/min. We pace each complaint by the tokens it
# just spent so the rolling minute stays under this budget (with headroom), since
# instructor's internal retries swallow the 429 before the client's Retry-After
# backoff can act — proactive spacing is the only lever left.
DEFAULT_TPM_BUDGET = 9000.0
RECOVERY_SLEEP_S = 30.0  # after an 'unavailable', let the rate window clear
MAX_PACE_S = 60.0  # never sleep longer than this between complaints
GOLD_PATH = REPO_ROOT / "fine_tuning" / "data" / "labeled" / "gold_labels.jsonl"
REGS_PATH = REPO_ROOT / "graph_seed" / "regulations.json"
OUT_PATH = REPO_ROOT / "fine_tuning" / "results" / "agent_benchmark_metrics.json"
TONE_DIMENSIONS = ("empathy", "professionalism", "actionability")


# --- injected stubs ----------------------------------------------------------


class StubVectors:
    """Vector store that returns no precedents — this run has no Qdrant.

    Signature mirrors ``VectorStore.search_similar`` (called positionally via
    ``asyncio.to_thread`` in the tool), so duck typing is enough.
    """

    def search_similar(
        self, vector: list[float], filters: dict | None = None, limit: int = 5
    ):
        return []


@dataclass(frozen=True)
class _Reg:
    """The few attributes ``lookup_regulations`` reads off a graph regulation."""

    title: str
    cfr_reference: str
    summary: str
    key_provisions: list[str]


class StubGraph:
    """Serves real seed regulations per product; no company profile.

    Matching the seed's ``applies_to`` category against the CFPB product string
    reproduces the Phase 3 ``Product-[:APPLIES_TO]-Regulation`` edge well enough
    to give the drafting model genuine regulations to (correctly or wrongly)
    cite — which is the whole point of exercising Layer 3. Company profiles
    return None: most companies aren't in a freshly seeded graph, and the agent
    is built to degrade on that.
    """

    def __init__(self, regs_raw: list[dict], limit: int = 5) -> None:
        self._regs_raw = regs_raw
        self._limit = limit

    async def get_regulations(
        self, product: str, issue: str | None = None
    ) -> list[_Reg]:
        p = (product or "").lower()
        out: list[_Reg] = []
        for r in self._regs_raw:
            if any(category.lower() in p for category in r.get("applies_to", [])):
                out.append(
                    _Reg(
                        title=r["title"],
                        cfr_reference=r["cfr_reference"],
                        summary=r["summary"],
                        key_provisions=r.get("key_provisions", []),
                    )
                )
                if len(out) >= self._limit:
                    break
        return out

    async def get_company_profile(self, company_name: str):
        return None


class RecordingGuardrail:
    """Wraps the real engine and keeps every validate() outcome.

    The ``AgentResult`` only exposes the *terminal* attempt's violations, but we
    want the per-layer breakdown and tone scores across *all* attempts — so we
    record each call as it passes through.
    """

    def __init__(self, engine: GuardrailEngine) -> None:
        self._engine = engine
        self.outcomes: list[GuardrailOutcome] = []

    async def validate(
        self, draft: DraftedResponse, context: DraftResponseInput
    ) -> GuardrailOutcome:
        outcome = await self._engine.validate(draft, context)
        self.outcomes.append(outcome)
        return outcome


# --- selection ---------------------------------------------------------------


def _is_escalation_worthy(classification: dict) -> bool:
    """Mirror the worker's escalation trigger: extreme tone or high urgency."""
    return (
        classification.get("sentiment") == "extreme_negative"
        or (classification.get("urgency") or 0) >= 4
    )


def load_candidates(limit: int, *, require_company: bool = False) -> list[dict]:
    """Pick escalation-worthy complaints (with a product) from the gold set."""
    records: list[dict] = []
    with GOLD_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            classification = r.get("classification") or {}
            if not r.get("narrative") or not r.get("product"):
                continue
            if not _is_escalation_worthy(classification):
                continue
            if require_company and not r.get("company"):
                continue
            records.append(r)
    random.Random(SHUFFLE_SEED).shuffle(records)
    return records[:limit]


# --- the run -----------------------------------------------------------------


async def run_benchmark(
    records: list[dict], classify_via_groq: bool, tpm_budget: float = DEFAULT_TPM_BUDGET
):
    """Run the agent over each complaint; return per-row data + every outcome."""
    regs_raw = json.loads(REGS_PATH.read_text())
    graph = StubGraph(regs_raw)
    vectors = StubVectors()
    classifier = Classifier() if classify_via_groq else None

    rows: list[dict] = []
    all_outcomes: list[GuardrailOutcome] = []
    consecutive_unavailable = 0

    for i, r in enumerate(records, 1):
        complaint = Complaint(
            narrative=r["narrative"],
            product=r.get("product"),
            issue=r.get("issue"),
            company=r.get("company"),
        )

        classify_cost = 0.0
        if classify_via_groq:
            outcome = await asyncio.to_thread(
                classifier.classify,
                complaint.narrative,
                product=complaint.product,
                issue=complaint.issue,
                company=complaint.company,
            )
            classification = outcome.classification
            classify_cost = estimate_cost_usd(
                outcome.model, outcome.prompt_tokens, outcome.completion_tokens
            )
        else:
            classification = ComplaintClassification(**r["classification"])

        recorder = RecordingGuardrail(GuardrailEngine())
        agent = ResolutionAgent(
            complaint,
            classification,
            vector_store=vectors,
            graph_store=graph,
            guardrails=recorder,
        )

        started = time.perf_counter()
        result = await agent.run()
        wall_ms = int((time.perf_counter() - started) * 1000)

        all_outcomes.extend(recorder.outcomes)
        draft_cost = sum(
            estimate_cost_usd(d.model, d.prompt_tokens, d.completion_tokens)
            for d in result.llm_calls
        )
        judge_cost = sum(
            estimate_cost_usd(j.model, j.prompt_tokens, j.completion_tokens)
            for j in result.judge_calls
        )
        scored = [o.scores for o in recorder.outcomes if o.scores]
        final_scores = scored[-1] if scored else {}
        providers = {d.provider for d in result.llm_calls} | {
            j.provider for j in result.judge_calls
        }
        fell_back = any(d.is_fallback for d in result.llm_calls) or any(
            j.is_fallback for j in result.judge_calls
        )

        rows.append(
            {
                "company": r.get("company"),
                "product": r.get("product"),
                "gold_sentiment": (r.get("classification") or {}).get("sentiment"),
                "gold_urgency": (r.get("classification") or {}).get("urgency"),
                "status": result.status,
                "attempts": result.attempts,
                "wall_ms": wall_ms,
                "draft_calls": len(result.llm_calls),
                "judge_calls": len(result.judge_calls),
                "tone_scores": final_scores,
                "violations": [
                    {"layer": v.layer, "code": v.code}
                    for o in recorder.outcomes
                    for v in o.violations
                ],
                "draft_cost_usd": round(draft_cost, 6),
                "judge_cost_usd": round(judge_cost, 6),
                "classify_cost_usd": round(classify_cost, 6),
                "total_cost_usd": round(draft_cost + judge_cost + classify_cost, 6),
                "providers": sorted(providers),
                "is_fallback": fell_back,
            }
        )

        logger.info(
            "[%d/%d] status=%s attempts=%d wall=%dms draft=%d judge=%d cost=$%.4f scores=%s",
            i,
            len(records),
            result.status,
            result.attempts,
            wall_ms,
            len(result.llm_calls),
            len(result.judge_calls),
            rows[-1]["total_cost_usd"],
            final_scores or "-",
        )

        # A sustained rate cap (per-minute or daily) shows up as repeated
        # "unavailable" — every provider retry failed. No point burning the rest
        # of the batch on it once it's clearly not a transient blip.
        if result.status == "unavailable":
            consecutive_unavailable += 1
            if consecutive_unavailable >= 3:
                logger.error(
                    "3 consecutive 'unavailable' results — Groq rate cap (TPM or "
                    "daily). Stopping early with %d complete.",
                    len(rows),
                )
                break
        else:
            consecutive_unavailable = 0

        # Pace by the tokens just spent so we stay under the per-minute budget.
        # A throttled complaint spent ~nothing, so back off a fixed recovery
        # window instead to let the rolling minute clear.
        if i < len(records):
            if result.status == "unavailable":
                await asyncio.sleep(RECOVERY_SLEEP_S)
            else:
                tokens_used = sum(
                    d.prompt_tokens + d.completion_tokens for d in result.llm_calls
                ) + sum(
                    j.prompt_tokens + j.completion_tokens for j in result.judge_calls
                )
                await asyncio.sleep(min(tokens_used / tpm_budget * 60.0, MAX_PACE_S))

    return rows, all_outcomes


# --- aggregation + reporting -------------------------------------------------


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 1]); None on empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def summarize(
    rows: list[dict], all_outcomes: list[GuardrailOutcome], config: dict
) -> dict:
    """Roll per-complaint rows + every guardrail outcome into headline metrics."""
    n = len(rows)
    status_counts = Counter(r["status"] for r in rows)
    attempts_counts = Counter(r["attempts"] for r in rows)
    n_passed = status_counts.get("passed", 0)
    n_available = sum(1 for r in rows if r["status"] != "unavailable")

    layer_counts = Counter(v.layer for o in all_outcomes for v in o.violations)
    code_counts = Counter(v.code for o in all_outcomes for v in o.violations)

    tone_by_dim = {
        dim: [o.scores[dim] for o in all_outcomes if o.scores and dim in o.scores]
        for dim in TONE_DIMENSIONS
    }
    wall = [r["wall_ms"] for r in rows]
    resolution_costs = [r["draft_cost_usd"] + r["judge_cost_usd"] for r in rows]
    total_costs = [r["total_cost_usd"] for r in rows]

    return {
        "config": config,
        "n_complaints": n,
        "status_distribution": dict(status_counts),
        "guardrail_pass_rate_all": round(n_passed / n, 4) if n else None,
        "guardrail_pass_rate_available": (
            round(n_passed / n_available, 4) if n_available else None
        ),
        "attempts_distribution": {
            str(k): v for k, v in sorted(attempts_counts.items())
        },
        "mean_attempts": round(statistics.mean([r["attempts"] for r in rows]), 3)
        if rows
        else None,
        "violations_by_layer": dict(layer_counts),
        "violations_by_code": dict(code_counts),
        "tone_scores": {
            dim: {
                "n": len(vals),
                "mean": round(statistics.mean(vals), 2) if vals else None,
                "min": min(vals) if vals else None,
                "p50": _percentile(vals, 0.5),
            }
            for dim, vals in tone_by_dim.items()
        },
        "cost_usd": {
            "per_resolution_mean": (
                round(statistics.mean(resolution_costs), 6)
                if resolution_costs
                else None
            ),
            "per_resolution_total": round(sum(resolution_costs), 6),
            "per_complaint_mean": round(statistics.mean(total_costs), 6)
            if total_costs
            else None,
            "grand_total": round(sum(total_costs), 6),
        },
        "latency_ms": {
            "wall_p50": _percentile(wall, 0.5),
            "wall_p95": _percentile(wall, 0.95),
            "wall_mean": round(statistics.mean(wall), 1) if wall else None,
        },
        "draft_calls_mean": (
            round(statistics.mean([r["draft_calls"] for r in rows]), 2)
            if rows
            else None
        ),
        "judge_calls_mean": (
            round(statistics.mean([r["judge_calls"] for r in rows]), 2)
            if rows
            else None
        ),
    }


def _print_summary(summary: dict, out_path: Path) -> None:
    s = summary
    print("\n" + "=" * 64)
    print("  RESOLUTION-AGENT + GUARDRAIL BENCHMARK")
    print("=" * 64)
    print(f"  complaints:        {s['n_complaints']}  ({s['config']['model']})")
    print(
        f"  classification:    {'live Groq' if s['config']['classify_via_groq'] else 'gold labels'}"
    )
    print(f"  status:            {s['status_distribution']}")
    print(
        f"  guardrail pass:    {s['guardrail_pass_rate_all']} (all) / "
        f"{s['guardrail_pass_rate_available']} (available)"
    )
    print(
        f"  attempts:          {s['attempts_distribution']}  mean={s['mean_attempts']}"
    )
    print(f"  violations/layer:  {s['violations_by_layer'] or '{}'}")
    print(f"  violations/code:   {s['violations_by_code'] or '{}'}")
    print("  tone scores (1-10):")
    for dim, st in s["tone_scores"].items():
        print(f"      {dim:<16} mean={st['mean']} min={st['min']} (n={st['n']})")
    c = s["cost_usd"]
    print(
        f"  cost/resolution:   ${c['per_resolution_mean']} mean  (${c['per_resolution_total']} total)"
    )
    print(
        f"  cost/complaint:    ${c['per_complaint_mean']} mean  (${c['grand_total']} grand total)"
    )
    lat = s["latency_ms"]
    print(
        f"  latency wall:      p50={lat['wall_p50']}ms p95={lat['wall_p95']}ms mean={lat['wall_mean']}ms"
    )
    print(
        f"  calls/resolution:  draft={s['draft_calls_mean']} judge={s['judge_calls_mean']}"
    )
    print("=" * 64)
    print(f"  written to: {out_path}")
    print("=" * 64 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the resolution agent + guardrails over "
        "escalation-worthy CFPB complaints."
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Number of complaints."
    )
    parser.add_argument(
        "--classify-via-groq",
        action="store_true",
        help="Re-classify each complaint via Groq instead of feeding gold labels.",
    )
    parser.add_argument(
        "--require-company",
        action="store_true",
        help="Only pick complaints that name a company.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_PATH, help="Metrics JSON output path."
    )
    parser.add_argument(
        "--tpm-budget",
        type=float,
        default=DEFAULT_TPM_BUDGET,
        help="Tokens/min to pace under (Groq free tier hard limit is 12000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select complaints and print the plan; make no LLM calls.",
    )
    args = parser.parse_args()

    if not GOLD_PATH.exists():
        logger.error("Gold labels not found at %s", GOLD_PATH)
        return 2

    records = load_candidates(args.limit, require_company=args.require_company)
    logger.info(
        "Selected %d escalation-worthy complaints (extreme_negative or urgency>=4).",
        len(records),
    )

    if args.dry_run:
        for r in records[:5]:
            c = r.get("classification") or {}
            logger.info(
                "  e.g. product=%r company=%r sentiment=%s urgency=%s",
                r.get("product"),
                r.get("company"),
                c.get("sentiment"),
                c.get("urgency"),
            )
        logger.info("Dry run: no LLM calls made.")
        return 0

    if not _GROQ_KEY:
        logger.error(
            "No GROQ_API_KEY in env or %s — cannot run live.", REPO_ROOT / ".env"
        )
        return 2

    rows, all_outcomes = asyncio.run(
        run_benchmark(records, args.classify_via_groq, tpm_budget=args.tpm_budget)
    )
    config = {
        "limit": args.limit,
        "classify_via_groq": args.classify_via_groq,
        "model": GROQ_MODEL,
        "source": "gold_labels.jsonl",
        "shuffle_seed": SHUFFLE_SEED,
    }
    summary = summarize(rows, all_outcomes, config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    _print_summary(summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
