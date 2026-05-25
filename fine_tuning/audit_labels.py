"""
Quality-audit the teacher-labeled dataset before we spend GPU hours on it.

Industry-standard checklist for an LLM-labeled supervised dataset:

  1. Schema integrity      — every row has all required fields, vocab is closed
  2. Distribution analysis — class balance, urgency histogram, crosstabs
  3. Leakage checks        — narrative length per class (length-leak),
                             top-token frequency per class (lexical-leak),
                             sentiment↔urgency consistency
  4. Reasoning audit       — template/boilerplate detection on the reasoning
                             field (teacher may have collapsed into a formula)
  5. Stratified sample     — balanced human-review CSV so you can eyeball
                             every minority class, not just the majority

The script reads from `complaint_labels` JOIN `complaints` (DB is the
source of truth) and writes three artifacts under
`fine_tuning/data/audit/`:

  - distribution.json   — machine-readable summary, diffable across runs
  - audit_report.md     — human-readable digest with red flags up top
  - sample_for_review.csv — stratified review sheet (open in Numbers/Excel,
                             fill in agree_y_n and notes columns)

Usage (inside the api container):

    PYTHONPATH=/app python /fine_tuning/audit_labels.py
    PYTHONPATH=/app python /fine_tuning/audit_labels.py \
        --label-source "bedrock:us.meta.llama3-3-70b-instruct-v1:0" \
        --sample-size 120

Exit codes:
    0 — audit completed (artifacts written)
    2 — fatal misconfiguration (no labels for the given source)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Bootstrap import path (mirrors 01_prepare_labels.py so this script runs
# from inside the api container without any sys.path gymnastics).
_BACKEND_DIR = Path("/app")
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
else:
    _local = Path(__file__).resolve().parent.parent / "backend"
    if _local.exists():
        sys.path.insert(0, str(_local))

from sqlalchemy import text  # noqa: E402

from app.database import AsyncSessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit_labels")
for _name in ("sqlalchemy.engine.Engine", "sqlalchemy.engine", "sqlalchemy"):
    logging.getLogger(_name).setLevel(logging.WARNING)

DEFAULT_SOURCE = "bedrock:us.meta.llama3-3-70b-instruct-v1:0"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "audit"
NARRATIVE_TRUNCATE_CHARS = 800
TOP_K_TOKENS_PER_CLASS = 20
TOP_K_REASONING_OPENINGS = 10
REASONING_OPENING_NGRAM = 4  # capture "the consumer is disputing" → length 4
SAMPLE_RANDOM_SEED = 17  # frozen so reviews are reproducible

# Closed vocabularies — duplicated from app.schemas.classification rather
# than imported to keep this script standalone and importable even if the
# schema module evolves. If you add a new sentiment/intent, update both.
SENTIMENTS = ("neutral", "negative", "extreme_negative")
INTENTS = (
    "information_request",
    "dispute_resolution",
    "account_action",
    "fraud_report",
    "regulatory_complaint",
)
ENTITY_TYPES = (
    "company",
    "product",
    "issue",
    "regulation",
    "amount",
    "person",
    "account_type",
    "other",
)

# Compact stopword list — enough to surface domain signal without pulling
# in nltk. Includes the obvious English function words plus CFPB-specific
# noise tokens (XXXX is the redaction marker, XX dates are common).
STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been
    before being but by can could did do does doing don for from had has
    have having he her here hers him his how i if in into is it its just
    me my no nor not now of off on once one only or other our ours out
    over own same she so some such than that the their them then there
    these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your
    yours xxxx xx
    """.split()
)

TOKEN_RE = re.compile(r"[a-z][a-z'\-]{2,}")  # words of 3+ chars, lowercase


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


async def load_labels(label_source: str) -> list[dict]:
    """Pull every label for `label_source` joined to its complaint.

    Uses raw SQL — SQLModel/ORM relationships are overkill for a one-shot
    analytic query and the explicit column list makes the schema contract
    obvious to a reader.
    """
    query = text(
        """
        SELECT
            l.complaint_id::text AS complaint_id,
            l.sentiment,
            l.intent,
            l.urgency,
            l.key_entities,
            l.reasoning,
            l.input_tokens,
            l.output_tokens,
            l.latency_ms,
            l.labeled_at,
            c.cfpb_complaint_id,
            c.narrative,
            c.product,
            c.issue,
            c.company
        FROM complaint_labels l
        JOIN complaints c ON c.id = l.complaint_id
        WHERE l.label_source = :source
        ORDER BY l.labeled_at
        """
    )
    async with AsyncSessionLocal() as session:
        result = await session.execute(query, {"source": label_source})
        rows = [dict(r) for r in result.mappings().all()]
    return rows


# --------------------------------------------------------------------------
# Distribution analysis
# --------------------------------------------------------------------------


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def _counter_to_dist(counter: Counter, total: int, key_order=None) -> dict:
    """Convert a Counter to {key: {count, pct}} with stable key ordering."""
    keys = key_order if key_order is not None else sorted(counter.keys())
    out = {}
    for k in keys:
        out[k] = {"count": counter[k], "pct": _pct(counter[k], total)}
    # Include any keys present in the data but not in the expected vocab —
    # these are violations of the closed vocabulary and worth surfacing.
    for k in counter:
        if k not in out:
            out[k] = {"count": counter[k], "pct": _pct(counter[k], total), "_unexpected": True}
    return out


def compute_distributions(rows: list[dict]) -> dict:
    """Class balance, urgency histogram, entity stats, crosstabs."""
    n = len(rows)
    sentiment_c = Counter(r["sentiment"] for r in rows)
    intent_c = Counter(r["intent"] for r in rows)
    urgency_c = Counter(r["urgency"] for r in rows)

    # Sentiment × Intent crosstab — sparsity here ("0 neutrals filed a
    # fraud_report") is informative but not necessarily wrong.
    sent_intent: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        sent_intent[r["sentiment"]][r["intent"]] += 1
    sent_intent_xt = {s: {i: sent_intent[s].get(i, 0) for i in INTENTS} for s in SENTIMENTS}

    # Sentiment × Urgency crosstab — this is the consistency check. A
    # neutral with urgency 5 or an extreme_negative with urgency 1 is a
    # rubric inconsistency we want to count.
    sent_urgency: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        sent_urgency[r["sentiment"]][r["urgency"]] += 1
    sent_urgency_xt = {
        s: {str(u): sent_urgency[s].get(u, 0) for u in range(1, 6)} for s in SENTIMENTS
    }

    # Entity stats
    ent_type_c: Counter = Counter()
    entities_per_label: list[int] = []
    for r in rows:
        ents = r.get("key_entities") or []
        entities_per_label.append(len(ents))
        for e in ents:
            t = e.get("type") if isinstance(e, dict) else None
            if t:
                ent_type_c[t] += 1

    # Token/cost stats — useful for retrospective budgeting and for
    # spotting outliers (a label that burned 4000 tokens is suspicious).
    in_tokens = [r["input_tokens"] for r in rows if r["input_tokens"]]
    out_tokens = [r["output_tokens"] for r in rows if r["output_tokens"]]
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"]]

    return {
        "total_labels": n,
        "sentiment": _counter_to_dist(sentiment_c, n, SENTIMENTS),
        "intent": _counter_to_dist(intent_c, n, INTENTS),
        "urgency": _counter_to_dist(urgency_c, n, key_order=[1, 2, 3, 4, 5]),
        "sentiment_x_intent": sent_intent_xt,
        "sentiment_x_urgency": sent_urgency_xt,
        "entity_type": _counter_to_dist(ent_type_c, sum(ent_type_c.values()), ENTITY_TYPES),
        "entities_per_label": _five_number_summary(entities_per_label),
        "operational": {
            "input_tokens": _five_number_summary(in_tokens),
            "output_tokens": _five_number_summary(out_tokens),
            "latency_ms": _five_number_summary(latencies),
            "total_input_tokens": sum(in_tokens),
            "total_output_tokens": sum(out_tokens),
        },
    }


def _five_number_summary(xs: list[float]) -> dict:
    """min / p25 / median / p75 / p95 / max / mean — same shape as pandas describe()."""
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return s[idx]

    return {
        "n": n,
        "min": s[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": s[-1],
        "mean": round(statistics.fmean(s), 2),
    }


# --------------------------------------------------------------------------
# Leakage checks
# --------------------------------------------------------------------------


def narrative_length_per_class(rows: list[dict]) -> dict:
    """Char-length distribution per sentiment + intent.

    Length-leak risk: if extreme_negative narratives are 2-3x longer than
    neutral ones, the model can hit high accuracy just by counting
    characters — which won't transfer to inference on shorter snippets.
    """
    by_sent: dict[str, list[int]] = defaultdict(list)
    by_intent: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        nl = len(r["narrative"] or "")
        by_sent[r["sentiment"]].append(nl)
        by_intent[r["intent"]].append(nl)
    return {
        "by_sentiment": {s: _five_number_summary(by_sent[s]) for s in SENTIMENTS},
        "by_intent": {i: _five_number_summary(by_intent[i]) for i in INTENTS},
    }


def top_tokens_per_class(rows: list[dict], top_k: int) -> dict:
    """Per-class top-K tokens, normalized by class size.

    A naive top-K-frequent-tokens-per-class report; not as principled as
    log-odds with informative Dirichlet prior, but cheap and good enough
    to spot the obvious "fraud lights up fraud_report" type leaks.
    """
    by_sent_tokens: dict[str, Counter] = defaultdict(Counter)
    by_intent_tokens: dict[str, Counter] = defaultdict(Counter)
    sent_class_size: Counter = Counter()
    intent_class_size: Counter = Counter()

    for r in rows:
        toks = [t for t in TOKEN_RE.findall((r["narrative"] or "").lower()) if t not in STOPWORDS]
        by_sent_tokens[r["sentiment"]].update(toks)
        by_intent_tokens[r["intent"]].update(toks)
        sent_class_size[r["sentiment"]] += 1
        intent_class_size[r["intent"]] += 1

    def top(counter: Counter, class_size: int) -> list[dict]:
        # Frequency per 1K complaints — comparable across classes of
        # different sizes (this is what you actually want vs raw counts).
        return [
            {"token": tok, "freq_per_1k": round(1000 * c / class_size, 2), "raw": c}
            for tok, c in counter.most_common(top_k)
        ]

    return {
        "by_sentiment": {s: top(by_sent_tokens[s], sent_class_size[s] or 1) for s in SENTIMENTS},
        "by_intent": {i: top(by_intent_tokens[i], intent_class_size[i] or 1) for i in INTENTS},
    }


def sentiment_urgency_consistency(rows: list[dict]) -> dict:
    """Count rubric-inconsistent rows.

    The labeling rubric implies a monotonic relationship: extreme_negative
    should rarely be urgency 1, neutral should rarely be urgency 5.
    Quantifying these violations tells us how reliable the teacher was
    at honoring its own rubric.
    """
    flags = {
        "neutral_high_urgency": 0,  # neutral + urgency >= 4
        "extreme_low_urgency": 0,  # extreme_negative + urgency <= 2
        "negative_floor_urgency": 0,  # negative + urgency == 1
    }
    for r in rows:
        s, u = r["sentiment"], r["urgency"]
        if s == "neutral" and u >= 4:
            flags["neutral_high_urgency"] += 1
        if s == "extreme_negative" and u <= 2:
            flags["extreme_low_urgency"] += 1
        if s == "negative" and u == 1:
            flags["negative_floor_urgency"] += 1
    n = len(rows)
    return {k: {"count": v, "pct": _pct(v, n)} for k, v in flags.items()}


def reasoning_audit(rows: list[dict]) -> dict:
    """Length stats + top opening n-grams of the reasoning field.

    Template detection: if 30% of reasonings start with "the consumer is",
    the teacher has collapsed into a formula and the model will too.
    """
    lengths = [len(r["reasoning"] or "") for r in rows]
    word_counts = [len((r["reasoning"] or "").split()) for r in rows]
    opening_c: Counter = Counter()
    n = REASONING_OPENING_NGRAM
    for r in rows:
        words = (r["reasoning"] or "").lower().split()
        if len(words) >= n:
            opening_c[" ".join(words[:n])] += 1
    top = [
        {"opening": op, "count": c, "pct": _pct(c, len(rows))}
        for op, c in opening_c.most_common(TOP_K_REASONING_OPENINGS)
    ]
    return {
        "char_length": _five_number_summary(lengths),
        "word_count": _five_number_summary(word_counts),
        "top_openings": top,
    }


# --------------------------------------------------------------------------
# Stratified sample for human review
# --------------------------------------------------------------------------


def stratified_sample(rows: list[dict], n_total: int) -> list[dict]:
    """Equal-sized strata per sentiment.

    If a class has fewer rows than its share calls for, take all of them
    and let the larger classes pick up the slack — we'd rather slightly
    over-represent the majority than under-represent the minority.
    """
    rng = random.Random(SAMPLE_RANDOM_SEED)
    by_sent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sent[r["sentiment"]].append(r)

    per_class = n_total // len(SENTIMENTS)
    picked: list[dict] = []
    leftover_budget = n_total - per_class * len(SENTIMENTS)

    overflow_pool: list[dict] = []
    for s in SENTIMENTS:
        pool = by_sent.get(s, [])
        if len(pool) <= per_class:
            picked.extend(pool)
            # The class is smaller than its quota — refund the slack to
            # the overflow pool so we still hit n_total.
            leftover_budget += per_class - len(pool)
        else:
            chosen = rng.sample(pool, per_class)
            picked.extend(chosen)
            chosen_ids = {c["complaint_id"] for c in chosen}
            overflow_pool.extend(p for p in pool if p["complaint_id"] not in chosen_ids)

    if leftover_budget > 0 and overflow_pool:
        picked.extend(rng.sample(overflow_pool, min(leftover_budget, len(overflow_pool))))

    rng.shuffle(picked)
    return picked


def write_sample_csv(sample: list[dict], path: Path) -> None:
    """Human review sheet — keep columns minimal so it's easy to scan."""
    cols = [
        "complaint_id",
        "cfpb_complaint_id",
        "product",
        "issue",
        "narrative_excerpt",
        "sentiment",
        "intent",
        "urgency",
        "reasoning",
        "agree_y_n",  # for reviewer to fill in
        "notes",  # for reviewer to fill in
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(cols)
        for r in sample:
            narrative = (r["narrative"] or "")[:NARRATIVE_TRUNCATE_CHARS]
            if r["narrative"] and len(r["narrative"]) > NARRATIVE_TRUNCATE_CHARS:
                narrative += " […]"
            writer.writerow(
                [
                    r["complaint_id"],
                    r["cfpb_complaint_id"] or "",
                    r["product"] or "",
                    r["issue"] or "",
                    narrative,
                    r["sentiment"],
                    r["intent"],
                    r["urgency"],
                    r["reasoning"] or "",
                    "",
                    "",
                ]
            )


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def render_report(
    label_source: str,
    distributions: dict,
    length_leak: dict,
    token_leak: dict,
    consistency: dict,
    reasoning: dict,
    sample_size: int,
    artifacts: dict[str, Path],
) -> str:
    n = distributions["total_labels"]
    s_dist = distributions["sentiment"]
    minority_share = min(s_dist[s]["pct"] for s in SENTIMENTS)
    minority_class = min(SENTIMENTS, key=lambda s: s_dist[s]["pct"])

    # Red flags surfaced up top — anything <5% class share or >10%
    # consistency violation gets a callout. Numbers tuned conservatively
    # for a 3-class problem with 10K labels.
    red_flags: list[str] = []
    if minority_share < 5.0:
        red_flags.append(
            f"⚠️  Minority class **{minority_class}** is only "
            f"{minority_share}% ({s_dist[minority_class]['count']} examples). "
            "Class-weighted loss strongly recommended; expect noisy per-class metrics."
        )
    if consistency["neutral_high_urgency"]["pct"] > 5:
        red_flags.append(
            f"⚠️  {consistency['neutral_high_urgency']['count']} "
            f"({consistency['neutral_high_urgency']['pct']}%) "
            "neutral-sentiment rows have urgency ≥ 4 — rubric inconsistency."
        )
    if consistency["extreme_low_urgency"]["pct"] > 5:
        red_flags.append(
            f"⚠️  {consistency['extreme_low_urgency']['count']} "
            f"({consistency['extreme_low_urgency']['pct']}%) "
            "extreme_negative rows have urgency ≤ 2 — rubric inconsistency."
        )
    top_opening = reasoning["top_openings"][0] if reasoning["top_openings"] else None
    if top_opening and top_opening["pct"] > 20:
        red_flags.append(
            f'⚠️  Reasoning template alert: "{top_opening["opening"]}…" '
            f"opens {top_opening['pct']}% of reasonings — teacher may be formulaic."
        )

    by_sent_len = length_leak["by_sentiment"]
    if all(by_sent_len[s].get("n") for s in SENTIMENTS):
        med_neg = by_sent_len["extreme_negative"]["median"]
        med_neu = by_sent_len["neutral"]["median"]
        if med_neu and med_neg / med_neu > 2.0:
            red_flags.append(
                f"⚠️  Length-leak suspect: extreme_negative narratives have "
                f"median length {med_neg} vs neutral {med_neu} "
                f"({round(med_neg / med_neu, 1)}× longer). Risk of model "
                "learning length as a shortcut."
            )

    lines: list[str] = []
    lines.append(f"# Label Audit Report — {label_source}")
    lines.append("")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_  ")
    lines.append(f"_Total labels: **{n:,}**_  ")
    lines.append("")

    lines.append("## Red Flags")
    if red_flags:
        lines.extend(f"- {rf}" for rf in red_flags)
    else:
        lines.append("- _None detected at the configured thresholds._")
    lines.append("")

    lines.append("## Class Balance")
    lines.append("")
    lines.append("### Sentiment")
    lines.append("| class | count | % |")
    lines.append("|---|---:|---:|")
    for s in SENTIMENTS:
        d = s_dist[s]
        lines.append(f"| {s} | {d['count']:,} | {d['pct']}% |")
    lines.append("")
    lines.append("### Intent")
    lines.append("| class | count | % |")
    lines.append("|---|---:|---:|")
    for i in INTENTS:
        d = distributions["intent"][i]
        lines.append(f"| {i} | {d['count']:,} | {d['pct']}% |")
    lines.append("")
    lines.append("### Urgency (1=low, 5=acute)")
    lines.append("| urgency | count | % |")
    lines.append("|---:|---:|---:|")
    for u in [1, 2, 3, 4, 5]:
        d = distributions["urgency"][u]
        lines.append(f"| {u} | {d['count']:,} | {d['pct']}% |")
    lines.append("")

    lines.append("## Sentiment × Urgency Consistency")
    lines.append("")
    lines.append("| sentiment \\ urgency | 1 | 2 | 3 | 4 | 5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in SENTIMENTS:
        row = distributions["sentiment_x_urgency"][s]
        lines.append(
            f"| {s} | {row['1']:,} | {row['2']:,} | {row['3']:,} | {row['4']:,} | {row['5']:,} |"
        )
    lines.append("")
    lines.append("### Inconsistency counts")
    for k, v in consistency.items():
        lines.append(f"- `{k}` — {v['count']:,} ({v['pct']}%)")
    lines.append("")

    lines.append("## Length-Leak Check (narrative chars per sentiment)")
    lines.append("")
    lines.append("| sentiment | n | median | p95 | mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in SENTIMENTS:
        d = by_sent_len[s]
        if d.get("n"):
            lines.append(f"| {s} | {d['n']:,} | {d['median']:,} | {d['p95']:,} | {d['mean']:,} |")
    lines.append("")

    lines.append(f"## Reasoning Audit (top-{TOP_K_REASONING_OPENINGS} openings)")
    lines.append("")
    lines.append(
        f"_n-gram length = {REASONING_OPENING_NGRAM} words. "
        "High concentration at the top = formulaic teacher._"
    )
    lines.append("")
    lines.append("| opening | count | % |")
    lines.append("|---|---:|---:|")
    for op in reasoning["top_openings"]:
        lines.append(f"| `{op['opening']}…` | {op['count']:,} | {op['pct']}% |")
    lines.append("")
    rl = reasoning["word_count"]
    lines.append(
        f"Reasoning word count: median {rl['median']}, p95 {rl['p95']}, "
        f"max {rl['max']} (n={rl['n']:,})."
    )
    lines.append("")

    lines.append("## Lexical Leak (top tokens per sentiment, per-1K-complaints)")
    lines.append("")
    lines.append(
        "_If the same token dominates a class with much higher freq than "
        "in other classes, the model may shortcut to it. Mostly diagnostic._"
    )
    for s in SENTIMENTS:
        lines.append("")
        lines.append(f"**{s}** — top {TOP_K_TOKENS_PER_CLASS}:")
        lines.append("")
        tops = token_leak["by_sentiment"][s]
        lines.append("| token | freq / 1K | raw |")
        lines.append("|---|---:|---:|")
        for t in tops:
            lines.append(f"| {t['token']} | {t['freq_per_1k']} | {t['raw']:,} |")
    lines.append("")

    lines.append("## Token Cost Summary")
    op = distributions["operational"]
    lines.append(f"- Total input tokens: **{op['total_input_tokens']:,}**  ")
    lines.append(f"- Total output tokens: **{op['total_output_tokens']:,}**  ")
    if op["input_tokens"].get("n"):
        lines.append(
            f"- Input tokens per call: median {op['input_tokens']['median']:,}, "
            f"p95 {op['input_tokens']['p95']:,}"
        )
    if op["output_tokens"].get("n"):
        lines.append(
            f"- Output tokens per call: median {op['output_tokens']['median']:,}, "
            f"p95 {op['output_tokens']['p95']:,}"
        )
    if op["latency_ms"].get("n"):
        lines.append(
            f"- Latency ms per call: median {op['latency_ms']['median']:,}, "
            f"p95 {op['latency_ms']['p95']:,}"
        )
    lines.append("")

    lines.append("## Artifacts")
    for k, p in artifacts.items():
        lines.append(f"- `{k}` → `{p}`")
    lines.append("")
    lines.append(
        f"_Human review: open `sample_for_review.csv` ({sample_size} rows, "
        "stratified by sentiment) and fill in the `agree_y_n` + `notes` "
        "columns. Disagreement rate > 10% = re-prompt the teacher or "
        "expand the rubric._"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading labels for source=%s", args.label_source)
    rows = await load_labels(args.label_source)
    if not rows:
        log.critical("no labels found for source=%s", args.label_source)
        return 2
    log.info("loaded %d labels", len(rows))

    log.info("computing distributions")
    distributions = compute_distributions(rows)

    log.info("running leakage checks")
    length_leak = narrative_length_per_class(rows)
    token_leak = top_tokens_per_class(rows, TOP_K_TOKENS_PER_CLASS)
    consistency = sentiment_urgency_consistency(rows)
    reasoning = reasoning_audit(rows)

    log.info("building stratified sample (n=%d)", args.sample_size)
    sample = stratified_sample(rows, args.sample_size)

    dist_path = out_dir / "distribution.json"
    report_path = out_dir / "audit_report.md"
    sample_path = out_dir / "sample_for_review.csv"

    log.info("writing distribution.json")
    payload = {
        "label_source": args.label_source,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "distributions": distributions,
        "narrative_length_per_class": length_leak,
        "sentiment_urgency_consistency": consistency,
        "reasoning_audit": reasoning,
        "top_tokens_per_class": token_leak,
    }
    dist_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    log.info("writing sample_for_review.csv")
    write_sample_csv(sample, sample_path)

    log.info("writing audit_report.md")
    report_md = render_report(
        args.label_source,
        distributions,
        length_leak,
        token_leak,
        consistency,
        reasoning,
        len(sample),
        {
            "distribution.json": dist_path,
            "audit_report.md": report_path,
            "sample_for_review.csv": sample_path,
        },
    )
    report_path.write_text(report_md, encoding="utf-8")

    log.info("DONE — see %s", report_path)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit teacher labels for QA.")
    p.add_argument(
        "--label-source",
        default=DEFAULT_SOURCE,
        help=f"Label source key to audit (default: {DEFAULT_SOURCE}).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Total rows in the human-review CSV (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where to write artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
