"""
Populate the Neo4j knowledge graph from curated JSON + mined Postgres data.

Two sources, as designed in Phase 3:
  - CURATED (this directory's JSON): Regulation and ResolutionPattern nodes —
    domain knowledge the complaint data doesn't contain.
  - MINED (Postgres complaints table): Product, Issue, and Company nodes plus
    the edges between them — aggregated straight from the 200K rows.

Build order matters (edges need both endpoints to exist first):
  1. constraints (uniqueness on node keys → MERGE is fast and safe)
  2. products + issues + HAS_ISSUE         (mined taxonomy)
  3. regulations + APPLIES_TO + GOVERNED_BY (curated, linked into taxonomy)
  4. resolution patterns + ADDRESSES        (curated, linked to issues)
  5. companies + HAS_COMPLAINTS_ABOUT + RESOLVED_WITH (mined, top-N by volume)

Idempotent: every write is MERGE/SET, so re-running updates in place. Use
--reset to wipe the graph first for a clean rebuild.

Usage (inside the api container, with the stack up):
    PYTHONPATH=/app python /scripts/seed_graph.py            # graph_seed is bind-mounted? see note
    PYTHONPATH=/app python graph_seed/seed_graph.py --limit 500
    PYTHONPATH=/app python graph_seed/seed_graph.py --reset

Note: graph_seed/ is not bind-mounted into the api container by default; run
this from the host venv (Neo4j on localhost:7688, Postgres on localhost:5433)
or add a bind mount. The --neo4j-uri / DATABASE_URL come from settings/env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path

# Bootstrap: make `app.*` importable whether run from repo root or the container.
_BACKEND = Path("/app")
if _BACKEND.exists() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
else:
    _local = Path(__file__).resolve().parent.parent / "backend"
    if _local.exists():
        sys.path.insert(0, str(_local))

from neo4j import AsyncGraphDatabase  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import AsyncSessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seed_graph")

_HERE = Path(__file__).resolve().parent

# --- Tunables (the risk model lives here on purpose — easy to recalibrate) ---

TOP_COMPANIES_DEFAULT = 500
NEO4J_BATCH = 1000

# Which company_response values count as a negative signal for risk. "Closed
# with explanation" is deliberately NOT here — it's the dominant outcome and
# means "no relief but answered", which would saturate the score if treated as
# adverse. We count only genuinely bad outcomes.
ADVERSE_RESPONSES = {"Untimely response", "Closed without relief"}

# risk_score = w_volume * (log-scaled volume) + w_adverse * (adverse rate).
# Both halves are in [0,1]; weights sum to 1. Tune freely — it's a prior, not
# a measured truth (Phase 6 can replace it with an outcome-derived score).
RISK_VOLUME_WEIGHT = 0.5
RISK_ADVERSE_WEIGHT = 0.5


# ---------------- Pure helpers (unit-tested) ----------------


def compute_risk_score(total: int, adverse: int, max_total: int) -> float:
    """Blend complaint volume (log-scaled) with the adverse-response rate.

    Log-scaling volume keeps a company with 50K complaints from being 1000x
    riskier than one with 50 — risk grows with magnitude, not raw count.
    """
    if total <= 0:
        return 0.0
    volume_factor = math.log10(total + 1) / math.log10(max_total + 1) if max_total > 1 else 0.0
    adverse_rate = adverse / total
    score = RISK_VOLUME_WEIGHT * volume_factor + RISK_ADVERSE_WEIGHT * adverse_rate
    return round(min(1.0, max(0.0, score)), 4)


def match_products_to_regulation(applies_to: list[str], product_names: list[str]) -> list[str]:
    """Fuzzy-link a regulation's product families to real mined product names.

    Substring match both directions so "Credit reporting" connects to the full
    CFPB string "Credit reporting, credit repair services, ...".
    """
    fams = [f.lower() for f in applies_to]
    out = []
    for prod in product_names:
        pl = prod.lower()
        if any(fam in pl or pl in fam for fam in fams):
            out.append(prod)
    return out


def match_issues_to_pattern(addresses: list[str], issue_names: list[str]) -> list[str]:
    """Keyword-link a resolution pattern to issues whose name contains a keyword."""
    if not addresses:
        return []
    kws = [k.lower() for k in addresses]
    return [iss for iss in issue_names if any(kw in iss.lower() for kw in kws)]


def build_response_to_pattern(patterns: list[dict]) -> dict[str, str]:
    """Map each raw company_response string to its ResolutionPattern id."""
    mapping: dict[str, str] = {}
    for p in patterns:
        for rv in p.get("response_values", []):
            mapping[rv] = p["id"]
    return mapping


def _load_json(name: str) -> list[dict]:
    with (_HERE / name).open() as f:
        return json.load(f)


# ---------------- Neo4j write helper ----------------


async def _unwind(session, query: str, rows: list[dict | str], batch: int = NEO4J_BATCH) -> int:
    """Run an UNWIND $rows query in chunks inside managed write transactions."""
    written = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]

        async def _tx(tx, _chunk=chunk):
            result = await tx.run(query, rows=_chunk)
            await result.consume()

        await session.execute_write(_tx)
        written += len(chunk)
    return written


# ---------------- Postgres mining ----------------


async def mine_postgres(top_n: int) -> dict:
    """Pull every aggregate the graph needs in one Postgres session."""
    async with AsyncSessionLocal() as s:
        products = [
            r["product"]
            for r in (
                await s.execute(
                    text("SELECT DISTINCT product FROM complaints WHERE product IS NOT NULL")
                )
            ).mappings()
        ]
        issues = [
            r["issue"]
            for r in (
                await s.execute(
                    text("SELECT DISTINCT issue FROM complaints WHERE issue IS NOT NULL")
                )
            ).mappings()
        ]
        product_issue = [
            dict(r)
            for r in (
                await s.execute(
                    text(
                        "SELECT product, issue, count(*) AS freq FROM complaints "
                        "WHERE product IS NOT NULL AND issue IS NOT NULL "
                        "GROUP BY product, issue"
                    )
                )
            ).mappings()
        ]
        companies = [
            dict(r)
            for r in (
                await s.execute(
                    text(
                        "SELECT company, count(*) AS total, "
                        "  sum(CASE WHEN company_response = ANY(:adverse) THEN 1 ELSE 0 END) "
                        "    AS adverse "
                        "FROM complaints WHERE company IS NOT NULL "
                        "GROUP BY company ORDER BY total DESC LIMIT :limit"
                    ),
                    {"adverse": list(ADVERSE_RESPONSES), "limit": top_n},
                )
            ).mappings()
        ]
        top_names = [c["company"] for c in companies]
        company_product = []
        company_response = []
        if top_names:
            company_product = [
                dict(r)
                for r in (
                    await s.execute(
                        text(
                            "SELECT company, product, count(*) AS cnt FROM complaints "
                            "WHERE company = ANY(:names) AND product IS NOT NULL "
                            "GROUP BY company, product"
                        ),
                        {"names": top_names},
                    )
                ).mappings()
            ]
            company_response = [
                dict(r)
                for r in (
                    await s.execute(
                        text(
                            "SELECT company, company_response, count(*) AS cnt FROM complaints "
                            "WHERE company = ANY(:names) AND company_response IS NOT NULL "
                            "GROUP BY company, company_response"
                        ),
                        {"names": top_names},
                    )
                ).mappings()
            ]

    return {
        "products": products,
        "issues": issues,
        "product_issue": product_issue,
        "companies": companies,
        "company_product": company_product,
        "company_response": company_response,
    }


# ---------------- Orchestration ----------------


async def seed(top_n: int, reset: bool) -> None:
    regulations = _load_json("regulations.json")
    patterns = _load_json("resolution_patterns.json")

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        await driver.verify_connectivity()
        log.info("connected to Neo4j at %s", settings.neo4j_uri)

        log.info("mining Postgres aggregates (top %d companies)...", top_n)
        data = await mine_postgres(top_n)
        log.info(
            "mined: %d products, %d issues, %d product-issue pairs, %d companies",
            len(data["products"]),
            len(data["issues"]),
            len(data["product_issue"]),
            len(data["companies"]),
        )

        async with driver.session(database=settings.neo4j_database) as session:
            if reset:
                log.warning("--reset: deleting all nodes and relationships")
                await session.run("MATCH (n) DETACH DELETE n")

            # 1. constraints
            for c in _CONSTRAINTS:
                await session.run(c)

            # 2. taxonomy (mined)
            await _unwind(
                session,
                "UNWIND $rows AS name MERGE (p:Product {name: name})",
                data["products"],
            )
            await _unwind(
                session,
                "UNWIND $rows AS name MERGE (i:Issue {name: name})",
                data["issues"],
            )
            await _unwind(
                session,
                "UNWIND $rows AS row "
                "MATCH (p:Product {name: row.product}) MATCH (i:Issue {name: row.issue}) "
                "MERGE (p)-[h:HAS_ISSUE]->(i) SET h.frequency = row.freq",
                data["product_issue"],
            )

            # 3. regulations (curated) + edges into taxonomy
            await _unwind(
                session,
                "UNWIND $rows AS row MERGE (r:Regulation {id: row.id}) "
                "SET r.title = row.title, r.cfr_reference = row.cfr_reference, "
                "    r.implementing_regulation = row.implementing_regulation, "
                "    r.summary = row.summary, r.key_provisions = row.key_provisions",
                regulations,
            )
            applies_to = [
                {"reg_id": reg["id"], "product": prod}
                for reg in regulations
                for prod in match_products_to_regulation(reg["applies_to"], data["products"])
            ]
            await _unwind(
                session,
                "UNWIND $rows AS row MATCH (r:Regulation {id: row.reg_id}) "
                "MATCH (p:Product {name: row.product}) MERGE (r)-[:APPLIES_TO]->(p)",
                applies_to,
            )
            # GOVERNED_BY derived transitively: reg APPLIES_TO product HAS_ISSUE issue
            await session.run(
                "MATCH (r:Regulation)-[:APPLIES_TO]->(:Product)-[:HAS_ISSUE]->(i:Issue) "
                "MERGE (i)-[:GOVERNED_BY]->(r)"
            )

            # 4. resolution patterns (curated) + ADDRESSES
            await _unwind(
                session,
                "UNWIND $rows AS row MERGE (rp:ResolutionPattern {id: row.id}) "
                "SET rp.pattern_type = row.pattern_type, rp.description = row.description, "
                "    rp.success_rate = row.success_rate",
                patterns,
            )
            addresses = [
                {"pattern_id": p["id"], "issue": iss}
                for p in patterns
                for iss in match_issues_to_pattern(p.get("addresses", []), data["issues"])
            ]
            await _unwind(
                session,
                "UNWIND $rows AS row MATCH (rp:ResolutionPattern {id: row.pattern_id}) "
                "MATCH (i:Issue {name: row.issue}) MERGE (rp)-[:ADDRESSES]->(i)",
                addresses,
            )

            # 5. companies (mined) + edges
            max_total = data["companies"][0]["total"] if data["companies"] else 0
            company_rows = [
                {
                    "name": c["company"],
                    "total": int(c["total"]),
                    "risk_score": compute_risk_score(
                        int(c["total"]), int(c["adverse"] or 0), int(max_total)
                    ),
                }
                for c in data["companies"]
            ]
            await _unwind(
                session,
                "UNWIND $rows AS row MERGE (c:Company {name: row.name}) "
                "SET c.total_complaints = row.total, c.risk_score = row.risk_score",
                company_rows,
            )
            await _unwind(
                session,
                "UNWIND $rows AS row MATCH (c:Company {name: row.company}) "
                "MATCH (p:Product {name: row.product}) "
                "MERGE (c)-[h:HAS_COMPLAINTS_ABOUT]->(p) SET h.count = row.cnt",
                [
                    {"company": r["company"], "product": r["product"], "cnt": int(r["cnt"])}
                    for r in data["company_product"]
                ],
            )
            resp_to_pattern = build_response_to_pattern(patterns)
            resolved_with = [
                {
                    "company": r["company"],
                    "pattern_id": resp_to_pattern[r["company_response"]],
                    "cnt": int(r["cnt"]),
                }
                for r in data["company_response"]
                if r["company_response"] in resp_to_pattern
            ]
            await _unwind(
                session,
                "UNWIND $rows AS row MATCH (c:Company {name: row.company}) "
                "MATCH (rp:ResolutionPattern {id: row.pattern_id}) "
                "MERGE (c)-[res:RESOLVED_WITH]->(rp) SET res.count = row.cnt",
                resolved_with,
            )

            counts = await _summarize(session)
        log.info("seed complete. graph now holds: %s", counts)
    finally:
        await driver.close()


async def _summarize(session) -> dict:
    """Count nodes per label + total relationships for the closing log line."""
    out = {}
    for label in ("Company", "Product", "Issue", "Regulation", "ResolutionPattern"):
        res = await session.run(f"MATCH (n:{label}) RETURN count(n) AS n")
        out[label] = (await res.single())["n"]
    res = await session.run("MATCH ()-[r]->() RETURN count(r) AS n")
    out["relationships"] = (await res.single())["n"]
    return out


_CONSTRAINTS = [
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT issue_name IF NOT EXISTS FOR (i:Issue) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT regulation_id IF NOT EXISTS FOR (r:Regulation) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (rp:ResolutionPattern) REQUIRE rp.id IS UNIQUE",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed the Neo4j knowledge graph.")
    p.add_argument(
        "--limit",
        type=int,
        default=TOP_COMPANIES_DEFAULT,
        help=f"Top-N companies by volume to load (default {TOP_COMPANIES_DEFAULT}).",
    )
    p.add_argument("--reset", action="store_true", help="Wipe the graph before seeding.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(seed(top_n=args.limit, reset=args.reset))


if __name__ == "__main__":
    main()
