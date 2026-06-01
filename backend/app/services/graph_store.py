"""
Neo4j wrapper for the knowledge graph (companies, products, issues,
regulations, resolution patterns).

Why this shape:
  - Mirrors `vector_store.py`: a thin class around the external client plus an
    lru_cache singleton getter. The driver is the expensive, pooled resource —
    we want exactly one per process, created lazily on first use.
  - We use the ASYNC driver (`AsyncGraphDatabase`). Graph calls are network
    I/O, which is what asyncio is for, and our API routes are already async —
    so they `await` these methods directly with no thread-pool detour. Contrast
    the embedder / SLM, which are CPU-bound and therefore get trampolined off
    the loop with `asyncio.to_thread`; you can't get concurrency out of pure
    Python CPU work by awaiting it, but you *can* for a socket read.
  - Every query is parameterized (`$name`, never f-strings). Cypher injection is
    the graph-DB cousin of SQL injection — string-building a query with a
    company name lets a crafted name rewrite the query.
  - Reads use READ routing, writes use WRITE routing. On single-node community
    Neo4j this is cosmetic, but it documents intent and lets a future causal
    cluster route reads to replicas for free.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from uuid import UUID

from neo4j import AsyncDriver, AsyncGraphDatabase, RoutingControl

from app.config import settings
from app.schemas.graph import (
    CompanyProfile,
    ProductBreakdown,
    Regulation,
    ResolutionPattern,
)

log = logging.getLogger(__name__)

DEFAULT_DATABASE = "neo4j"  # community edition serves a single db named "neo4j"


class GraphStore:
    def __init__(self, driver: AsyncDriver, database: str = DEFAULT_DATABASE):
        self._driver = driver
        self._database = database

    async def verify(self) -> None:
        """Raise if the server is unreachable. Use in health checks / the seeder."""
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close the connection pool. Called on app shutdown."""
        await self._driver.close()

    # --- Reads ---

    async def get_company_profile(self, company_name: str) -> CompanyProfile | None:
        """Risk view for one company: counts, risk score, violations, product mix.

        Returns None if the company isn't in the graph.
        """
        rows = await self._run(
            """
            MATCH (c:Company {name: $company_name})
            OPTIONAL MATCH (c)-[:VIOLATED]->(reg:Regulation)
            WITH c, collect(DISTINCT reg.title) AS violations
            OPTIONAL MATCH (c)-[h:HAS_COMPLAINTS_ABOUT]->(p:Product)
            WITH c, violations, p, h ORDER BY h.count DESC
            RETURN c.name AS name,
                   c.total_complaints AS total_complaints,
                   c.risk_score AS risk_score,
                   violations AS violations,
                   collect(
                     CASE WHEN p IS NULL THEN null
                     ELSE {product: p.name, count: h.count} END
                   ) AS product_breakdown
            """,
            company_name=company_name,
        )
        if not rows:
            return None
        row = rows[0]
        return CompanyProfile(
            name=row["name"],
            total_complaints=row.get("total_complaints") or 0,
            risk_score=row.get("risk_score"),
            violations=row.get("violations") or [],
            product_breakdown=[
                ProductBreakdown(product=pb["product"], count=pb["count"])
                for pb in (row.get("product_breakdown") or [])
            ],
        )

    async def get_regulations(self, product: str, issue: str | None = None) -> list[Regulation]:
        """Regulations governing a product (optionally narrowed to one issue).

        With an issue we walk Product→Issue→Regulation; without one we fall back
        to the direct Regulation→Product APPLIES_TO edge so the caller still gets
        the product-level regulatory context.

        Issue nodes are shared across products (the CFPB taxonomy reuses issue
        names), so the transitive GOVERNED_BY edge alone would leak in
        regulations that govern the issue under a *different* product. The
        `WHERE (r)-[:APPLIES_TO]->(p)` guard re-imposes the queried product's
        context so only regulations applicable to THIS product come back.
        """
        if issue:
            query = """
                MATCH (p:Product {name: $product})-[:HAS_ISSUE]->
                      (i:Issue {name: $issue})-[:GOVERNED_BY]->(r:Regulation)
                WHERE (r)-[:APPLIES_TO]->(p)
                RETURN DISTINCT r{.id, .title, .cfr_reference, .summary,
                                  .key_provisions} AS regulation
                ORDER BY regulation.id
            """
            params = {"product": product, "issue": issue}
        else:
            query = """
                MATCH (r:Regulation)-[:APPLIES_TO]->(p:Product {name: $product})
                RETURN DISTINCT r{.id, .title, .cfr_reference, .summary,
                                  .key_provisions} AS regulation
                ORDER BY regulation.id
            """
            params = {"product": product}

        rows = await self._run(query, **params)
        return [Regulation(**row["regulation"]) for row in rows]

    async def get_resolution_patterns(
        self, issue: str, company: str | None = None
    ) -> list[ResolutionPattern]:
        """Resolution patterns that address an issue, ranked by success rate.

        If `company` is given, annotate each pattern with how often that company
        has used it (RESOLVED_WITH count).
        """
        rows = await self._run(
            """
            MATCH (i:Issue {name: $issue})<-[:ADDRESSES]-(rp:ResolutionPattern)
            OPTIONAL MATCH (c:Company {name: $company})-[res:RESOLVED_WITH]->(rp)
            RETURN rp.pattern_type AS pattern_type,
                   rp.description AS description,
                   rp.success_rate AS success_rate,
                   res.count AS company_usage_count
            ORDER BY coalesce(rp.success_rate, -1) DESC
            """,
            issue=issue,
            company=company,
        )
        return [
            ResolutionPattern(
                pattern_type=row["pattern_type"],
                description=row["description"],
                success_rate=row.get("success_rate"),
                company_usage_count=row.get("company_usage_count"),
            )
            for row in rows
        ]

    async def get_graph_neighborhood(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        """Subgraph within `depth` hops of a node, shaped for a viz library.

        `node_id` matches either a name (Company/Product/Issue) or an id
        (Regulation/ResolutionPattern). Returns {"nodes": [...], "edges": [...]}.
        Uses APOC (enabled via NEO4J_PLUGINS) for the bounded traversal.
        """
        rows = await self._run(
            """
            MATCH (n) WHERE n.name = $node_id OR n.id = $node_id
            CALL apoc.path.subgraphAll(n, {maxLevel: $depth})
            YIELD nodes, relationships
            RETURN
              [x IN nodes |
                {id: coalesce(x.id, x.name), labels: labels(x), props: properties(x)}
              ] AS nodes,
              [r IN relationships |
                {source: coalesce(startNode(r).id, startNode(r).name),
                 target: coalesce(endNode(r).id, endNode(r).name),
                 type: type(r), props: properties(r)}
              ] AS edges
            """,
            node_id=node_id,
            depth=depth,
        )
        if not rows:
            return {"nodes": [], "edges": []}
        return {"nodes": rows[0]["nodes"], "edges": rows[0]["edges"]}

    # --- Writes ---

    async def upsert_complaint_entities(
        self,
        complaint_id: str | UUID,
        company: str | None,
        product: str | None,
        issue: str | None,
        regulations: list[str] | None = None,
    ) -> None:
        """Fold one freshly-classified complaint into the graph's aggregates.

        Ensures the Company/Product/Issue nodes exist and bumps the
        HAS_COMPLAINTS_ABOUT / HAS_ISSUE edge counters; links VIOLATED edges for
        any regulation ids the classifier surfaced.

        NOTE: this increments counters, so it is NOT idempotent per complaint —
        reprocessing the same complaint (possible under Redis at-least-once
        delivery) double-counts. Acceptable for now; a Complaint-level dedup or
        a per-complaint edge is the Phase 6 fix. Tracked in the phase notes.
        """
        if company and product:
            await self._run(
                """
                MERGE (c:Company {name: $company})
                  ON CREATE SET c.total_complaints = 0
                MERGE (p:Product {name: $product})
                MERGE (c)-[h:HAS_COMPLAINTS_ABOUT]->(p)
                  ON CREATE SET h.count = 0
                SET h.count = h.count + 1
                """,
                write=True,
                company=company,
                product=product,
            )
        if product and issue:
            await self._run(
                """
                MERGE (p:Product {name: $product})
                MERGE (i:Issue {name: $issue})
                MERGE (p)-[r:HAS_ISSUE]->(i)
                  ON CREATE SET r.frequency = 0
                SET r.frequency = r.frequency + 1
                """,
                write=True,
                product=product,
                issue=issue,
            )
        for reg_id in regulations or []:
            await self._run(
                """
                MATCH (r:Regulation {id: $reg_id})
                MERGE (c:Company {name: $company})
                MERGE (c)-[v:VIOLATED]->(r)
                  ON CREATE SET v.count = 0
                SET v.count = v.count + 1, v.last_date = date()
                """,
                write=True,
                reg_id=reg_id,
                company=company or "Unknown",
            )

    # --- Internals ---

    async def _run(self, query: str, *, write: bool = False, **params: Any) -> list[dict]:
        """Execute a Cypher query and return records as plain dicts.

        Centralizing execution here gives the public methods a clean
        list[dict] contract and a single seam to mock in tests.
        """
        records, _summary, _keys = await self._driver.execute_query(
            query,
            database_=self._database,
            routing_=RoutingControl.WRITE if write else RoutingControl.READ,
            **params,
        )
        return [r.data() for r in records]


@lru_cache(maxsize=1)
def get_default_graph_store() -> GraphStore:
    """Process-wide singleton wired to the configured Neo4j instance."""
    log.info("connecting to Neo4j at %s", settings.neo4j_uri)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    return GraphStore(driver)


def reset_default_graph_store() -> None:
    """Drop the singleton. Test hook only."""
    get_default_graph_store.cache_clear()
