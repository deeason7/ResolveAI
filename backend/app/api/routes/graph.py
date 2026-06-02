"""
Knowledge-graph endpoints (Neo4j-backed):

  GET /graph/company/{name}                — company risk profile
  GET /graph/product/{name}/regulations    — regulations governing a product
  GET /graph/explore                        — bounded neighborhood for viz

All three are authenticated. The company aggregates are derived from the same
consumer-PII dataset as /complaints, so even read access needs an account.
These are read-only traversals — the classification worker, not these routes,
is what mutates the graph.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import get_current_user, get_graph_store
from app.models.user import User
from app.schemas.graph import CompanyProfile, GraphNeighborhood, Regulation
from app.services.graph_store import GraphStore

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/company/{name}", response_model=CompanyProfile)
async def get_company_profile(
    name: str,
    store: GraphStore = Depends(get_graph_store),
    _: User = Depends(get_current_user),
) -> CompanyProfile:
    """Risk view for one company: totals, risk score, violations, product mix."""
    profile = await store.get_company_profile(name)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return profile


@router.get("/product/{name}/regulations", response_model=list[Regulation])
async def get_product_regulations(
    name: str,
    issue: str | None = Query(default=None, max_length=255),
    store: GraphStore = Depends(get_graph_store),
    _: User = Depends(get_current_user),
) -> list[Regulation]:
    """Regulations governing a product, optionally narrowed to a single issue.

    An empty list is a valid 200 — the product exists in the taxonomy but has no
    regulations mapped to it yet — so we don't 404 here the way the company
    lookup does.
    """
    return await store.get_regulations(name, issue)


@router.get("/explore", response_model=GraphNeighborhood)
async def explore_graph(
    node_id: str = Query(..., min_length=1, max_length=255),
    depth: int = Query(default=2, ge=1, le=3),
    store: GraphStore = Depends(get_graph_store),
    _: User = Depends(get_current_user),
) -> GraphNeighborhood:
    """Subgraph within `depth` hops of a node (matched by name or id).

    `depth` is capped at 3: each hop fans out across the dense company/product/
    issue web, so an unbounded traversal could pull most of the graph into one
    response.
    """
    neighborhood = await store.get_graph_neighborhood(node_id, depth)
    if not neighborhood["nodes"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    return GraphNeighborhood(**neighborhood)
