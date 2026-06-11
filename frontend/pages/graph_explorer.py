"""
Knowledge Graph Explorer — walk the Neo4j graph visually.

The backend's /graph/explore returns a bounded subgraph (APOC traversal,
depth-capped) as plain node/edge dicts; this page renders it with pyvis —
a vis.js HTML canvas embedded via st.components. Clicks stay in the
browser with this library, so expanding happens through the inspect panel:
pick a node, hit "Explore from here", and the graph re-centers on it.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

import api_client
import tour
from api_client import ApiError

DEFAULT_NODE = "EQUIFAX, INC."
PENDING_KEY = "graph_center_pending"
INPUT_KEY = "graph_node_input"

NODE_COLORS = {
    "Company": "#1565c0",
    "Product": "#2e7d32",
    "Issue": "#f9a825",
    "Regulation": "#c62828",
    "ResolutionPattern": "#6a1b9a",
}
LEGEND = "🔵 Company · 🟢 Product · 🟡 Issue · 🔴 Regulation · 🟣 Resolution pattern"
MAX_LABEL_CHARS = 28
# Above this, vis.js physics drags the browser — the dense issue web makes
# depth-2 company neighborhoods hit hundreds of nodes (EQUIFAX@2 ≈ 670).
PHYSICS_MAX_NODES = 300


def _display_name(node: dict) -> str:
    props = node.get("props", {})
    return str(props.get("name") or props.get("title") or node["id"])


def _hover_text(node: dict) -> str:
    lines = [", ".join(node.get("labels", []))]
    lines += [f"{k}: {v}" for k, v in node.get("props", {}).items()]
    return "\n".join(lines)


def _render_network(nodes: list[dict], edges: list[dict]) -> None:
    # in_line embeds vis.js into the HTML string itself — the component is
    # fully self-contained, no CDN or sidecar files to serve.
    net = Network(height="600px", width="100%", directed=True, cdn_resources="in_line")
    for node in nodes:
        label = _display_name(node)
        primary = (node.get("labels") or ["?"])[0]
        net.add_node(
            node["id"],
            label=label[:MAX_LABEL_CHARS] + ("…" if len(label) > MAX_LABEL_CHARS else ""),
            color=NODE_COLORS.get(primary, "#9e9e9e"),
            title=_hover_text(node),
        )
    for edge in edges:
        net.add_edge(edge["source"], edge["target"], title=edge["type"])
    if len(nodes) > PHYSICS_MAX_NODES:
        net.toggle_physics(False)
        st.caption(
            f"{len(nodes)} nodes — physics layout disabled for performance; "
            "drop the depth for the animated view."
        )
    components.html(net.generate_html(notebook=False), height=620, scrolling=False)


def _inspect_panel(nodes: list[dict]) -> None:
    by_id = {n["id"]: n for n in nodes}
    chosen_id = st.selectbox(
        "Inspect node",
        options=sorted(by_id, key=lambda i: _display_name(by_id[i]).lower()),
        format_func=lambda i: f"{_display_name(by_id[i])}  ({(by_id[i]['labels'] or ['?'])[0]})",
    )
    node = by_id[chosen_id]

    if "Company" in node.get("labels", []):
        try:
            profile = api_client.company_profile(_display_name(node))
        except ApiError:
            st.caption("No graph profile for this company.")
        else:
            st.metric("Complaints", f"{profile['total_complaints']:,}")
            risk = profile["risk_score"]
            st.metric("Graph risk score", f"{risk:.2f}" if risk is not None else "n/a")
            if profile["violations"]:
                st.caption("Linked violations: " + ", ".join(profile["violations"]))
    else:
        props = node.get("props", {})
        if props:
            st.write(props)

    if st.button("🧭 Explore from here", use_container_width=True):
        st.session_state[PENDING_KEY] = _display_name(node)
        st.rerun()


st.title("🕸️ Graph Explorer")
tour.render("graph_explorer")

# One-shot recenter handoff: must be applied before the text_input exists
# this run (a widget's key can't be written after it's instantiated).
pending = st.session_state.pop(PENDING_KEY, None)
if pending:
    st.session_state[INPUT_KEY] = pending

search_col, depth_col = st.columns([3, 1])
with search_col:
    node_query = st.text_input(
        "Start node (company, product, issue, or regulation id)",
        value=DEFAULT_NODE,
        key=INPUT_KEY,
    )
with depth_col:
    depth = st.slider("Depth", 1, 3, 1, help="Hops from the start node; capped server-side")

if not node_query.strip():
    st.info("Name a node to start exploring — try a company from the Triage Queue.")
    st.stop()

try:
    with st.spinner("Traversing the graph…"):
        neighborhood = api_client.graph_explore(node_query.strip(), depth=depth)
except ApiError as exc:
    if exc.status_code == 401:
        st.warning("Session expired — please sign in again.")
        st.rerun()
    if exc.status_code == 404:
        st.warning(f'No node named "{node_query.strip()}" in the graph.')
    else:
        st.error(f"Could not explore the graph: {exc.detail}")
    st.stop()

nodes, edges = neighborhood["nodes"], neighborhood["edges"]
st.caption(f"{len(nodes)} nodes · {len(edges)} relationships · {LEGEND}")

canvas, panel = st.columns([3, 1])
with canvas:
    _render_network(nodes, edges)
with panel:
    _inspect_panel(nodes)
