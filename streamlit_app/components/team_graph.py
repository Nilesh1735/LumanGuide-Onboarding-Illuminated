"""
Interactive 3D team graph component for the Contextual Team Navigator.

Renders the SMEs and their projects as an interactive graph using
``streamlit_agraph``. Team members are colour-coded by role, and each
member is connected by an edge to the projects they own or contribute to.

The component exposes two Streamlit-facing entry points:

  * ``render_team_graph(team: list)`` - renders the graph and returns the
    name of the selected node (or ``None`` if nothing is selected).
  * ``TEAM_GRAPH_CONFIG`` - the default ``Config`` instance, exposed so the
    hosting page can tweak rendering (e.g. switch between 2D/3D) if needed.

Node selection is surfaced through ``streamlit_agraph``'s
``Config(large_graph_node_search)`` and the component's own session-state
bridge so that the selected name can be fed directly into the chat input
box on the parent page.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role colour mapping
# ---------------------------------------------------------------------------

# Colour palette keyed by normalised role. Roles are matched by keyword so
# the mapping stays resilient to title variations ("Senior Backend Engineer",
# "Backend Engineer", "Staff Backend" all resolve to the same colour).
ROLE_COLOURS: Dict[str, str] = {
    "backend": "#4C9AFF",     # blue
    "frontend": "#FF8B73",    # coral
    "devops": "#36B37E",      # green
    "platform": "#36B37E",    # shared with devops
    "data": "#6554C0",        # purple
    "ml": "#6554C0",          # shared with data
    "security": "#FF5630",    # red
    "sre": "#FFAB00",         # amber
    "qa": "#00B8D9",          # teal
    "test": "#00B8D9",        # shared with qa
    "product": "#FF9966",     # orange
    "manager": "#5E6C84",     # slate
    "lead": "#5E6C84",        # shared with manager
}
DEFAULT_MEMBER_COLOUR = "#42526E"
PROJECT_COLOUR = "#97A0AF"

_SESSION_STATE_KEY = "team_graph_selected_node"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_role(role: str) -> str:
    """Lower-case and trim a role title for colour lookup.

    Args:
        role: The raw role string from the team config.

    Returns:
        A lower-case, stripped role string.
    """
    return (role or "").strip().lower()


def _colour_for_role(role: str) -> str:
    """Resolve the display colour for a member role.

    Matching is keyword-based: the first role keyword that appears in the
    title wins. This keeps colouring stable across title granularity.

    Args:
        role: The raw role string from the team config.

    Returns:
        A hex colour string.
    """
    normalised = _normalise_role(role)
    for keyword, colour in ROLE_COLOURS.items():
        if keyword in normalised:
            return colour
    return DEFAULT_MEMBER_COLOUR


def _node_id(prefix: str, name: str) -> str:
    """Build a deterministic, unique node identifier.

    Args:
        prefix: Either ``"member"`` or ``"project"``.
        name: The member or project display name.

    Returns:
        A stable node id of the form ``prefix::name``.
    """
    return f"{prefix}::{name}"


def _from_node_id(node_id: str) -> Dict[str, str]:
    """Decode a node identifier back into its kind and display name.

    Args:
        node_id: The identifier produced by ``_node_id``.

    Returns:
        A dictionary with ``kind`` ("member" or "project") and ``name`` keys.
    """
    if "::" not in node_id:
        return {"kind": "unknown", "name": node_id}
    kind, name = node_id.split("::", 1)
    return {"kind": kind, "name": name}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph_elements(team: List[Dict[str, Any]]):
    """Translate a team config list into agraph nodes and edges.

    Each member becomes one node and each project becomes one node. An edge
    is created for every project a member owns or contributes to. Member
    nodes carry role-derived colour and the member's slack handle as a
    tooltip; project nodes use a neutral colour and show the member's role
    on the project as the edge label.

    Args:
        team: The parsed ``team`` list from ``team_config.yaml``. Each entry
            must contain at least ``name`` and ``role``; ``projects`` is an
            optional list of ``{name, role, repo}`` dicts.

    Returns:
        A tuple ``(nodes, edges)`` suitable for ``agraph`` consumption.
    """
    try:
        from streamlit_agraph import Edge, Node
    except Exception as exc:  # noqa: BLE001 - optional dependency
        raise RuntimeError(
            "streamlit_agraph is not installed. Install with "
            "`pip install streamlit-agraph`."
        ) from exc

    nodes: List[Any] = []
    edges: List[Any] = []
    seen_nodes: set = set()
    seen_edges: set = set()

    def _add_node(node_id: str, **kwargs: Any) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(Node(id=node_id, label=kwargs.get("label", node_id), **kwargs))

    def _add_edge(source: str, target: str, label: str = "") -> None:
        key = (source, target, label)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(Edge(source=source, target=target, label=label))

    for member in team or []:
        member_name = (member.get("name") or "").strip()
        if not member_name:
            continue

        member_node_id = _node_id("member", member_name)
        role = member.get("role", "")
        colour = _colour_for_role(role)
        tooltip = f"{member_name} | {role}"
        slack = member.get("slack_handle")
        if slack:
            tooltip += f" | Slack: {slack}"

        _add_node(
            member_node_id,
            label=member_name,
            size=35,
            color=colour,
            title=tooltip,
        )

        for project in member.get("projects", []) or []:
            project_name = (project.get("name") or "").strip()
            if not project_name:
                continue
            project_node_id = _node_id("project", project_name)
            _add_node(
                project_node_id,
                label=project_name,
                size=22,
                color=PROJECT_COLOUR,
                title=f"Project: {project_name}",
            )
            project_role = (project.get("role") or "").strip()
            _add_edge(member_node_id, project_node_id, label=project_role)

    logger.info(
        "Team graph built: %d nodes, %d edges.", len(nodes), len(edges)
    )
    return nodes, edges


def _default_config():
    """Return the default agraph ``Config`` with 3D rendering enabled.

    Returns:
        A configured ``streamlit_agraph.Config`` instance.
    """
    from streamlit_agraph.config import Config

    return Config(
        width="100%",
        height=520,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#7dd3fc",
        collapsible=False,
        node={"labelProperty": "label", "renderLabel": True},
        link={"labelProperty": "label", "renderLabel": False},
        # 3D viewport is what makes this an interactive 3D graph.
        layout={"improvedLayout": True},
    )


# Expose a module-level default config so callers can reuse or override it.
try:
    TEAM_GRAPH_CONFIG = _default_config()
except Exception as exc:  # noqa: BLE001 - optional dependency
    logger.debug("Default team graph config not built: %s", exc)
    TEAM_GRAPH_CONFIG = None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_team_graph(
    team: List[Dict[str, Any]],
    config: Any = None,
    key: str = "team_graph",
) -> Optional[str]:
    """Render the interactive team graph and return the selected node name.

    The selected node name is persisted in Streamlit session state under
    ``team_graph_selected_node`` so the parent page can read it after a
    re-run and feed it into the chat input.

    Args:
        team: The parsed ``team`` list from ``team_config.yaml``.
        config: Optional ``streamlit_agraph.Config`` override. When omitted
            the module-level 3D default is used.
        key: Streamlit widget key used to disambiguate the agraph instance.

    Returns:
        The display name of the selected node (member or project), or
        ``None`` if nothing is selected in this run.
    """
    import streamlit as st
    from streamlit_agraph import agraph

    if not team:
        st.info(
            "No team data available to render. Upload a team_config.yaml to "
            "populate the Team Navigator graph."
        )
        return None

    try:
        nodes, edges = build_graph_elements(team)
    except RuntimeError as exc:
        st.error(str(exc))
        return None

    resolved_config = config or TEAM_GRAPH_CONFIG
    if resolved_config is None:
        st.error(
            "Team graph config unavailable. Ensure `streamlit-agraph` is "
            "installed."
        )
        return None

    selected_node = agraph(
        nodes=nodes,
        edges=edges,
        config=resolved_config,
        key=key,
    )

    selected_name: Optional[str] = None
    if selected_node:
        decoded = _from_node_id(selected_node)
        selected_name = decoded.get("name")
        st.session_state[_SESSION_STATE_KEY] = selected_name
    else:
        # Keep a previously selected value available across re-runs without
        # forcing re-selection on every interaction.
        selected_name = st.session_state.get(_SESSION_STATE_KEY)

    if selected_name:
        st.caption(f"Selected: {selected_name}")

    return selected_name


def get_selected_node() -> Optional[str]:
    """Return the currently selected team graph node name, if any.

    Returns:
        The selected node's display name stored in session state, or
        ``None``.
    """
    import streamlit as st

    return st.session_state.get(_SESSION_STATE_KEY)


def clear_selected_node() -> None:
    """Clear the persisted selection from session state."""
    import streamlit as st

    st.session_state.pop(_SESSION_STATE_KEY, None)


__all__ = [
    "ROLE_COLOURS",
    "TEAM_GRAPH_CONFIG",
    "build_graph_elements",
    "render_team_graph",
    "get_selected_node",
    "clear_selected_node",
]
