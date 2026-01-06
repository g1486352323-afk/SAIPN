from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def detect_communities_louvain(
    G: nx.Graph,
    *,
    resolution: float = 1.0,
    min_community_size: int = 3,
    max_communities: Optional[int] = None,
) -> Dict[int, List[str]]:
    if G.number_of_nodes() == 0:
        return {}

    G_und = G.to_undirected() if isinstance(G, nx.DiGraph) else G

    comm_sets: List[Set[str]] = []
    try:
        import networkx.algorithms.community as nx_comm

        comm_sets = list(nx_comm.louvain_communities(G_und, resolution=resolution, weight=None))
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities

            comm_sets = list(greedy_modularity_communities(G_und))
        except Exception:
            return {}

    comm_sets = [set(c) for c in comm_sets if len(c) >= int(min_community_size)]
    comm_sets.sort(key=len, reverse=True)

    if max_communities is not None and int(max_communities) > 0:
        comm_sets = comm_sets[: int(max_communities)]

    return {i: [str(n) for n in c] for i, c in enumerate(comm_sets)}


def calculate_cnlr_fast(
    current_edges_set: Set[Tuple[str, str]],
    prev_edges_set: Set[Tuple[str, str]],
    communities_dict: Mapping[int, Sequence[str]],
) -> float:
    if not communities_dict or not current_edges_set:
        return 0.0

    new_edges = current_edges_set - prev_edges_set
    if not new_edges:
        return 0.0

    cnlr_scores: List[float] = []
    for nodes in communities_dict.values():
        if len(nodes) < 3:
            continue
        node_set = set(nodes)
        possible = len(node_set) * (len(node_set) - 1)
        if possible <= 0:
            continue
        internal_new = sum(1 for u, v in new_edges if u in node_set and v in node_set)
        cnlr_scores.append(internal_new / possible)

    val = float(np.mean(cnlr_scores)) if cnlr_scores else 0.0
    return 0.0 if np.isnan(val) else val


def compute_basic_graph_metrics(
    G: nx.DiGraph,
    *,
    communities: Optional[Mapping[int, Sequence[str]]] = None,
    prev_edges_snapshot: Optional[Set[Tuple[str, str]]] = None,
    compute_cnlr: bool = True,
) -> Tuple[Dict[str, Any], Set[Tuple[str, str]]]:
    num_edges = int(G.number_of_edges())
    metrics: Dict[str, Any] = {
        "nodes": int(G.number_of_nodes()),
        "edges": num_edges,
        "avg_pagerank": 0.0,
        "assortativity": 0.0,
        "modularity": 0.0,
        "cnlr": 0.0,
        "comp_intensity": 0.0,
        "prop_scope": 0,
        "collab_intensity": 0,
    }

    if num_edges <= 0:
        return metrics, (prev_edges_snapshot or set())

    comms: Dict[int, List[str]] = {}
    if communities is not None:
        comms = {int(k): [str(x) for x in v] for k, v in communities.items()}
    else:
        comms = detect_communities_louvain(G)

    try:
        pr = nx.pagerank(G, weight="weight")
        metrics["avg_pagerank"] = float(np.mean(list(pr.values()))) if pr else 0.0
    except Exception:
        metrics["avg_pagerank"] = 0.0

    try:
        import networkx.algorithms.community as nx_comm

        if comms:
            comm_sets = [set(v) for v in comms.values()]
            metrics["modularity"] = float(nx_comm.modularity(G.to_undirected(), comm_sets))
    except Exception:
        metrics["modularity"] = 0.0

    try:
        assort = nx.degree_assortativity_coefficient(G)
        metrics["assortativity"] = 0.0 if np.isnan(assort) else float(assort)
    except Exception:
        metrics["assortativity"] = 0.0

    try:
        metrics["comp_intensity"] = float(G.size(weight="weight"))
    except Exception:
        metrics["comp_intensity"] = 0.0

    try:
        und = G.to_undirected()
        metrics["collab_intensity"] = int(sum(nx.triangles(und).values()) / 3)
    except Exception:
        metrics["collab_intensity"] = 0

    try:
        if nx.is_directed(G):
            comps = nx.weakly_connected_components(G)
        else:
            comps = nx.connected_components(G)  # type: ignore[arg-type]
        metrics["prop_scope"] = int(max((len(c) for c in comps), default=0))
    except Exception:
        metrics["prop_scope"] = 0

    curr_edges_snapshot = set((str(u), str(v)) for u, v in G.edges())
    if compute_cnlr and prev_edges_snapshot is not None and comms:
        metrics["cnlr"] = float(calculate_cnlr_fast(curr_edges_snapshot, prev_edges_snapshot, comms))

    return metrics, curr_edges_snapshot
