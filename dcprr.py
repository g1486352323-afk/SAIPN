from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np


def _compute_dcprr_score(
    G: nx.DiGraph,
    community_nodes: Sequence[str],
    *,
    walk_length: int,
    num_walks: int,
    rng: np.random.Generator,
    neighbors_cache: Dict[str, List[str]],
    max_start_nodes: int,
    deadend_policy: str,
) -> float:
    if len(community_nodes) < 2:
        return 0.0

    start_nodes = list(community_nodes)
    if max_start_nodes > 0 and len(start_nodes) > int(max_start_nodes):
        idx = rng.choice(len(start_nodes), size=int(max_start_nodes), replace=False)
        start_nodes = [start_nodes[int(i)] for i in idx]

    comm_set = set(str(x) for x in community_nodes)
    retention_rates: List[float] = []

    for node in start_nodes:
        node = str(node)
        neigh0 = neighbors_cache.get(node)
        if neigh0 is None:
            neigh0 = [str(n) for n in G.neighbors(node)]
            neighbors_cache[node] = neigh0

        if not neigh0:
            if deadend_policy == "break":
                continue
            if deadend_policy == "fail":
                retention_rates.append(0.0)
                continue
            raise ValueError(f"Unknown deadend_policy: {deadend_policy}")

        total_steps = 0
        retained_steps = 0

        for _ in range(int(num_walks)):
            current = node
            for _ in range(int(walk_length)):
                neighs = neighbors_cache.get(current)
                if neighs is None:
                    neighs = [str(n) for n in G.neighbors(current)]
                    neighbors_cache[current] = neighs

                if not neighs:
                    if deadend_policy == "break":
                        break
                    if deadend_policy == "fail":
                        total_steps += 1
                        break
                    raise ValueError(f"Unknown deadend_policy: {deadend_policy}")

                nxt = neighs[int(rng.integers(0, len(neighs)))]
                total_steps += 1
                if nxt in comm_set:
                    retained_steps += 1
                current = nxt

        if total_steps > 0:
            retention_rates.append(retained_steps / total_steps)

    return float(np.mean(retention_rates)) if retention_rates else 0.0


def calculate_all_dcprr_scores(
    G: nx.DiGraph,
    communities: Mapping[int, Sequence[str]],
    *,
    walk_length: int = 3,
    num_walks: int = 10,
    seed: int = 42,
    min_community_size: int = 4,
    max_start_nodes: int = 200,
    deadend_policy: str = "fail",
) -> Dict[int, float]:
    rng = np.random.default_rng(int(seed))
    neighbors_cache: Dict[str, List[str]] = {}
    scores: Dict[int, float] = {}

    for comm_id, nodes in communities.items():
        if len(nodes) < int(min_community_size):
            continue
        score = _compute_dcprr_score(
            G,
            nodes,
            walk_length=int(walk_length),
            num_walks=int(num_walks),
            rng=rng,
            neighbors_cache=neighbors_cache,
            max_start_nodes=int(max_start_nodes),
            deadend_policy=str(deadend_policy),
        )
        scores[int(comm_id)] = float(score)

    return scores
