"""Filtration functions for graphs.

Three filtration types are provided, all returning the same triple:
    node_time : dict  node -> int   filtration time of each node
    edge_time : dict  (u,v) -> int  filtration time of each edge  (u < v)
    n_steps   : int                 number of distinct time steps used

1. random_filtration
   Nodes get times drawn uniformly at random; edge times are chosen
   independently but strictly greater than both endpoint node times.
   All times are compacted to be contiguous.

2. vertex_filtration
   Nodes get times drawn uniformly at random; each edge appears at
   exactly max(node_time[u], node_time[v]).
   All times are compacted to be contiguous.

3. degree_filtration
   Each node appears at a time equal to its degree in G.  Edge times
   are max(node_time[u], node_time[v]).  Times are NOT compacted —
   gaps are possible when no node has a given degree value.
"""

import numpy as np
import networkx as nx


def _compact(raw_node: dict, raw_edge: dict) -> tuple[dict, dict, int]:
    """Remap raw integer times to contiguous 0-based integers."""
    all_times = sorted(set(raw_node.values()) | set(raw_edge.values()))
    remap     = {t: i for i, t in enumerate(all_times)}
    return (
        {v: remap[t] for v, t in raw_node.items()},
        {e: remap[t] for e, t in raw_edge.items()},
        len(all_times),
    )
 

def _edge_times_max(G: nx.Graph, node_time: dict) -> dict:
    """Return edge_time[(u,v)] = max(node_time[u], node_time[v]) for all edges."""
    return {
        (min(u, v), max(u, v)): max(node_time[u], node_time[v])
        for u, v in G.edges()
    }


def random_filtration(
    G: nx.Graph,
    n_steps: int = None,
    seed: int = None,
) -> tuple[dict, dict, int]:
    """Random filtration with independent node and edge times.

    Node times are drawn uniformly from {0, ..., n_steps-1}.
    Edge times are drawn uniformly from {max(t_u, t_v), ..., ceil}
    where ceil = max node time + 1, so every edge always has a valid slot.
    All times are compacted to be contiguous.
    """
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if n_steps is None:
        n_steps = len(nodes)

    raw_node = {v: int(rng.integers(0, n_steps)) for v in nodes}
    ceil     = max(raw_node.values()) + 1

    raw_edge = {}
    for u, v in G.edges():
        lo = max(raw_node[u], raw_node[v])
        raw_edge[(min(u, v), max(u, v))] = int(rng.integers(lo, ceil + 1))

    return _compact(raw_node, raw_edge)


def vertex_filtration(
    G: nx.Graph,
    n_steps: int = None,
    seed: int = None,
) -> tuple[dict, dict, int]:
    """Vertex-based filtration: each edge appears when its later endpoint does.

    Node times are drawn uniformly at random.  Edge time = max(t_u, t_v).
    All times are compacted to be contiguous.
    """
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if n_steps is None:
        n_steps = len(nodes)

    raw_times = rng.integers(0, n_steps, size=len(nodes))
    raw_node  = {v: int(t) for v, t in zip(nodes, raw_times)}

    return _compact(raw_node, _edge_times_max(G, raw_node))


def degree_filtration(G: nx.Graph) -> tuple[dict, dict, int]:
    """Degree-based filtration: node time equals its degree in G.

    Times are NOT compacted — the filtration value is the actual degree,
    so the time axis may have gaps.
    """
    node_time = dict(G.degree())
    edge_time = _edge_times_max(G, node_time)
    n_steps   = max(node_time.values()) + 1 if node_time else 0
    return node_time, edge_time, n_steps
