"""Utilities shared across graph generation algorithms."""

from __future__ import annotations

from typing import Callable

import networkx as nx

from topo_gen.persistence import persistence_diagrams, check_match
from topo_gen.result import GenerationResult


def build_index(pd0: list, pd1: list) -> tuple:
    """Pre-index persistence diagrams by timestep.

    Returns
    -------
    n   : int           max timestep
    B0  : dict[k, list] PD0 pairs born at k
    D0  : dict[k, list] PD0 pairs dying at k
    B1  : dict[k, int]  number of H1 births at k
    """
    n = max(
        max(b for b, d in pd0),
        max((d for b, d in pd0 if d != float("inf")), default=0),
        max((b for b, d in pd1), default=0),
    )
    B0 = {k: [] for k in range(n + 1)}
    D0 = {k: [] for k in range(n + 1)}
    B1 = {k: 0  for k in range(n + 1)}
    for b, d in pd0:
        B0[b].append((b, d))
        if d != float("inf"):
            D0[int(d)].append((b, int(d)))
    for b, d in pd1:
        B1[int(b)] += 1
    return n, B0, D0, B1


def generate_graphs(
    generation_fn: Callable[..., GenerationResult],
    pd0: list,
    pd1: list,
    n: int = 20,
    base_seed: int = 0,
) -> list[GenerationResult]:
    """Generate n graphs by calling generation_fn n times with distinct seeds.

    Parameters
    ----------
    generation_fn : callable with signature (pd0, pd1, seed) -> GenerationResult
    pd0           : H0 persistence diagram
    pd1           : H1 persistence diagram
    n             : number of attempts
    base_seed     : seeds used are base_seed, base_seed+1, ..., base_seed+n-1

    Returns
    -------
    list of GenerationResult (Success or Failure), one per attempt.
    Use result.ok to distinguish them.
    """
    return [generation_fn(pd0, pd1, seed=base_seed + i) for i in range(n)]


def build_exhaustion(
    G: nx.Graph,
    node_time: dict,
    edge_time: dict,
    n_steps: int,
) -> list[nx.Graph]:
    """Return the filtration exhaustion sequence G_0 ⊆ G_1 ⊆ ... ⊆ G_{T-1}.

    Parameters
    ----------
    G         : full graph (nodes/edges used as reference)
    node_time : dict  node -> int filtration time
    edge_time : dict  (u, v) -> int filtration time
    n_steps   : number of timesteps T

    Returns
    -------
    list of nx.Graph, one per timestep k in 0..n_steps-1
    """
    subgraphs = []
    for k in range(n_steps):
        Gk = nx.Graph()
        Gk.add_nodes_from(v for v, t in node_time.items() if t <= k)
        Gk.add_edges_from(e for e, t in edge_time.items() if t <= k)
        subgraphs.append(Gk)
    return subgraphs


def isomorphism_classes(graphs: list[nx.Graph]) -> list[list[int]]:
    """Partition a list of graphs into isomorphism equivalence classes.

    Parameters
    ----------
    graphs : list of nx.Graph

    Returns
    -------
    list of groups, where each group is a list of indices into `graphs`
    that are mutually isomorphic
    """
    classes  = []
    assigned = [False] * len(graphs)
    for i, Gi in enumerate(graphs):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(graphs)):
            if not assigned[j] and nx.is_isomorphic(Gi, graphs[j]):
                group.append(j)
                assigned[j] = True
        classes.append(group)
    return classes


def ph_roundtrip_check(pd0: list, pd1: list, successes: list) -> int:
    """Count how many successful generations reproduce the target PD exactly.

    Parameters
    ----------
    pd0       : target H0 persistence diagram
    pd1       : target H1 persistence diagram
    successes : list of Success results (with .node_time, .edge_time)

    Returns
    -------
    number of successes whose computed PD matches (pd0, pd1)
    """
    return sum(
        check_match(pd0, pd1, *persistence_diagrams(r.node_time, r.edge_time), verbose=False)
        for r in successes
    )
