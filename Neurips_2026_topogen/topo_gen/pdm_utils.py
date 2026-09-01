"""Utilities for the Persistence Diagram Matching (PDM) loss and
persistence-landscape-based graph embeddings (µ_G₀) used in TAGG.

Public API
----------
compute_pd_nx(G)
    Compute H0/H1 persistence diagrams for a NetworkX graph using
    degree filtration.

persistence_landscape(pd, num_landscapes, resolution)
    Vectorize a persistence diagram into a persistence landscape vector.

graph_mu(G, num_landscapes, resolution)
    Full pipeline: G → (PD0, PD1) → concatenated landscape µ_G₀.

mean_mu(graphs, num_landscapes, resolution)
    Compute the mean µ̄ over a collection of graphs (used at inference).

pdm_loss(pd0_true, pd1_true, pd0_pred, pd1_pred)
    1-Wasserstein PDM loss between two pairs of persistence diagrams,
    summed over H0 and H1 dimensions.

pdm_loss_from_adj(adj_true, adj_pred, node_mask)
    Convenience wrapper: takes dense adjacency tensors, converts to
    NetworkX, computes PDs, returns scalar PDM loss (numpy float).
"""

import math
import numpy as np
import networkx as nx

from gudhi.representations import Landscape

from topo_gen.filtrations import degree_filtration
from topo_gen.persistence import persistence_diagrams


# ---------------------------------------------------------------------------
# PD computation
# ---------------------------------------------------------------------------

def compute_pd_nx(G: nx.Graph) -> tuple[list, list]:
    """Return (pd0, pd1) for G under the degree filtration.

    H0 points: (birth, death) — death=inf for surviving components.
    H1 points: (birth, inf)   — all H1 bars persist forever in 1-complexes.

    Empty graphs return ([], []).
    """
    if G.number_of_nodes() == 0:
        return [], []
    node_time, edge_time, _ = degree_filtration(G)
    return persistence_diagrams(node_time, edge_time)


# ---------------------------------------------------------------------------
# Persistence landscape
# ---------------------------------------------------------------------------

def _pd_to_finite_array(pd: list) -> np.ndarray:
    """Convert a PD list to a float32 array, dropping points with death=inf.

    Gudhi's Landscape transformer cannot handle inf values — they produce NaN.
    For H0 the surviving component (death=inf) is dropped; it contributes no
    finite bar length.
    For H1 all bars have death=inf in a 1-complex, so the finite array will
    be empty and the landscape will be all-zeros.
    """
    finite = [(b, d) for b, d in pd if not math.isinf(d)]
    if not finite:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(finite, dtype=np.float32)


def persistence_landscape(
    pd: list,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Vectorize a persistence diagram into a persistence landscape.

    Parameters
    ----------
    pd : list of (birth, death) pairs (inf deaths are dropped)
    num_landscapes : L — number of landscape layers
    resolution : S — number of discretization points per layer

    Returns
    -------
    np.ndarray of shape (num_landscapes * resolution,) — µ for one PD dimension
    """
    arr = _pd_to_finite_array(pd)
    out_dim = num_landscapes * resolution

    if arr.shape[0] == 0:
        return np.zeros(out_dim, dtype=np.float32)

    lsc = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    vec = lsc.fit_transform([arr])   # shape (1, L*S)
    return vec[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Graph-level µ_G₀
# ---------------------------------------------------------------------------

def graph_mu(
    G: nx.Graph,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Compute the topology embedding µ_G₀ for a single graph.

    Concatenates persistence landscapes for H0 and H1, giving a vector of
    shape (2 * num_landscapes * resolution,).

    At inference time (G₀ unknown) replace with mean_mu() over training data.
    """
    pd0, pd1 = compute_pd_nx(G)
    mu0 = persistence_landscape(pd0, num_landscapes, resolution)
    mu1 = persistence_landscape(pd1, num_landscapes, resolution)
    return np.concatenate([mu0, mu1], axis=0)  # (2 * L * S,)


def mean_mu(
    graphs: list,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Compute the mean µ̄ over a list of NetworkX graphs.

    Used at inference time when the original G₀ is unknown (TAGG §3.2).
    Returns array of shape (2 * num_landscapes * resolution,).
    """
    mus = [graph_mu(G, num_landscapes, resolution) for G in graphs]
    return np.mean(mus, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# PDM loss (1-Wasserstein with diagonal padding)
# ---------------------------------------------------------------------------

def _wasserstein1_pd(pd_a: list, pd_b: list) -> float:
    """1-Wasserstein distance between two persistence diagrams.

    Unmatched points are paired to their nearest point on the diagonal
    (b, d) → ((b+d)/2, (b+d)/2) with cost = persistence/2.

    Uses scipy linear_sum_assignment for the optimal matching.
    Handles empty diagrams gracefully.
    """
    from scipy.optimize import linear_sum_assignment

    # Drop inf-death points — they carry no finite bar length.
    def finite(pd):
        return [(b, d) for b, d in pd if not math.isinf(d)]

    fa = finite(pd_a)
    fb = finite(pd_b)

    # Diagonal projections
    diag_a = [((b + d) / 2, (b + d) / 2) for b, d in fa]
    diag_b = [((b + d) / 2, (b + d) / 2) for b, d in fb]

    pts_a = fa + diag_b   # real points of A  +  diagonals of B
    pts_b = fb + diag_a   # real points of B  +  diagonals of A

    n = len(pts_a)
    if n == 0:
        return 0.0

    # Cost matrix: L∞ distance (standard for persistence Wasserstein)
    cost = np.zeros((n, n), dtype=np.float64)
    for i, (b1, d1) in enumerate(pts_a):
        for j, (b2, d2) in enumerate(pts_b):
            cost[i, j] = max(abs(b1 - b2), abs(d1 - d2))

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum())


def pdm_loss(
    pd0_true: list,
    pd1_true: list,
    pd0_pred: list,
    pd1_pred: list,
) -> float:
    """PDM loss = W₁(PD0_true, PD0_pred) + W₁(PD1_true, PD1_pred).

    Both terms use the L∞-ground-metric 1-Wasserstein with diagonal padding.
    """
    return _wasserstein1_pd(pd0_true, pd0_pred) + _wasserstein1_pd(pd1_true, pd1_pred)


# ---------------------------------------------------------------------------
# Batched convenience wrapper for training
# ---------------------------------------------------------------------------

def _adj_to_nx(adj: np.ndarray, node_mask: np.ndarray) -> nx.Graph:
    """Convert a dense binary adjacency (n×n) to NetworkX, respecting mask."""
    n = int(node_mask.sum())
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                G.add_edge(i, j)
    return G


def pdm_loss_from_adj(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
    node_mask: np.ndarray,
) -> float:
    """Compute PDM loss between two dense binary adjacency matrices.

    Parameters
    ----------
    adj_true  : (n, n) binary array — clean graph G₀
    adj_pred  : (n, n) binary array — predicted graph p̂G₀ (argmax of logits)
    node_mask : (n,) bool array — which nodes are real

    Returns
    -------
    Scalar PDM loss (float).
    """
    G_true = _adj_to_nx(adj_true, node_mask)
    G_pred = _adj_to_nx(adj_pred, node_mask)
    pd0_true, pd1_true = compute_pd_nx(G_true)
    pd0_pred, pd1_pred = compute_pd_nx(G_pred)
    return pdm_loss(pd0_true, pd1_true, pd0_pred, pd1_pred)
