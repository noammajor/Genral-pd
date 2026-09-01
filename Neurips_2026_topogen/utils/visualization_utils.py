"""Visualization utilities for TopoGen.

Functions for plotting persistence diagrams, filtration exhaustion sequences,
and generated graph grids. All functions are algorithm-agnostic and work with
any filtration type.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def print_diagrams(pd0: list, pd1: list) -> None:
    """Print H0 and H1 persistence diagrams to stdout."""
    print("H0 persistence diagram:")
    for b, d in sorted(pd0):
        ds = f"{d:.0f}" if d != float("inf") else "∞"
        print(f"  ({b}, {ds})   lifetime={ds if d == float('inf') else d - b}")
    print(f"\nH1 persistence diagram:  ({len(pd1)} classes, all persist forever)")
    for b, _ in sorted(pd1):
        print(f"  ({b}, ∞)")


def plot_persistence_diagrams(pd0: list, pd1: list, T: int, title: str = "") -> None:
    """Scatter plot of H0 and H1 persistence diagrams.

    Parameters
    ----------
    pd0   : list of (birth, death) pairs for H0
    pd1   : list of (birth, inf) pairs for H1
    T     : number of filtration steps (used to position the ∞ proxy)
    title : figure suptitle
    """
    INF_PROXY = T + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    diagrams = [
        (pd0, "H0 — connected components", "steelblue"),
        (pd1, "H1 — independent cycles",   "darkorange"),
    ]

    for ax, (pd, label, color) in zip(axes, diagrams):
        if pd:
            births = [b for b, _ in pd]
            deaths = [d if d != float("inf") else INF_PROXY for _, d in pd]
            ax.scatter(births, deaths, c=color, s=40, zorder=3)

        lo, hi = -0.3, INF_PROXY + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=1)
        ax.axhline(INF_PROXY, color="gray", lw=0.6, ls=":", zorder=1)
        ax.text(lo + 0.1, INF_PROXY + 0.05, "∞", fontsize=9, color="gray")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, INF_PROXY + 0.5)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.set_title(label, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(title or "Persistence diagrams", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_exhaustion(
    G: nx.Graph,
    node_time: dict,
    edge_time: dict,
    n_steps: int,
    title: str = "",
) -> None:
    """Plot the filtration exhaustion sequence, one subplot per timestep.

    Parameters
    ----------
    G         : full graph (used for a consistent spring layout)
    node_time : dict  node -> int filtration time
    edge_time : dict  (u, v) -> int filtration time
    n_steps   : number of distinct timesteps
    title     : figure suptitle
    """
    subgraphs = []
    for k in range(n_steps):
        Gk = nx.Graph()
        Gk.add_nodes_from(v for v, t in node_time.items() if t <= k)
        Gk.add_edges_from(e for e, t in edge_time.items() if t <= k)
        subgraphs.append(Gk)

    pos  = nx.spring_layout(G, seed=42)
    cmap = plt.cm.plasma
    vmin, vmax = 0, n_steps - 1

    ncols = min(n_steps, 6)
    nrows = (n_steps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()

    for k, (ax, Gk) in enumerate(zip(axes, subgraphs)):
        node_list = list(Gk.nodes())
        edge_list = list(Gk.edges())
        nc = [node_time[v] for v in node_list]
        ec = [edge_time[(min(u, v), max(u, v))] for u, v in edge_list]

        nx.draw_networkx_nodes(Gk, pos, nodelist=node_list,
                               node_color=nc, cmap=cmap,
                               vmin=vmin, vmax=vmax, node_size=80, ax=ax)
        if edge_list:
            nx.draw_networkx_edges(Gk, pos, edgelist=edge_list,
                                   edge_color=ec, edge_cmap=cmap,
                                   edge_vmin=vmin, edge_vmax=vmax,
                                   width=1.5, ax=ax)
        ax.set_title(f"t ≤ {k}  |  {Gk.number_of_nodes()}v  {Gk.number_of_edges()}e", fontsize=8)
        ax.axis("off")

    for ax in axes[n_steps:]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:n_steps].tolist(), label="filtration time", shrink=0.6, pad=0.02)
    fig.suptitle(title or f"Exhaustion sequence  (T={n_steps})", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_generated_graphs(
    successes: list,
    ref_graph: nx.Graph = None,
    ref_node_time: dict = None,
    ncols: int = 5,
    title: str = "",
) -> None:
    """Plot successfully generated graphs in a grid.

    The reference graph (if provided) is shown first with a red border.

    Parameters
    ----------
    successes     : list of Success result objects (with .H, .node_time)
    ref_graph     : optional reference graph to show first
    ref_node_time : node_time dict for ref_graph (required if ref_graph given)
    ncols         : number of columns in the grid
    title         : figure suptitle
    """
    show_ref = ref_graph is not None
    total    = len(successes) + (1 if show_ref else 0)
    nrows    = (total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    cmap = plt.cm.plasma

    def _draw(ax, Gx, ntx, label, is_ref=False):
        T_max = max(ntx.values()) if ntx else 0
        pos   = nx.spring_layout(Gx, seed=42)
        nc    = [ntx[v] for v in Gx.nodes()]
        nx.draw_networkx_nodes(Gx, pos, node_color=nc, cmap=cmap,
                               vmin=0, vmax=max(T_max, 1), node_size=50, ax=ax)
        if Gx.number_of_edges():
            ec = [max(ntx[u], ntx[v]) for u, v in Gx.edges()]
            nx.draw_networkx_edges(Gx, pos, edge_color=ec, edge_cmap=cmap,
                                   edge_vmin=0, edge_vmax=max(T_max, 1), width=1.0, ax=ax)
        ax.set_title(label, fontsize=8, color="crimson" if is_ref else "black")
        if is_ref:
            for spine in ax.spines.values():
                spine.set_edgecolor("crimson")
                spine.set_linewidth(2)
                spine.set_visible(True)
        ax.axis("off")

    idx = 0
    if show_ref:
        _draw(axes[idx], ref_graph, ref_node_time,
              f"REF  {ref_graph.number_of_nodes()}v {ref_graph.number_of_edges()}e",
              is_ref=True)
        idx += 1

    for i, r in enumerate(successes):
        _draw(axes[idx], r.H, r.node_time,
              f"#{i+1}  {r.H.number_of_nodes()}v {r.H.number_of_edges()}e")
        idx += 1

    for ax in axes[idx:]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:idx].tolist(), label="filtration time", shrink=0.5, pad=0.02)
    fig.suptitle(title or f"{len(successes)} PD-equivalent generated graphs", fontsize=13)
    plt.tight_layout()
    plt.show()
