"""DiGress / SPECTRE graph metrics, self-contained.

The vendored implementation
(baselines/ConStruct/ConStruct/metrics/spectre_utils.py) cannot be imported on
this cluster: ``graph_tool`` dies with SIGILL (built for a newer instruction
set) and neither ``pyemd`` nor ``pygsp`` is installed.  The definitions below
are transcribed from it so the numbers stay comparable to the DiGress and
ConStruct tables:

  kernel        gaussian_tv, exp(-d^2 / 2 sigma^2) with d = total variation
                (the compute_emd=False path those papers report)
  MMD           biased estimate, histograms normalised to pmf first
  degree        degree histogram, sigma = 1.0
  clustering    clustering-coefficient histogram, 100 bins on [0, 1],
                sigma = 0.1
  spectral      normalised-Laplacian eigenvalue histogram, 200 bins on
                [-1e-5, 2], sigma = 1.0
  orbit         mean orca orbit counts (size-4 graphlets), gaussian kernel,
                sigma = 30.0  -- needs the compiled orca binary

Not reproduced: ``is_sbm_graph``, which needs graph_tool's blockmodel fit.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh

ORCA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "baselines", "ConStruct", "ConStruct", "analysis", "orca",
                    "orca")


# ---------------------------------------------------------------- kernels ---

def gaussian_tv(x, y, sigma=1.0):
    support = max(len(x), len(y))
    x, y = x.astype(float), y.astype(float)
    if len(x) < support:
        x = np.hstack((x, [0.0] * (support - len(x))))
    if len(y) < support:
        y = np.hstack((y, [0.0] * (support - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return float(np.exp(-dist * dist / (2 * sigma * sigma)))


def gaussian(x, y, sigma=1.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    dist = np.linalg.norm(x - y, 2)
    return float(np.exp(-dist * dist / (2 * sigma * sigma)))


def _disc(a, b, kernel, **kw):
    if not len(a) or not len(b):
        return 1e6
    return sum(kernel(s1, s2, **kw) for s1 in a for s2 in b) / (len(a) * len(b))


def compute_mmd(a, b, kernel, is_hist=True, **kw):
    if is_hist:
        a = [s / (np.sum(s) + 1e-6) for s in a]
        b = [s / (np.sum(s) + 1e-6) for s in b]
    return (_disc(a, a, kernel, **kw) + _disc(b, b, kernel, **kw)
            - 2 * _disc(a, b, kernel, **kw))


# ------------------------------------------------------------ descriptors ---

def degree_stats(ref, pred):
    a = [np.array(nx.degree_histogram(G)) for G in ref]
    b = [np.array(nx.degree_histogram(G)) for G in pred if G.number_of_nodes()]
    return compute_mmd(a, b, gaussian_tv)


def clustering_stats(ref, pred, bins=100):
    def hist(G):
        c = list(nx.clustering(G).values())
        h, _ = np.histogram(c, bins=bins, range=(0.0, 1.0), density=False)
        return h
    return compute_mmd([hist(G) for G in ref], [hist(G) for G in pred],
                       gaussian_tv, sigma=1.0 / 10)


def spectral_stats(ref, pred):
    def pmf(G):
        try:
            eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        except Exception:
            eigs = np.zeros(G.number_of_nodes())
        h, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        return h / max(1, h.sum())
    return compute_mmd([pmf(G) for G in ref], [pmf(G) for G in pred],
                       gaussian_tv)


def orca(G):
    """Size-4 orbit counts per node, via the compiled orca binary."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    Gr = nx.convert_node_labels_to_integers(G)
    tmp.write(f"{Gr.number_of_nodes()} {Gr.number_of_edges()}\n")
    for u, v in Gr.edges():
        tmp.write(f"{u} {v}\n")
    tmp.close()
    out = subprocess.check_output([ORCA, "node", "4", tmp.name, "std"],
                                  stderr=subprocess.DEVNULL).decode()
    idx = out.find("orbit counts:") + 15
    rows = [list(map(int, r.split())) for r in out[idx:].strip().split("\n")]
    os.unlink(tmp.name)
    return np.array(rows)


def orbit_stats_all(ref, pred):
    def counts(G):
        try:
            return np.sum(orca(G), axis=0) / G.number_of_nodes()
        except Exception:
            return np.zeros(15)
    return compute_mmd([counts(G) for G in ref], [counts(G) for G in pred],
                       gaussian, is_hist=False, sigma=30.0)


# -------------------------------------------------------------- validity ----

def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def _iso(a, b):
    return nx.is_isomorphic(a, b)


def vun(fake, train, validity_func=None):
    """(unique, unique&novel, V.U.N.) exactly as DiGress defines them."""
    count_valid = count_iso = count_non_unique = 0
    seen = []
    for g in fake:
        unique = True
        for old in seen:
            if nx.faster_could_be_isomorphic(g, old) and _iso(g, old):
                count_non_unique += 1
                unique = False
                break
        if unique:
            seen.append(g)
            novel = True
            for t in train:
                if nx.faster_could_be_isomorphic(g, t) and _iso(g, t):
                    count_iso += 1
                    novel = False
                    break
            if novel and (validity_func is None or validity_func(g)):
                count_valid += 1
    n = float(len(fake))
    return ((n - count_non_unique) / n,
            (n - count_non_unique - count_iso) / n,
            count_valid / n)
