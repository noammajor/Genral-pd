"""Distributional plausibility score s(H) for the reward.

    R(H) = 1[trajectory completed] * exp(beta * s(H))

The indicator keeps exact topological compliance as a HARD support constraint;
s(H) shapes the density WITHIN that support.

Descriptors must be DEGREE-ORTHOGONAL.  Every PD-compliant graph the process can
emit has the identical degree multiset and edge count as the target (capacity
starts at each vertex's intended degree, exactly |E| edges are placed, and
capacity never goes negative, so all slack is zero).  Degree-based statistics are
therefore constant across the whole solution set: they carry no signal, and
degree MMD is identically zero by construction rather than as a result.

What does vary -- measured over the solution set of a fixed target -- is
triangles, clustering, spectral shape, assortativity and 4-cycles.  Uniform
sampling is measurably biased on them: on an ER(0.4) target at n=10 the source
graph sat at the 0th percentile of the sampler's triangle and assortativity
distributions, i.e. outside its entire range.
"""

from __future__ import annotations

import numpy as np

DESCRIPTOR_NAMES = [
    "triangles", "avg_clustering", "spectral_gap", "spectral_max",
    "spectral_mean", "assortativity", "four_cycles",
]
NUM_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def descriptors(adj: np.ndarray) -> np.ndarray:
    """(NUM_DESCRIPTORS,) float64 from a dense boolean/0-1 adjacency."""
    A = adj.astype(np.float64)
    n = A.shape[0]
    deg = A.sum(1)
    A2 = A @ A
    A3d = (A2 @ A).diagonal()

    tri = float(A3d.sum() / 6.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = deg * (deg - 1)
        loc = np.where(den > 0, A3d / np.where(den > 0, den, 1.0), 0.0)
    clust = float(loc.mean()) if n else 0.0

    d = np.where(deg > 0, deg, 1.0)
    L = np.eye(n) - A / np.sqrt(np.outer(d, d))
    ev = np.sort(np.linalg.eigvalsh(L))
    gap = float(ev[1]) if n > 1 else 0.0
    emax = float(ev[-1]) if n else 0.0
    emean = float(ev[1:].mean()) if n > 1 else 0.0

    src, dst = np.nonzero(A)
    if len(src):
        x, y = deg[src], deg[dst]
        sd = x.std() * y.std()
        assort = float(((x * y).mean() - x.mean() * y.mean()) / sd) if sd > 1e-12 else 0.0
    else:
        assort = 0.0

    four = float((np.triu(A2, 1) ** 2).sum())
    return np.array([tri, clust, gap, emax, emean, assort, four], dtype=np.float64)


class DescriptorScorer:
    """s(H) = -||(phi(H) - mu) / sigma||^2 / D, fitted on a reference family.

    Negative squared Mahalanobis-ish distance, so s <= 0 with 0 the best
    achievable.  Dividing by D keeps beta comparable across descriptor sets.

    Fit on the TRAIN split and evaluate MMD on TEST -- reusing the same
    descriptors for both reward and metric is training on the metric, and must
    be reported as such if it is done.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.sigma = np.where(self.sigma > 1e-9, self.sigma, 1.0)

    @classmethod
    def fit(cls, adjs) -> "DescriptorScorer":
        P = np.stack([descriptors(a) for a in adjs])
        return cls(P.mean(0), P.std(0))

    def score(self, adj: np.ndarray) -> float:
        z = (descriptors(adj) - self.mu) / self.sigma
        return float(-(z ** 2).mean())

    def score_many(self, adjs) -> np.ndarray:
        return np.array([self.score(a) for a in adjs], dtype=np.float64)

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist(),
                "names": DESCRIPTOR_NAMES}

    @classmethod
    def from_dict(cls, d: dict) -> "DescriptorScorer":
        return cls(np.array(d["mu"]), np.array(d["sigma"]))


class ConstantScorer:
    """s(H) = 0 for every graph -- reduces the reward to completion-only.

    This is the ablation that isolates 'does the policy learn to finish' from
    'does it learn to match the distribution'.
    """

    def score(self, adj: np.ndarray) -> float:
        return 0.0

    def score_many(self, adjs) -> np.ndarray:
        return np.zeros(len(adjs), dtype=np.float64)
