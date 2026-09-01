"""Dataset loading utilities for TopoGen.

Handles loading processed PyTorch Geometric splits and converting them to
NetworkX graphs for use in generation experiments.
"""

from __future__ import annotations

import os

import networkx as nx
import torch
import torch_geometric

# Datasets that use the SPECTRE-style single {split}.pt file
SPECTRE_DATASETS = {"planar", "sbm", "comm20", "ego", "enzymes"}

# Molecular datasets processed by ConStruct loaders
MOLECULAR_DATASETS = {"qm9", "guacamol", "moses"}


def load_split(data_root: str, dataset_name: str, split: str) -> list:
    """Load a processed split from disk.

    Dispatches to the correct loader based on dataset type:
    - SPECTRE/graph datasets  → single ``{split}.pt`` file
    - Molecular datasets      → ``{split}_{suffix}.pt`` file (ConStruct format)

    Parameters
    ----------
    data_root    : path to the top-level data directory
    dataset_name : e.g. "planar", "sbm", "comm20", "ego", "enzymes",
                   "qm9", "guacamol", "moses"
    split        : "train", "val", or "test"

    Returns
    -------
    list of torch_geometric.data.Data objects
    """
    if dataset_name in SPECTRE_DATASETS:
        return _load_spectre_split(data_root, dataset_name, split)
    elif dataset_name in MOLECULAR_DATASETS:
        return _load_molecular_split(data_root, dataset_name, split)
    else:
        # Fallback: try SPECTRE-style
        return _load_spectre_split(data_root, dataset_name, split)


def _load_spectre_split(data_root: str, dataset_name: str, split: str) -> list:
    """Load a SPECTRE-style processed split (single {split}.pt file)."""
    path = os.path.join(data_root, dataset_name, "processed", f"{split}.pt")
    data, slices = torch.load(path, weights_only=False)
    return _unpack_pyg(data, slices)


def _load_molecular_split(data_root: str, dataset_name: str, split: str) -> list:
    """Load a ConStruct molecular processed split.

    ConStruct molecular loaders write ``{split}_{suffix}.pt`` where suffix
    encodes dataset-specific flags (e.g. ``noh``/``h`` for QM9, ``f``/`` ``
    for GuacaMol, empty for MOSES).
    """
    suffix = _molecular_pt_suffix(dataset_name)
    filename = f"{split}_{suffix}.pt" if suffix else f"{split}_.pt"
    path = os.path.join(data_root, dataset_name, "processed", filename)
    data, slices = torch.load(path, weights_only=False)
    return _unpack_pyg(data, slices)


def _molecular_pt_suffix(dataset_name: str) -> str:
    """Return the processed-file suffix used by ConStruct for each molecular dataset.

    These mirror the defaults set in the dataset yaml configs:
    - qm9      : remove_h=True  → suffix "noh"
    - guacamol : filter=True    → suffix "f"
    - moses    : no flag        → suffix "" (file is ``{split}_.pt``)
    """
    if dataset_name == "qm9":
        return "noh"       # remove_h: True (default in qm9.yaml)
    elif dataset_name == "guacamol":
        return "f"         # filter: True (default in guacamol.yaml)
    elif dataset_name == "moses":
        return ""          # no suffix flag in MosesDataset
    return ""


def _unpack_pyg(data, slices) -> list:
    """Unpack a (data, slices) pair into a list of PyG Data objects."""
    graphs = []
    num_graphs = slices["x"].shape[0] - 1
    for i in range(num_graphs):
        g = torch_geometric.data.Data()
        for key in slices.keys():
            start, end = slices[key][i].item(), slices[key][i + 1].item()
            val = data[key]
            if key == "edge_index":
                g[key] = val[:, start:end]
            else:
                g[key] = val[start:end]
        graphs.append(g)
    return graphs


def pyg_to_nx(data) -> nx.Graph:
    """Convert a PyG Data object to an undirected NetworkX graph.

    Deduplicates edges (each undirected pair appears once).
    """
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0]))
    seen = set()
    for u, v in data.edge_index.t().tolist():
        key = (min(u, v), max(u, v))
        if key not in seen:
            G.add_edge(u, v)
            seen.add(key)
    return G
