"""
Download and process graph datasets into TopoGen/data/.

SPECTRE / graph datasets:
    python dump_data.py dataset=planar
    python dump_data.py dataset=sbm
    python dump_data.py dataset=comm20
    python dump_data.py dataset=ego
    python dump_data.py dataset=enzymes   # requires data/enzymes/raw/ENZYMES.pkl

Molecular datasets:
    python dump_data.py dataset=qm9
    python dump_data.py dataset=guacamol
    python dump_data.py dataset=moses
"""

import os
import sys

import hydra
from omegaconf import DictConfig

# Make ConStruct importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines", "ConStruct"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SPECTRE_DATASETS = {"planar", "sbm", "comm20", "ego", "enzymes"}
MOLECULAR_DATASETS = {"qm9", "guacamol", "moses"}


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from ConStruct.datasets.dataset_utils import RemoveYTransform

    dataset_name = cfg.dataset.name
    root = os.path.join(PROJECT_ROOT, "data", dataset_name)

    print(f"Dataset : {dataset_name}")
    print(f"Root    : {root}")

    if dataset_name in SPECTRE_DATASETS:
        _dump_spectre(cfg, dataset_name, root)

    elif dataset_name == "qm9":
        _dump_qm9(cfg, root)

    elif dataset_name == "guacamol":
        _dump_guacamol(cfg, root)

    elif dataset_name == "moses":
        _dump_moses(cfg, root)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {sorted(SPECTRE_DATASETS | MOLECULAR_DATASETS)}"
        )

    print("Done. Files saved to:", root)


# ---------------------------------------------------------------------------
# SPECTRE / graph datasets
# ---------------------------------------------------------------------------

def _dump_spectre(cfg, dataset_name, root):
    from ConStruct.datasets.spectre_dataset import SpectreGraphDataset
    from ConStruct.datasets.dataset_utils import RemoveYTransform

    transform = RemoveYTransform()
    for split in ["train", "val", "test"]:
        ds = SpectreGraphDataset(
            dataset_name=dataset_name,
            split=split,
            root=root,
            fraction=cfg.dataset.fraction,
            transform=transform,
        )
        print(f"  {split:5s}: {len(ds)} graphs")


# ---------------------------------------------------------------------------
# Molecular datasets
# ---------------------------------------------------------------------------

def _dump_qm9(cfg, root):
    from ConStruct.datasets.qm9_dataset import QM9Dataset
    from ConStruct.datasets.dataset_utils import RemoveYTransform

    remove_h = cfg.dataset.remove_h
    for split in ["train", "val", "test"]:
        ds = QM9Dataset(
            split=split,
            root=root,
            remove_h=remove_h,
            transform=RemoveYTransform(),
        )
        print(f"  {split:5s}: {len(ds)} molecules")


def _dump_guacamol(cfg, root):
    from ConStruct.datasets.guacamol_dataset import GuacamolDataset

    filter_dataset = cfg.dataset.filter
    for split in ["train", "val", "test"]:
        ds = GuacamolDataset(
            split=split,
            root=root,
            filter_dataset=filter_dataset,
        )
        print(f"  {split:5s}: {len(ds)} molecules")


def _dump_moses(cfg, root):
    from ConStruct.datasets.moses_dataset import MosesDataset

    for split in ["train", "val", "test"]:
        ds = MosesDataset(
            split=split,
            root=root,
        )
        print(f"  {split:5s}: {len(ds)} molecules")


if __name__ == "__main__":
    main()
