import os
import os.path as osp
import pickle
from typing import Any, Sequence

from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from ConStruct import metrics
from ConStruct.utils import to_dense
from tqdm import tqdm


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    adj = torch.from_numpy(adj)
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()
    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    data = Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        charges=all_charges,
        smiles=smiles,
    )
    return data


def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    return Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        charges=data.charges[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def files_exist(files) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        degree_hist,
        charge_types=None,
        valencies=None,
    ):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.degree_hist = degree_hist
        self.charge_types = charge_types
        self.valencies = valencies


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class MuAttachTransform:
    """Per-item transform that attaches µ_G₀ from a pre-computed array.

    Loaded once at dataset init; attached to each Data object at __getitem__
    time so PyG's collation batches it automatically as data.mu (bs, mu_dim).
    Only active when use_tam=True — otherwise this transform is never created.

    PyG's InMemoryDataset.get(idx) calls transform(data) after slicing, so
    we use a side-channel `_current_idx` set by a patched __getitem__ in the
    dataset subclass. See SpectreGraphDataset for the wiring.
    """

    def __init__(self, mu_array: np.ndarray):
        self.mu = torch.from_numpy(mu_array).float()  # (N, mu_dim)
        self._current_idx = 0

    def set_idx(self, idx: int):
        self._current_idx = idx

    def __call__(self, data):
        data.mu = self.mu[self._current_idx].unsqueeze(0)  # (1, mu_dim) — PyG stacks → (bs, mu_dim)
        return data


def compute_reference_metrics(dataset_infos, datamodule, training_mode=False):
    ref_metrics_path = os.path.join(
        datamodule.train_dataloader().dataset.processed_dir, f"ref_metrics.pkl"
    )

    if not os.path.exists(ref_metrics_path):
        if training_mode:
            # During training, reference metrics are only used for ratio logging (wandb).
            # Skip expensive computation and use empty dicts — no disk write.
            print("Reference metrics not found. Skipping computation in training mode.")
            dataset_infos.val_reference_metrics = {}
            dataset_infos.test_reference_metrics = {}
            return
        print("Reference metrics not found. Computing them now.")
        # Transform the training dataset into a list of graphs in the appropriate format
        training_graphs = []
        print("Converting training dataset to placeholders.")
        for batch in tqdm(datamodule.train_dataloader()):
            training_graphs.append(
                to_dense(batch, dataset_infos).collapse(
                    dataset_infos.collapse_charges
                    if hasattr(dataset_infos, "collapse_charges")
                    else None
                )
            )

        print("Computing validation reference metrics.")
        val_sampling_metrics = metrics.sampling_metrics.SamplingMetrics(
            dataset_infos=dataset_infos,
            test=False,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader(),
        )
        val_reference_metrics = val_sampling_metrics.domain_metrics.forward(
            training_graphs, current_epoch=None, local_rank=0
        )
        print("Computing test reference metrics.")
        test_sampling_metrics = metrics.sampling_metrics.SamplingMetrics(
            dataset_infos=dataset_infos,
            test=False,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.test_dataloader(),  # datamodule.test_dataloader(),
        )
        test_reference_metrics = test_sampling_metrics.domain_metrics.forward(
            training_graphs, current_epoch=None, local_rank=0
        )
        print("Saving reference metrics.")
        # print(f"deg: {test_reference_metrics['degree']} | clus: {test_reference_metrics['clustering']} | orbit: {test_reference_metrics['orbit']}")
        # breakpoint()
        save_pickle((val_reference_metrics, test_reference_metrics), ref_metrics_path)

    print("Loading reference metrics.")
    (
        dataset_infos.val_reference_metrics,
        dataset_infos.test_reference_metrics,
    ) = load_pickle(ref_metrics_path)
    # print(dataset_infos.test_reference_metrics)
    # breakpoint()
