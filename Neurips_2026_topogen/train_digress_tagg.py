"""
Train or evaluate a DiGress-TAGG model (PDM loss + TAM attention bias, marginal transition).

Same backbone as train_digress.py but with use_pdm=True and use_tam=True enabled
via the digress_tagg_experiment configs. Outputs go to runs/digress_tagg/.

Usage
-----
Train:
    python train_digress_tagg.py +digress_tagg_experiment=planar

Evaluate only (runs 5 seeds):
    python train_digress_tagg.py +digress_tagg_experiment=planar \
        general.test_only=/abs/path/to/checkpoint.ckpt
"""

from __future__ import annotations

import json
import re
import os
import sys
import pathlib
import time
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# ── Make ConStruct importable ────────────────────────────────────────────────
_TOPOGEN_ROOT = pathlib.Path(__file__).parent.resolve()
_CONSTRUCT_ROOT = _TOPOGEN_ROOT / "baselines" / "ConStruct"
_CONSTRUCT_SRC  = _CONSTRUCT_ROOT / "ConStruct"

for p in [str(_CONSTRUCT_ROOT), str(_CONSTRUCT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import graph_tool as gt  # noqa: F401
except ModuleNotFoundError:
    print("[train_digress_tagg] graph_tool not found — planarity/isomorphism metrics unavailable")

warnings.filterwarnings("ignore", category=PossibleUserWarning)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_datadir(cfg: DictConfig) -> str:
    datadir = cfg.dataset.datadir
    if os.path.isabs(datadir):
        return datadir
    return str(_TOPOGEN_ROOT / datadir)


def _build_datamodule(cfg: DictConfig, root_path: str):
    name = cfg.dataset.name

    from omegaconf import OmegaConf
    with open_dict_safe(cfg) as c:
        c.dataset.datadir = root_path

    if name in {"sbm", "comm20", "planar", "ego", "grid", "enzymes", "lobster", "tree"}:
        from ConStruct.datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        datamodule    = SpectreGraphDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule)

    elif name == "qm9":
        from datasets.qm9_dataset import QM9DataModule, QM9Infos
        datamodule    = QM9DataModule(cfg)
        dataset_infos = QM9Infos(datamodule, cfg)

    elif name == "guacamol":
        from datasets.guacamol_dataset import GuacamolDataModule, GuacamolInfos
        datamodule    = GuacamolDataModule(cfg)
        dataset_infos = GuacamolInfos(datamodule, cfg)

    elif name == "moses":
        from datasets.moses_dataset import MosesDataModule, MosesInfos
        datamodule    = MosesDataModule(cfg)
        dataset_infos = MosesInfos(datamodule, cfg)

    else:
        raise NotImplementedError(f"Unknown dataset '{name}'")

    return datamodule, dataset_infos


def _dump_metrics(metrics: dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) or hasattr(v, "item")}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


def _placeholder_batch_to_nx_list(batch):
    """Convert one PlaceHolder batch (X, E, node_mask) to a list of nx.Graph objects.
    No torch / no PyG deps in the result — pure networkx."""
    import networkx as nx
    import numpy as np
    X = batch.X.detach().cpu().numpy()
    E = batch.E.detach().cpu().numpy()
    mask = batch.node_mask.detach().cpu().numpy()
    if E.ndim == 4:
        adj = (E.argmax(-1) > 0).astype(np.int8)
    else:
        adj = (E > 0).astype(np.int8)
    graphs = []
    for b in range(X.shape[0]):
        n = int(mask[b].sum())
        a = adj[b][:n, :n]
        a = ((a + a.T) > 0).astype(np.int8)
        np.fill_diagonal(a, 0)
        graphs.append(nx.from_numpy_array(a))
    return graphs


def _dump_generated_graphs(samples_pkl_path: str, out_path: str):
    import pickle as _pkl
    if not os.path.exists(samples_pkl_path):
        return None
    with open(samples_pkl_path, "rb") as f:
        batches = _pkl.load(f)
    graphs = []
    for batch in batches:
        graphs.extend(_placeholder_batch_to_nx_list(batch))
    with open(out_path, "wb") as f:
        _pkl.dump(graphs, f, protocol=_pkl.HIGHEST_PROTOCOL)
    return graphs


def _dump_reference_graphs(datamodule, split: str, out_path: str):
    import pickle as _pkl
    from torch_geometric.utils import to_networkx
    loader_fn = {"val": datamodule.val_dataloader, "test": datamodule.test_dataloader}[split]
    graphs = []
    for batch in loader_fn():
        for data in batch.to_data_list():
            graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None,
                                      to_undirected=True, remove_self_loops=True))
    with open(out_path, "wb") as f:
        _pkl.dump(graphs, f, protocol=_pkl.HIGHEST_PROTOCOL)
    return graphs


class open_dict_safe:
    """Context manager that temporarily allows OmegaConf struct mutation."""
    def __init__(self, cfg):
        self.cfg = cfg
    def __enter__(self):
        from omegaconf import OmegaConf
        OmegaConf.set_struct(self.cfg, False)
        return self.cfg
    def __exit__(self, *_):
        from omegaconf import OmegaConf
        OmegaConf.set_struct(self.cfg, True)


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="configs", config_name="config_digress_tagg")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)

    root_path = _resolve_datadir(cfg)
    print(f"[train_digress_tagg] dataset   : {cfg.dataset.name}")
    print(f"[train_digress_tagg] data root : {root_path}")
    print(f"[train_digress_tagg] run name  : {cfg.general.name}")
    print(f"[train_digress_tagg] transition: {cfg.model.transition}")

    from ConStruct.metrics.sampling_metrics import SamplingMetrics

    datamodule, dataset_infos = _build_datamodule(cfg, root_path)

    val_sampling_metrics = SamplingMetrics(
        dataset_infos,
        test=False,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )
    test_sampling_metrics = SamplingMetrics(
        dataset_infos,
        test=True,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.test_dataloader(),
    )

    from diffusion_model_discrete import DiscreteDenoisingDiffusion

    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        val_sampling_metrics=val_sampling_metrics,
        test_sampling_metrics=test_sampling_metrics,
    )

    params_to_ignore = [
        "module.model.dataset_infos",
        "module.model.val_sampling_metrics",
        "module.model.test_sampling_metrics",
    ]
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
        model, params_to_ignore
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            monitor="val/epoch_NLL",
            save_top_k=5,
            mode="min",
            every_n_epochs=1,
        )
        last_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="last",
            every_n_epochs=1,
        )
        callbacks += [last_ckpt, checkpoint_callback]

    # ── Trainer ────────────────────────────────────────────────────────────
    is_debug = cfg.general.name == "debug"
    use_gpu  = torch.cuda.is_available() and not is_debug

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp",
        accelerator="gpu" if use_gpu else "cpu",
        devices=-1 if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=is_debug,
        enable_progress_bar=cfg.train.get("progress_bar", False),
        callbacks=callbacks,
        log_every_n_steps=1 if is_debug else cfg.general.log_every_steps,
        logger=[],
    )

    # ── Run ────────────────────────────────────────────────────────────────
    if not cfg.general.test_only:
        fit_start = time.time()
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        fit_time_s = time.time() - fit_start
        print(f"[train_digress_tagg] training time: {fit_time_s:.1f}s ({fit_time_s/3600:.2f}h)")
        if cfg.train.save_model:
            print(f"[train_digress_tagg] best ckpt: {checkpoint_callback.best_model_path}")
        _dump_metrics({"train/total_time_s": fit_time_s}, "train_timing.json")
    else:
        # ── Step 0: pick top-3 NLL ckpts via best_k_models embedded in any ckpt ─
        ckpt_dir = pathlib.Path(cfg.general.test_only).parent

        def _epoch_of(p):
            import re as _re
            m = _re.search(r"epoch=(\d+)", pathlib.Path(p).name)
            return int(m.group(1)) if m else -1

        epoch_ckpts = [p for p in ckpt_dir.glob("epoch=*.ckpt") if "-v" not in p.name]
        if not epoch_ckpts:
            raise RuntimeError(f"No epoch=*.ckpt files (without -v suffix) in {ckpt_dir}")
        latest = max(epoch_ckpts, key=lambda p: _epoch_of(p))
        state = torch.load(str(latest), map_location="cpu", weights_only=False)
        bkm = next(
            v for k, v in state.get("callbacks", {}).items()
            if "val/epoch_NLL" in k
        )["best_k_models"]
        nll_pairs = []
        for orig_path, nll in bkm.items():
            name = pathlib.Path(orig_path).name
            local = ckpt_dir / name
            if not local.exists():
                # Strip "-v<digit>" suffix (Lightning adds these on training-run resume).
                # The bare-epoch ckpt has identical content; try it as a fallback.
                stripped = re.sub(r"-v\d+(?=\.ckpt$)", "", name)
                alt = ckpt_dir / stripped
                if alt.exists():
                    local = alt
            if local.exists():
                nll_pairs.append((local, float(nll)))
        nll_pairs.sort(key=lambda x: x[1])
        top3 = nll_pairs[:3]
        print(f"[train_digress_tagg] best_k_models has {len(nll_pairs)} resolvable ckpts; selecting top-3 by NLL:")
        for p, nll in top3:
            print(f"  NLL={nll:.4f}  {p.name}")

        n_val  = len(datamodule.val_dataloader().dataset)
        n_test = len(datamodule.test_dataloader().dataset)
        with open_dict_safe(cfg) as c:
            c.general.sample_every_val = 1
            c.general.samples_to_generate = n_val
            c.general.final_model_samples_to_generate = n_test
        print(f"[train_digress_tagg] samples_to_generate: val={n_val}  test={n_test}")

        MMD_KEYS = ["degree", "clustering", "orbit", "spectre", "wavelet"]
        scan_results = {}
        best_ckpt_path = None
        best_score = float("inf")

        for ckpt_file, nll in top3:
            print(f"[train_digress_tagg] val-scanning: {ckpt_file.name}  (NLL={nll:.4f})")
            pl.seed_everything(cfg.train.seed)
            model._current_save_dir = os.path.join(os.getcwd(), "val_scan", ckpt_file.stem)
            trainer.validate(model, datamodule=datamodule, ckpt_path=str(ckpt_file))
            metrics = dict(trainer.logged_metrics)
            present = [k for k in MMD_KEYS if k in metrics]
            mmd_avg = (sum(float(metrics[k]) for k in present) / len(present)
                       if present else float("inf"))
            scan_results[str(ckpt_file)] = {
                "nll": nll, "mmd_avg": mmd_avg,
                **{k: float(metrics[k]) for k in present},
            }
            _dump_metrics(metrics, f"val_scan/{ckpt_file.stem}_metrics.json")
            print(f"[train_digress_tagg]   mmd_avg = {mmd_avg:.4f}")
            if mmd_avg < best_score:
                best_score = mmd_avg
                best_ckpt_path = str(ckpt_file)

        _dump_metrics(
            {f"top3/{pathlib.Path(p).name}/{kk}": vv
             for p, d in scan_results.items() for kk, vv in d.items()},
            "val_scan/top3_summary.json",
        )

        print(f"\n[train_digress_tagg] top-3 ranking by mean MMD (lower = better):")
        for p, r in sorted(scan_results.items(), key=lambda x: x[1]["mmd_avg"]):
            marker = " <- best" if p == best_ckpt_path else ""
            print(f"  mmd_avg={r['mmd_avg']:.4f}  NLL={r['nll']:.4f}  {pathlib.Path(p).name}{marker}")

        print(f"\n[train_digress_tagg] running 5-seed test eval on: {best_ckpt_path}")
        seed_metric_paths = []
        seed_graph_lists = {}
        for seed_idx in range(5):
            new_seed = seed_idx * 1000
            pl.seed_everything(new_seed)
            with open_dict_safe(cfg) as c:
                c.train.seed = new_seed
            print(f"[train_digress_tagg] seed {new_seed}")
            seed_dir = os.path.join(os.getcwd(), "test", f"seed={new_seed}")
            model._current_save_dir = seed_dir
            trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)
            seed_path = f"test/seed={new_seed}_metrics.json"
            _dump_metrics(dict(trainer.logged_metrics), seed_path)
            seed_metric_paths.append(seed_path)
            seed_graphs = _dump_generated_graphs(
                os.path.join(seed_dir, "generated_samples_rank0.pkl"),
                os.path.join(seed_dir, "generated_graphs_nx.pkl"),
            )
            if seed_graphs is not None:
                seed_graph_lists[new_seed] = seed_graphs

        if seed_graph_lists:
            import pickle as _pkl
            with open("test/test_graphs.pkl", "wb") as f:
                _pkl.dump(seed_graph_lists, f, protocol=_pkl.HIGHEST_PROTOCOL)
            print(f"[train_digress_tagg] dumped test/test_graphs.pkl: {len(seed_graph_lists)} seeds")
        try:
            ref_graphs = _dump_reference_graphs(datamodule, "test", "test/reference_graphs.pkl")
            print(f"[train_digress_tagg] dumped test/reference_graphs.pkl: {len(ref_graphs)} reference graphs")
        except Exception as e:
            print(f"[train_digress_tagg] WARNING: could not dump reference graphs: {e}")

        import statistics, math as _math
        per_seed = []
        for p in seed_metric_paths:
            try:
                with open(p) as f: per_seed.append(json.load(f))
            except Exception as e:
                print(f"[train_digress_tagg] WARNING: could not read {p}: {e}")
        if per_seed:
            all_keys = sorted({k for d in per_seed for k in d})
            agg = {}
            for k in all_keys:
                vals = [d[k] for d in per_seed
                        if k in d and isinstance(d[k], (int, float))
                        and not (isinstance(d[k], float) and _math.isnan(d[k]))]
                if not vals: continue
                agg[f"{k}/mean"] = statistics.mean(vals)
                agg[f"{k}/std"]  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                agg[f"{k}/n"]    = len(vals)
            agg["best_ckpt"] = best_ckpt_path
            agg["n_test_graphs"] = n_test
            with open("test/seed_aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            tg_mean = agg.get("test/per_graph_generation_time_s/mean", float("nan"))
            tg_std  = agg.get("test/per_graph_generation_time_s/std",  float("nan"))
            print(f"\n[train_digress_tagg] 5-seed aggregate written to test/seed_aggregate.json")
            print(f"[train_digress_tagg] per-graph generation time: mean={tg_mean:.3f}s std={tg_std:.3f}s")


if __name__ == "__main__":
    main()
