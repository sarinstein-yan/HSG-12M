"""
Benchmark script for HSG-12M GNN baselines.

Usage
-----
```bash
python src/hsg/training.py  \\
    --root /path/to/HSG-12M \\
    --subset one-band --epochs 100 \\
    ...
```
Notes
-----
- New spatial baselines: `cgcnn`, `spline`, and `monet` require an explicit
  `edge_dim` derived from `data.edge_attr.size(-1)`.
- For `spline` and `monet`, we *automatically* apply
  `torch_geometric.transforms.Cartesian(cat=False)` to the dataset during
  their runs so that `edge_attr` becomes pseudo-coordinates and `edge_dim`
  matches that pseudo-dimension.
"""

__all__ = [
    "HSGLitDataModule",
    "LitGNN",
    "summarise_csv",
    "run_experiment",
]

import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision

from hsg.pyg import HSGInMemory
from hsg.sampler import rebalance_batch #StaticBatchSampler
from hsg.gnn_baselines import get_model_instance

# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class HSGLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        subset,
        seed: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        batch_size: int = 512,
        size_mode: str = "edge",  # "node" or "edge"
        max_num_per_batch: int = 2_000_000,
    ):
        super().__init__()
        self.root, self.subset = root, subset
        self.seed = int(seed)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.ds_train = self.ds_val = self.ds_test = None
        self.sz_train = self.sz_val = self.sz_test = None
        self.batch_size = int(batch_size)
        self.mode = size_mode
        assert self.mode in ["node", "edge"], "size_mode must be 'node' or 'edge'"
        self.max_num = int(max_num_per_batch)
        self.base_dataset = None

    def prepare_data(self):
        _ = HSGInMemory(self.root, self.subset,
                        transform=self.transform,
                        pre_transform=self.pre_transform,
                        pre_filter=self.pre_filter)

    def setup(self, stage=None):
        full = HSGInMemory(self.root, self.subset, transform=self.transform)
        self.base_dataset = full
        y_all = full.y.numpy()
        size_key = "n_nodes" if self.mode == "node" else "n_edges"
        if not hasattr(full, size_key):
            raise RuntimeError(f"Dataset missing size attribute: {size_key}")
        size_array = np.asarray(getattr(full, size_key), dtype=np.int64)
        if len(size_array) != len(full):
            raise RuntimeError("Size array length mismatch with dataset length.")

        # Stratified 80/20, then split 20% into 50/50 val/test using the *single* seed
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=self.seed)
        idx_train, idx_tmp = next(splitter.split(np.zeros_like(y_all), y_all))
        y_tmp = y_all[idx_tmp]

        counts = np.bincount(y_tmp)
        safe_min = counts[counts > 0].min() if (counts > 0).any() else 0
        if safe_min < 2:
            ss = ShuffleSplit(n_splits=1, train_size=0.5, random_state=self.seed)
            rel_val, rel_test = next(ss.split(idx_tmp))
        else:
            splitter_val = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=self.seed)
            rel_val, rel_test = next(splitter_val.split(np.zeros_like(y_tmp), y_tmp))
        idx_val, idx_test = idx_tmp[rel_val], idx_tmp[rel_test]

        idx_train = rebalance_batch(size_array[idx_train], self.batch_size, self.max_num)
        idx_val   = rebalance_batch(size_array[idx_val],   self.batch_size, self.max_num)
        idx_test  = rebalance_batch(size_array[idx_test],  self.batch_size, self.max_num)
        if idx_train is None or idx_val is None or idx_test is None:
            raise RuntimeError("Rebalancing failed. Try increasing `max_num_per_batch`.")

        self.ds_train, self.ds_val, self.ds_test = Subset(full, idx_train), Subset(full, idx_val), Subset(full, idx_test)
        self.sz_train, self.sz_val, self.sz_test = size_array[idx_train], size_array[idx_val], size_array[idx_test]

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class LitGNN(pl.LightningModule):
    def __init__(self, hparams, num_classes: int, in_dim: int):
        super().__init__()
        self.save_hyperparameters(dict(vars(hparams)))

        # --- Prepare extra kwargs for spatial baselines that need edge_dim ---
        extra_kwargs = {}
        needs_edge_dim = self.hparams.model in {"cgcnn", "spline", "monet", "ecc"}
        edge_dim = getattr(self.hparams, "edge_dim", None)
        if needs_edge_dim and edge_dim is not None:
            extra_kwargs["edge_dim"] = int(edge_dim)

        # Optional hyper-params for spline/monet (kept optional; rely on model defaults otherwise)
        if self.hparams.model in {"spline", "monet"}:
            ek = getattr(self.hparams, "edge_kernel", 0)
            if isinstance(ek, int) and ek > 0:
                extra_kwargs["kernel_size"] = ek
            if self.hparams.model == "spline":
                deg = getattr(self.hparams, "spline_degree", None)
                if isinstance(deg, int) and deg > 0:
                    extra_kwargs["degree"] = deg

        self.model = get_model_instance(
            self.hparams.model,
            dim_in=in_dim,
            dim_h_gnn=self.hparams.dim_gnn,
            dim_h_mlp=self.hparams.dim_mlp,
            dim_out=num_classes,
            num_layers_gnn=self.hparams.layers_gnn,
            num_layers_mlp=self.hparams.layers_mlp,
            dropout=self.hparams.dropout,
            num_heads=self.hparams.heads,
            **extra_kwargs,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.topks = [1, 2, 5, 10]

        self._tm = SimpleNamespace()
        self._tm.train_acc = Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self._tm.val_accs = torch.nn.ModuleList([
            Accuracy(top_k=k, task="multiclass", num_classes=num_classes) for k in self.topks
        ])
        self._tm.val_f1_macro = F1Score(average="macro", task="multiclass", num_classes=num_classes)
        self._tm.val_f1_micro = F1Score(average="micro", task="multiclass", num_classes=num_classes)
        self._tm.val_auc = AUROC(average="macro", task="multiclass", num_classes=num_classes)
        self._tm.val_ap = AveragePrecision(average="macro", task="multiclass", num_classes=num_classes)
        for m in [self._tm.train_acc, *self._tm.val_accs,
                  self._tm.val_f1_macro, self._tm.val_f1_micro,
                  self._tm.val_auc, self._tm.val_ap]:
            m.to("cpu") # Keep all metric state on CPU to avoid CUDA OOM with many classes
            m.sync_on_compute = True # Let torchmetrics do cross-process reduction at compute() time

        self._train_samples = 0
        self._epoch_start_time = 0.0

    # optimiser + scheduler (Warm Restarts)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr_init, amsgrad=True, weight_decay=0.0
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.t0, T_mult=self.hparams.t_mult, eta_min=self.hparams.lr_min
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    # forward + shared step
    def forward(self, data):
        return self.model(data)

    def _step(self, batch, stage: str):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        y = batch.y.view(-1)
        # Move inputs for metric updates to CPU (metrics live on CPU)
        logits_cpu = logits.detach().float().cpu()
        y_cpu = y.detach().cpu()

        if stage == "train":
            self._tm.train_acc.update(logits_cpu, y_cpu)
            self._train_samples += y.size(0)
        else:
            for acc in self._tm.val_accs:
                acc.update(logits_cpu, y_cpu)
            self._tm.val_f1_macro.update(logits_cpu, y_cpu)
            self._tm.val_f1_micro.update(logits_cpu, y_cpu)
            self._tm.val_auc.update(logits_cpu, y_cpu)
            self._tm.val_ap.update(logits_cpu, y_cpu)

        self.log(f"{stage}_loss", loss, 
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # training / val / test hooks
    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    # epoch boundary hooks 
    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()
        self._train_samples = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self):
        # ---- core metrics
        self.log("train_top1", self._tm.train_acc.compute(), sync_dist=False)
        # ^ torchmetrics already synced across ranks at compute()
        self._tm.train_acc.reset()
        # ---- wall time, throughput, GPU memory
        wall_time = time.perf_counter() - self._epoch_start_time
        throughput = self._train_samples / wall_time if wall_time > 0 else 0.0
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            mem_gb = 0.0

        self.log("train_wall_time_s", wall_time, sync_dist=True)
        self.log("train_throughput_samples_s", throughput, sync_dist=True)
        self.log("train_gpu_mem_gb", mem_gb, sync_dist=True)

    def on_validation_epoch_end(self):
        for k, acc in zip(self.topks, self._tm.val_accs):
            self.log(f"val_top{k}", acc.compute(), sync_dist=False); acc.reset()
        self.log("val_macro_f1",  self._tm.val_f1_macro.compute(), sync_dist=False)
        self.log("val_micro_f1",  self._tm.val_f1_micro.compute(), sync_dist=False)
        self.log("val_macro_auc", self._tm.val_auc.compute(), sync_dist=False)
        self.log("val_macro_ap",  self._tm.val_ap.compute(), sync_dist=False)
        self._tm.val_f1_macro.reset(); self._tm.val_f1_micro.reset()
        self._tm.val_auc.reset(); self._tm.val_ap.reset()

    def on_test_epoch_end(self):
        for k, acc in zip(self.topks, self._tm.val_accs):
            self.log(f"test_top{k}", acc.compute(), sync_dist=False)
        self.log("test_macro_f1",  self._tm.val_f1_macro.compute(), sync_dist=False)
        self.log("test_micro_f1",  self._tm.val_f1_micro.compute(), sync_dist=False)
        self.log("test_macro_auc", self._tm.val_auc.compute(), sync_dist=False)
        self.log("test_macro_ap",  self._tm.val_ap.compute(), sync_dist=False)


# ---------------------------------------------------------------------------
# Helper for summarising seeds → mean ± std CSV
# ---------------------------------------------------------------------------
def summarise_csv(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    rows = [{"metric": k, "mean": means[k], "std": stds[k]} for k in means.index]
    pd.DataFrame(rows).to_csv(csv_out, index=False)


# --- run experiment interactive wrapper --- #
def run_experiment(args: Namespace):
    """
    Train/validate/test all requested GNN architectures using static
    size-capped batches (nodes or edges), avoiding CUDA OOM while keeping a
    fixed number of steps/epoch.
    """
    # ---------- house-keeping ----------
    save_path = Path(args.save_dir) / args.subset
    save_path.mkdir(parents=True, exist_ok=True)

    if args.models == ["all"]:
        args.models = ["gcn", "sage", "gat", "gatv2", "gin", "gine",
                       "mf", "cgcnn", "ecc", "spline", "monet"]

    print(f"⏳  Loading PolyGraph from {args.root}, subset={args.subset}, "
          f"batch_cap={int(args.max_num)} {args.mode}s, "
          f"models={args.models}, seeds={args.seeds}")

    # ---------- data ----------
    dm = HSGLitDataModule(
        root=args.root,
        subset=args.subset,
        seeds=args.seeds,
        max_num=int(args.max_num),                # cap by total nodes/edges
        mode=str(getattr(args, "mode", "edge")),  # 'node' or 'edge'
        skip_too_big=bool(getattr(args, "skip_too_big", True)),
        drop_last=bool(getattr(args, "drop_last", False)),
        pack_strategy=str(getattr(args, "pack_strategy", "sorted_desc")),
        shuffle_batch_order=bool(getattr(args, "shuffle_batch_order", False)),
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", True)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
    )
    dm.prepare_data(); dm.setup()

    # Underlying base dataset (shared across seeds):
    base_dataset = dm.datasets[0][0].dataset  # Subset(...).dataset

    # ---------- loop over architectures ----------
    for model_name in args.models:
        print(f"\n🧠  ▶ Training {model_name} …")
        args.model = model_name
        summaries = []

        # (Optional) add Cartesian pseudo-coordinates for specific models:
        if model_name in {"spline", "monet"}:
            base_dataset.transform = T.Cartesian(cat=False)
            print("   • Applied transforms.Cartesian(cat=False) for pseudo-coordinates.")

        # Detect edge_dim once (same across splits):
        probe = dm.datasets[0][0][0]
        args.edge_dim = (int(probe.edge_attr.size(-1))
                         if getattr(probe, "edge_attr", None) is not None else None)
        if model_name in {"cgcnn", "ecc", "spline", "monet"}:
            print(f"   • Detected edge_dim={args.edge_dim} for {model_name}.")

        for seed_idx, seed in enumerate(args.seeds):
            pl.seed_everything(seed, workers=True)

            train_loader = dm.train_dataloader(seed_idx)
            val_loader   = dm.val_dataloader(seed_idx)
            test_loader  = dm.test_dataloader(seed_idx)

            # Log steps/epoch (now fixed thanks to static sampler):
            try:
                n_steps = len(train_loader)
                print(f"   • Seed {seed}: steps/epoch={n_steps} "
                      f"(cap {int(args.max_num)} {args.mode}s)")
            except Exception:
                pass

            num_classes = train_loader.dataset.dataset.num_classes
            in_dim      = train_loader.dataset.dataset.num_node_features
            print(f"   • Seed {seed}: {num_classes} classes, {in_dim} input feats")

            lit = LitGNN(args, num_classes, in_dim)

            logger = TensorBoardLogger(
                save_path / "tb_logs" / model_name,
                name=f"{model_name}_seed{seed}"
            )
            ckpt_dir = save_path / model_name / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            ckpt_top1 = ModelCheckpoint(
                monitor="val_top1", mode="max", dirpath=ckpt_dir,
                filename="best-top1-{epoch:03d}-{val_top1:.4f}"
            )
            ckpt_f1 = ModelCheckpoint(
                monitor="val_macro_f1", mode="max", dirpath=ckpt_dir,
                filename="best-f1-{epoch:03d}-{val_macro_f1:.4f}"
            )
            stopper = EarlyStopping(
                monitor="val_macro_f1", patience=args.early_stop_patience, mode="max"
            )

            trainer = pl.Trainer(
                accelerator="gpu",
                # devices=2,
                # strategy=...,
                # use_distributed_sampler=False,
                max_epochs=args.epochs,
                logger=logger,
                log_every_n_steps=args.log_every_n_steps,
                callbacks=[ckpt_top1, ckpt_f1, stopper],
                deterministic=True,
                fast_dev_run=getattr(args, "fast_dev_run", False)
            )

            trainer.fit(lit, train_loader, val_loader)
            trainer.test(ckpt_path=ckpt_f1.best_model_path, dataloaders=test_loader)

            # ---------- gather metrics ----------
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
            summaries.append({"seed": seed, **metrics})

        # Reset transform so it doesn't leak:
        base_dataset.transform = None

        # ---------- write per-seed + summary CSV ----------
        df_path = save_path / f"{model_name}.csv"
        pd.DataFrame(summaries).to_csv(df_path, index=False)
        summarise_csv(df_path, df_path.with_name(df_path.stem + "_summary.csv"))
        print(f"✅  Finished {model_name}; results → {df_path}")




if __name__ == "__main__":

    def main():
        parser = ArgumentParser()
        parser.add_argument("--root", type=Path)
        parser.add_argument("--save_dir", type=Path)
        parser.add_argument("--subset", type=str, default="one-band")
        parser.add_argument("--models", nargs="*", default=["all"],
                            help="Architectures to run; omit for all.")
        parser.add_argument("--epochs", type=int, default=40)
        parser.add_argument("--mode", type=str, default="edge")
        parser.add_argument("--max_num", type=int, default=1e5)
        parser.add_argument("--skip_too_big", type=bool, default=True)
        parser.add_argument("--seeds", nargs="*", type=int, default=[42, 624, 706])
        ### GNN params
        parser.add_argument("--dim_gnn", type=int, default=128)
        parser.add_argument("--dim_mlp", type=int, default=128)
        parser.add_argument("--layers_gnn", type=int, default=4)
        parser.add_argument("--layers_mlp", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.)
        # GAT hyper-params
        parser.add_argument("--heads", type=int, default=1)
        # spatial conv hyper-params; used only for spline/monet if provided
        parser.add_argument("--edge_kernel", type=int, default=0,
                            help="Kernel size for spline/monet. If 0, use model defaults (5 for spline, 25 for MoNet)." )
        parser.add_argument("--spline_degree", type=int, default=1,
                            help="Spline polynomial degree. Only used by --models spline when >0.")
        ### Optimizer params
        parser.add_argument("--lr_init", type=float, default=1e-3, help="Initial learning rate")
        parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
        parser.add_argument("--t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts")
        parser.add_argument("--t_mult", type=int, default=4, help="T_mult for CosineAnnealingWarmRestarts")
        parser.add_argument("--log_every_n_steps", type=int, default=5)
        parser.add_argument("--early_stop_patience", type=int, default=10)
        parser.add_argument("--fast_dev_run", type=bool, default=False)
        args = parser.parse_args()
        run_experiment(args)
    
    main()