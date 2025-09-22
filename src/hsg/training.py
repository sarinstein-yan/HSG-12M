import os
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

# Assuming these are in your project structure
from hsg.pyg import HSGInMemory
from hsg.sampler import rebalance_batch
from hsg.gnn_baselines import get_model_instance
from hsg.callback import DDPMonitorCallback

# =================================================================================
# 1. Lightning DataModule
# =================================================================================
class HSGLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        subset: str,
        seed: int | None = None,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        batch_size: int = 512,
        size_mode: str = "edge",  # "node" or "edge"
        max_num_per_batch: int = 2_000_000,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.base_dataset = None
        self.idx_train = self.idx_val = self.idx_test = None
        self.ds_train = self.ds_val = self.ds_test = None

    def prepare_data(self) -> None:
        _ = HSGInMemory(
            self.hparams.root, self.hparams.subset,
            transform=self.hparams.transform,
            pre_transform=self.hparams.pre_transform,
            pre_filter=self.hparams.pre_filter,
        )

    def setup(self, stage: str | None = None) -> None:
        full = HSGInMemory(self.hparams.root, self.hparams.subset, 
                           transform=self.hparams.transform)
        self.base_dataset = full

        y = self._to_np(full.y)
        size_key = "n_nodes" if self.hparams.size_mode == "node" else "n_edges"
        if not hasattr(full, size_key):
            raise RuntimeError(f"Dataset missing size attribute: {size_key}")
        sizes = np.asarray(getattr(full, size_key), dtype=np.int64)
        if sizes.shape[0] != len(full):
            raise RuntimeError("Size array length mismatch with dataset length.")

        idx_tr, idx_va, idx_te = self._split_indices(y, self.hparams.seed)
        idx_tr = self._rebalance(idx_tr, sizes, "train")
        idx_va = self._rebalance(idx_va, sizes, "val")
        idx_te = self._rebalance(idx_te, sizes, "test")

        self.idx_train, self.idx_val, self.idx_test = idx_tr, idx_va, idx_te
        self.ds_train, self.ds_val, self.ds_test = Subset(full, idx_tr), Subset(full, idx_va), Subset(full, idx_te)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.ds_train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.ds_val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.ds_test)

    @property
    def edge_dim(self) -> int:
        if self.base_dataset is None:
            raise RuntimeError("Base dataset not initialized. Call setup() first.")
        return self.base_dataset.num_edge_features

    @staticmethod
    def _to_np(x) -> np.ndarray:
        try:
            return x.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(x)

    def _split_indices(self, y: np.ndarray, seed: int):
        n = y.shape[0]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=seed)
        idx_tr, idx_tmp = next(sss.split(np.zeros(n, dtype=np.int8), y))

        y_tmp = y[idx_tmp]
        counts = np.bincount(y_tmp)
        can_stratify = (counts > 0).any() and counts[counts > 0].min() >= 2

        if can_stratify:
            sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=seed)
            rel_va, rel_te = next(sss2.split(np.zeros_like(y_tmp), y_tmp))
        else:
            ss = ShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
            rel_va, rel_te = next(ss.split(idx_tmp))

        return idx_tr, idx_tmp[rel_va], idx_tmp[rel_te]

    def _rebalance(self, idx: np.ndarray, sizes: np.ndarray, name: str) -> np.ndarray:
        # Assuming `rebalance_batch` is a valid, imported function
        out = rebalance_batch(
            sample_sizes=sizes[idx],
            batch_size=self.hparams.batch_size,
            max_num_per_batch=self.hparams.max_num_per_batch
        )
        if out is None:
            raise RuntimeError(
                f"Rebalancing failed for '{name}'. Increase `max_num_per_batch` (current: {self.hparams.max_num_per_batch})."
            )
        return idx[np.asarray(out, dtype=np.int64)]

    def _loader(self, subset: Subset) -> DataLoader:
        return DataLoader(subset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=True)

# =================================================================================
# 2. Lightning Module
# =================================================================================
class LightningGNN(pl.LightningModule):
    def __init__(
        self, 
        model_name: str,
        dim_in: int,
        dim_h_gnn: int,
        dim_h_mlp: int,
        dim_out: int,
        num_layers_gnn: int,
        num_layers_mlp: int,
        dropout: float = 0.0,
        lr_init: float = 1e-3,
        lr_min: float = 1e-5,
        weight_decay: float = 0.0,
        T_0: int = 100,
        *,
        edge_dim: int | None = None,
        num_heads: int = 1,  # only for GAT | GATv2
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model_instance(
            model_name=model_name,
            dim_in=dim_in, dim_h_gnn=dim_h_gnn, 
            dim_h_mlp=dim_h_mlp, dim_out=dim_out, 
            num_layers_gnn=num_layers_gnn, 
            num_layers_mlp=num_layers_mlp,
            dropout=dropout, edge_dim=edge_dim,
            num_heads=num_heads, **model_kwargs,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=dim_out),
            'f1': MulticlassF1Score(num_classes=dim_out, average='macro'),
        }, prefix="train_")
        self.val_metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=dim_out),
            'acc_top5': MulticlassAccuracy(num_classes=dim_out, top_k=5),
            'acc_top10': MulticlassAccuracy(num_classes=dim_out, top_k=10),
            'f1': MulticlassF1Score(num_classes=dim_out, average='macro'),
        }, prefix="val_")
        self.test_metrics = self.val_metrics.clone(prefix="test_")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr_init, amsgrad=True, weight_decay=self.hparams.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.T_0, eta_min=self.hparams.lr_min
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    def forward(self, data):
        return self.model(data)

    def _step(self, batch):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        y = batch.y.view(-1).long()
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_metrics.update(logits, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        self.val_metrics.update(logits, y)
        if not self.trainer.sanity_checking:
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.update(logits, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

# =================================================================================
# 3. Configuration Dataclass
# =================================================================================
@dataclass
class Config:
    # Environment
    data_root: str
    save_dir: str
    subset: str
    seed: int | None = None
    
    # Training
    model_name: str = "gcn"
    batch_size: int = 4096
    max_epochs: int = 100
    max_steps: int = -1
    
    # DataModule
    size_mode: str = "edge"
    max_num_per_batch: int = 2_000_000
    transform: Any | None = None # "cartesian" for spline/monet, else None

    # Model Hyperparameters
    dim_h_gnn: int = 64
    dim_h_mlp: int = 64
    num_layers_gnn: int = 4
    num_layers_mlp: int = 2
    dropout: float = 0.0
    num_heads: int = 1  # For GAT/GATv2
    kernel_size: int = 5  # For MoNet/SplineCNN

    # Optimizer
    lr_init: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 0.0
    T_0: int = 100

    # Trainer
    devices: int | str = "auto"
    strategy: str = "auto"
    log_every_n_steps: int = 50
    profiler: str | None = None
    fast_dev_run: bool = False
    num_sanity_val_steps: int = 0
    deterministic: bool = True

# =================================================================================
# 4. Experiment Runner Wrapper
# =================================================================================
def run_experiment(cfg: Config) -> Dict[str, float]:
    """
    Sets up and runs a single training and testing experiment.
    """
    print(f"--- Running Experiment: {cfg.model_name} | Subset: {cfg.subset} | Seed: {cfg.seed} ---")
    pl.seed_everything(cfg.seed, workers=True)

    # 1. Setup DataModule
    if cfg.model_name in ["spline", "monet"]:
        cfg.transform = T.Cartesian(cat=False)
    
    dm = HSGLightningDataModule(
        root=cfg.data_root, subset=cfg.subset, transform=cfg.transform, seed=cfg.seed,
        batch_size=cfg.batch_size, size_mode=cfg.size_mode, max_num_per_batch=cfg.max_num_per_batch
    )
    dm.prepare_data(); dm.setup()

    # 2. Setup Model
    gnn = LightningGNN(
        model_name=cfg.model_name,
        dim_in=dm.base_dataset.num_node_features,
        dim_out=dm.base_dataset.num_classes,
        dim_h_gnn=cfg.dim_h_gnn,
        dim_h_mlp=cfg.dim_h_mlp,
        num_layers_gnn=cfg.num_layers_gnn,
        num_layers_mlp=cfg.num_layers_mlp,
        dropout=cfg.dropout,
        lr_init=cfg.lr_init,
        lr_min=cfg.lr_min,
        weight_decay=cfg.weight_decay,
        T_0=cfg.T_0,
        edge_dim=dm.edge_dim,
        num_heads=cfg.num_heads,
        kernel_size=cfg.kernel_size,
    )

    # 3. Setup Logging & Callbacks
    save_path = Path(cfg.save_dir) / cfg.subset
    save_path.mkdir(parents=True, exist_ok=True)
    
    version_str = f"{cfg.model_name}-bs{cfg.batch_size}-ep{cfg.epochs}-seed{cfg.seed}-{int(time.time())}"
    loggers = [
        TensorBoardLogger(save_path / "tb_logs", name=cfg.model_name, version=version_str),
        CSVLogger(save_path / "csv_logs", name=cfg.model_name, version=version_str)
    ]
    
    ckpt_f1 = ModelCheckpoint(
        dirpath=save_path / "ckpts" / version_str, monitor="val_f1", mode="max",
        filename="best-{val_f1:.4f}-{epoch:03d}"
    )
    train_stats = DDPMonitorCallback()

    # 4. Setup Trainer & Run
    trainer = pl.Trainer(
        devices=cfg.devices,
        strategy=cfg.strategy,
        callbacks=[ckpt_f1, train_stats],
        logger=loggers,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        profiler=cfg.profiler,
        fast_dev_run=cfg.fast_dev_run,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        deterministic=cfg.deterministic,
    )

    trainer.fit(gnn, datamodule=dm)
    test_results = trainer.test(ckpt_path=ckpt_f1.best_model_path, datamodule=dm)

    # Clean up on other ranks and exit the function
    if not trainer.is_global_zero:
        del trainer, gnn, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    
    # 5. Collect and return results
    final_results = test_results[0] # Results are in a list
    final_results["avg_throughput"] = train_stats.avg_throughput
    final_results["peak_gpu_mem_sum"] = train_stats.peak_gpu_mem_sum
    final_results["peak_gpu_mem_per_graph"] = train_stats.peak_gpu_mem_per_graph

    # Clean up to free memory
    del trainer, gnn, dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_results


# =================================================================================
# 6. Main Sweeping Logic
# =================================================================================
if __name__ == "__main__":
    import torch.distributed as dist
    def is_rank_zero():
        """Checks if the current process is rank 0 or if DDP is not in use."""
        return dist.get_rank() == 0 if dist.is_available() and dist.is_initialized() else True
    
    torch.set_float32_matmul_precision("medium")

    # --- Sweep Configuration ---
    DATA_ROOT = ...
    SAVE_DIR = ...

    SUBSETS = ["one-band", "two-band", "three-band", "topology", "all"]
    MODEL_NAMES = ["mf", "gcn", "sage", "gat", "gin", "cgcnn", "monet"]
    SEEDS = [42, 2025, 666]
    MAX_EPOCHS = 100
    MAX_STEPS = 5000
    BATCH_SIZE = 3500

    # Model dimensions are tuned per subset
    DIM_H_GNN = {
        "one-band":   dict(zip(MODEL_NAMES, [100, 467, 330, 452, 312, 202, 194])),
        "two-band":   dict(zip(MODEL_NAMES, [200, 933, 661, 933, 621, 410, 402])),
        "three-band": dict(zip(MODEL_NAMES, [300, 1279, 963, 1279, 852, 601, 548])),
        "topology":   dict(zip(MODEL_NAMES, [300, 1279, 963, 1279, 852, 601, 548])),
        "all":        dict(zip(MODEL_NAMES, [300, 1279, 963, 1279, 852, 601, 548])),
    }
    DIM_H_MLP = {
        "one-band": 128, "two-band": 256, "three-band": 1500,
        "topology": 1500, "all": 1500
    }

    # Define path for incremental results and ensure directory exists
    results_csv_path = Path(SAVE_DIR) / "sweep_results.csv"
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📝 Sweep results will be saved to: {results_csv_path}")

    # --- Start Sweeping ---
    # for subset in SUBSETS:
    for subset in SUBSETS:
        for model_name in MODEL_NAMES:
            for seed in SEEDS:
                # Create config for the current run
                cfg = Config(
                    data_root=DATA_ROOT,
                    save_dir=SAVE_DIR,
                    subset=subset,
                    seed=seed,
                    model_name=model_name,
                    max_epochs=MAX_EPOCHS,
                    max_steps=MAX_STEPS,
                    batch_size=BATCH_SIZE,
                    dim_h_gnn=DIM_H_GNN[subset][model_name],
                    dim_h_mlp=DIM_H_MLP[subset],
                )

                try:
                    results = run_experiment(cfg)
                    # Add config details for easy grouping later
                    if is_rank_zero():
                        results['subset'] = subset
                        results['model_name'] = model_name
                        results['seed'] = seed

                        # --- Flush result to disk immediately ---
                        current_result_df = pd.DataFrame([results])
                        # Append to CSV, write header only if file doesn't exist
                        current_result_df.to_csv(
                            results_csv_path,
                            mode='a',
                            header=not results_csv_path.exists(),
                            index=False
                        )

                        # --- Live Preview ---
                        print(f"✅ Result saved. Preview of results file:")
                        live_preview_df = pd.read_csv(results_csv_path)
                        print(live_preview_df.tail())
                        print("-" * 50)
                
                except Exception as e:
                    if is_rank_zero():
                        print(f"‼️ ERROR running {model_name} on {subset} with seed {seed}: {e}")
                        with open(Path(SAVE_DIR) / "error_log.txt", "a") as f:
                            f.write(f"[{time.ctime()}] ERROR on {model_name}/{subset}/seed{seed}: {e}\n")
                    continue

    # --- Aggregate and Summarize Final Results ---
    if is_rank_zero():
        if not results_csv_path.exists():
            print("No experiments were successfully completed. No summary to generate.")
        else:
            results_df = pd.read_csv(results_csv_path)

            # Identify metric columns to aggregate
            metric_cols = [col for col in results_df.columns if col not in ['subset', 'model_name', 'seed']]

            # Group by subset and model, then calculate mean and std
            grouped = results_df.groupby(['subset', 'model_name'])
            mean_results = grouped[metric_cols].mean()
            std_results = grouped[metric_cols].std().fillna(0)

            # Format results as "mean ± std" strings
            summary_df = pd.DataFrame(index=mean_results.index)
            for col in metric_cols:
                summary_df[col] = (
                    mean_results[col].map('{:.4f}'.format) + ' ± ' +
                    std_results[col].map('{:.4f}'.format)
                )

            # Display and save the final summary
            print("\n\n" + "="*80)
            print(" " * 28 + "EXPERIMENT SWEEP SUMMARY")
            print("="*80)
            print(summary_df)

            summary_path = Path(SAVE_DIR) / "benchmark_summary.csv"
            summary_df.to_csv(summary_path)
            print(f"\nFinal summary saved to {summary_path}")