# ---------------------------------------------------------------------------
# 0. Early Environment Setup (CRITICAL for DDP)
# This must happen BEFORE importing torch or pytorch_lightning.
# ---------------------------------------------------------------------------
import os

# ----------------- robust single-node DDP env -----------------
# Keep the rendezvous local.
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500" # Ensure this port is free

# FIX: The logs show NCCL using 'eno2', causing hangs.
# We explicitly force the loopback interface ('lo') to ensure reliable communication.
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

# Optional: Clarify logs and ensure robust error handling
os.environ["NCCL_DEBUG"] = "INFO"
# Use the modern variable names as suggested by the warnings in the logs
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL" # Uncomment for verbose debugging

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
from argparse import Namespace
from types import SimpleNamespace

import torch
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
from hsg import HSGInMemory, get_model_instance, StaticBatchSampler

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision

# Set precision for Tensor Cores optimization (as suggested by logs for A5000)
torch.set_float32_matmul_precision('medium') # or 'high'


# ---------------------------------------------------------------------------
# DataModule using StaticBatchSampler
# ---------------------------------------------------------------------------
class HSGLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        subset,
        *,
        max_num: int,
        mode: str = "edge",
        seeds: Sequence[int] = (0,),
        skip_too_big: bool = True,
        drop_last: bool = False,
        pack_strategy: str = "sequential",  # or "sorted_desc"
        shuffle_batch_order: bool = False,  # shuffle batch *order* each epoch
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        if mode not in ("node", "edge"):
            raise ValueError("mode must be 'node' or 'edge'")
        self.root, self.subset = root, subset
        self.max_num = int(max_num)
        self.mode = mode
        self.seeds = list(seeds)
        self.skip_too_big = bool(skip_too_big)
        self.drop_last = bool(drop_last)
        self.pack_strategy = pack_strategy
        self.shuffle_batch_order = bool(shuffle_batch_order)

        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and (self.num_workers > 0)

        self.datasets: List[Tuple[Subset, Subset, Subset]] = []
        self.sizes_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def prepare_data(self):
        # Trigger initial download/processing (executed on rank 0 only)
        _ = HSGInMemory(self.root, self.subset)

    def setup(self, stage=None):
        # Setup (executed on all ranks)
        self.datasets.clear()
        self.sizes_triplets.clear()

        full = HSGInMemory(self.root, self.subset)
        if len(full) == 0:
            return

        y_all = full.y.numpy()

        # Precomputed sizes provided by your dataset:
        size_key = "n_nodes" if self.mode == "node" else "n_edges"
        if not hasattr(full, size_key):
             raise RuntimeError(f"Dataset missing size attribute: {size_key}")
             
        size_array = np.asarray(
            getattr(full, size_key),
            dtype=np.int64,
        )
        if len(size_array) != len(full):
            raise RuntimeError("Size array length mismatch with dataset length.")

        for seed in self.seeds:
            # 1) train/test split (stratified on labels)
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=0.8, test_size=0.2, random_state=seed
            )
            idx_train, idx_tmp = next(splitter.split(np.zeros_like(y_all), y_all))
            y_tmp = y_all[idx_tmp]

            # 2) val/test split
            counts = np.bincount(y_tmp)
            # NOTE: if some classes are absent, counts has zeros; guard with >0
            safe_min = counts[counts > 0].min() if (counts > 0).any() else 0
            if safe_min < 2:
                ss = ShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
                rel_val, rel_test = next(ss.split(idx_tmp))
                # idx_val = idx_tmp[rel_val] # Unused in this branch
                # idx_test = idx_tmp[rel_test] # Unused in this branch
            else:
                splitter_val = StratifiedShuffleSplit(
                    n_splits=1, train_size=0.5, test_size=0.5, random_state=seed
                )
                rel_val, rel_test = next(
                    splitter_val.split(np.zeros_like(y_tmp), y_tmp)
                )

            idx_val = idx_tmp[rel_val]
            idx_test = idx_tmp[rel_test]

            # Build Subsets
            ds_train = Subset(full, idx_train)
            ds_val = Subset(full, idx_val)
            ds_test = Subset(full, idx_test)

            # Slice precomputed sizes to align with subset-relative indices
            sz_train = size_array[idx_train]
            sz_val = size_array[idx_val]
            sz_test = size_array[idx_test]

            self.datasets.append((ds_train, ds_val, ds_test))
            self.sizes_triplets.append((sz_train, sz_val, sz_test))

    # Helper to build a loader with a StaticBatchSampler
    def _make_loader(self, subset: Subset, sizes: np.ndarray, seed: int):
        sampler = StaticBatchSampler(
            dataset=subset,
            sizes=sizes,
            max_num=self.max_num,
            skip_too_big=self.skip_too_big,
            drop_last=self.drop_last,
            pack_strategy=self.pack_strategy,
            shuffle_batches_each_epoch=self.shuffle_batch_order,
            seed=seed,
            dist_shard=True, # Sampler handles DDP sharding
            ensure_equal_batch_counts=True,
        )
        # Important: when using batch_sampler, DO NOT pass batch_size/shuffle
        return DataLoader(
            subset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self, seed_idx: int = 0):
        ds_train, _, _ = self.datasets[seed_idx]
        sz_train, _, _ = self.sizes_triplets[seed_idx]
        return self._make_loader(ds_train, sz_train, seed=self.seeds[seed_idx])

    def val_dataloader(self, seed_idx: int = 0):
        _, ds_val, _ = self.datasets[seed_idx]
        _, sz_val, _ = self.sizes_triplets[seed_idx]
        # Keep validation deterministic; reuse the same seed
        return self._make_loader(ds_val, sz_val, seed=self.seeds[seed_idx])

    def test_dataloader(self, seed_idx: int = 0):
        _, _, ds_test = self.datasets[seed_idx]
        _, _, sz_test = self.sizes_triplets[seed_idx]
        return self._make_loader(ds_test, sz_test, seed=self.seeds[seed_idx])

# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class LitGNN(pl.LightningModule):
    def __init__(self, hparams, num_classes: int, in_dim: int):
        super().__init__()
        # Ensure hparams is handled correctly whether it's Namespace or dict
        hparams_dict = vars(hparams) if isinstance(hparams, (Namespace, SimpleNamespace)) else dict(hparams)
        self.save_hyperparameters(hparams_dict)

        # --- Prepare extra kwargs (logic remains the same) ---
        extra_kwargs = {}
        needs_edge_dim = self.hparams.model in {"cgcnn", "spline", "monet", "ecc"}
        edge_dim = getattr(self.hparams, "edge_dim", None)
        if needs_edge_dim and edge_dim is not None:
            extra_kwargs["edge_dim"] = int(edge_dim)

        # Optional hyper-params for spline/monet
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

        # Metrics setup (keeping metrics on CPU)
        self._tm = SimpleNamespace()
        self._tm.train_acc = Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        # We will reuse the same ModuleList for validation and testing
        self._tm.eval_accs = torch.nn.ModuleList([
            Accuracy(top_k=k, task="multiclass", num_classes=num_classes) for k in self.topks
        ])
        self._tm.eval_f1_macro = F1Score(average="macro", task="multiclass", num_classes=num_classes)
        self._tm.eval_f1_micro = F1Score(average="micro", task="multiclass", num_classes=num_classes)
        self._tm.eval_auc = AUROC(average="macro", task="multiclass", num_classes=num_classes)
        self._tm.eval_ap = AveragePrecision(average="macro", task="multiclass", num_classes=num_classes)

        for m in [self._tm.train_acc, *self._tm.eval_accs,
                  self._tm.eval_f1_macro, self._tm.eval_f1_micro,
                  self._tm.eval_auc, self._tm.eval_ap]:
            m.to("cpu")
            # Let torchmetrics handle cross-process reduction at compute() time
            m.sync_on_compute = True

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
            self._train_samples += y.size(0) # Local count on this rank
        else:
            # Update eval metrics
            for acc in self._tm.eval_accs:
                acc.update(logits_cpu, y_cpu)
            self._tm.eval_f1_macro.update(logits_cpu, y_cpu)
            self._tm.eval_f1_micro.update(logits_cpu, y_cpu)
            self._tm.eval_auc.update(logits_cpu, y_cpu)
            self._tm.eval_ap.update(logits_cpu, y_cpu)

        # Must use sync_dist=True for loss in DDP (averages loss across ranks)
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
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        # DDP FIX: CRITICAL for custom samplers when shuffling is enabled.
        # We must manually call set_epoch to ensure different randomization across epochs.
        if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
             self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)

    # DDP FIX: Correctly aggregate throughput and memory usage across DDP ranks
    def on_train_epoch_end(self):
        # ---- core metrics
        self.log("train_top1", self._tm.train_acc.compute(), sync_dist=False)
        # ^ sync_dist=False because torchmetrics already synced across ranks at compute()
        self._tm.train_acc.reset()

        # ---- wall time, throughput, GPU memory (DDP Aware Updates)
        wall_time = time.perf_counter() - self._epoch_start_time
        # Log wall time (sync_dist=True averages it across ranks)
        self.log("train_wall_time_s_avg", wall_time, sync_dist=True)

        # DDP: Aggregate sample count across all ranks (SUM)
        # Move local count to the current GPU device for synchronization
        local_samples = torch.tensor(self._train_samples, dtype=torch.float64, device=self.device)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(local_samples, op=torch.distributed.ReduceOp.SUM)
        total_samples = local_samples.item()

        # Calculate total throughput
        throughput = total_samples / wall_time if wall_time > 0 else 0.0

        # DDP: Get the maximum GPU memory usage across all ranks (MAX)
        mem_gb = 0.0
        if torch.cuda.is_available() and self.device.type == 'cuda':
            mem_alloc = torch.cuda.max_memory_allocated(self.device)
            mem_tensor = torch.tensor(mem_alloc, dtype=torch.float64, device=self.device)
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                 # Get the MAX memory usage across ranks
                 torch.distributed.all_reduce(mem_tensor, op=torch.distributed.ReduceOp.MAX)
            mem_gb = mem_tensor.item() / 1024**3

        # Log aggregated metrics only on rank 0.
        self.log("train_throughput_samples_s_total", throughput, sync_dist=False, rank_zero_only=True)
        self.log("train_gpu_mem_gb_max", mem_gb, sync_dist=False, rank_zero_only=True)

    def _on_eval_epoch_end(self, stage: str):
        # Helper for validation and test epoch end
        for k, acc in zip(self.topks, self._tm.eval_accs):
            self.log(f"{stage}_top{k}", acc.compute(), sync_dist=False); acc.reset()
        self.log(f"{stage}_macro_f1",  self._tm.eval_f1_macro.compute(), sync_dist=False)
        self.log(f"{stage}_micro_f1",  self._tm.eval_f1_micro.compute(), sync_dist=False)
        self.log(f"{stage}_macro_auc", self._tm.eval_auc.compute(), sync_dist=False)
        self.log(f"{stage}_macro_ap",  self._tm.eval_ap.compute(), sync_dist=False)
        self._tm.eval_f1_macro.reset(); self._tm.eval_f1_micro.reset()
        self._tm.eval_auc.reset(); self._tm.eval_ap.reset()

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

# Helper for summarising seeds → mean ± std CSV
def summarise_csv(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    rows = [{"metric": k, "mean": means[k], "std": stds[k]} for k in means.index]
    pd.DataFrame(rows).to_csv(csv_out, index=False)


# ---------------------------------------------------------------------------
# 2. Experiment Runner (Updated for DDP correctness)
# ---------------------------------------------------------------------------

def run_experiment(args: Namespace):
    """
    Train/validate/test all requested GNN architectures.
    """
    # Helper for clean DDP logging (only print on the main process)
    def rank_zero_print(*p_args, **p_kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(*p_args, **p_kwargs)

    # ---------- house-keeping ----------
    save_path = Path(args.save_dir) / args.subset
    # Ensure directory creation happens only on rank 0
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        save_path.mkdir(parents=True, exist_ok=True)

    if args.models == ["all"]:
        args.models = ["gcn", "sage", "gat", "gatv2", "gin", "gine",
                       "mf", "cgcnn", "ecc", "spline", "monet"]

    rank_zero_print(f"⏳  Loading PolyGraph from {args.root}, subset={args.subset}, "
          f"batch_cap={int(args.num_max)} {args.mode}s, "
          f"models={args.models}, seeds={args.seeds}")

    # ---------- data ----------
    # Provide defaults for arguments that might be missing in the Namespace
    dm = HSGLitDataModule(
        root=args.root,
        subset=args.subset,
        max_num=int(args.num_max),
        mode=str(getattr(args, "mode", "edge")),
        seeds=args.seeds,
        skip_too_big=bool(getattr(args, "skip_too_big", True)),
        drop_last=bool(getattr(args, "drop_last", False)),
        pack_strategy=str(getattr(args, "pack_strategy", "sorted_desc")),
        shuffle_batch_order=bool(getattr(args, "shuffle_batch_order", False)),
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", True)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
    )
    dm.prepare_data()
    dm.setup()

    if not dm.datasets:
        rank_zero_print("❌ Dataset setup failed or dataset is empty. Exiting.")
        return

    # Underlying base dataset (shared across seeds):
    base_dataset = dm.datasets[0][0].dataset

    # ---------- loop over architectures ----------
    for model_name in args.models:
        rank_zero_print(f"\n🧠  ▶ Training {model_name} …")
        args.model = model_name
        summaries = []

        # (Optional) add Cartesian pseudo-coordinates for specific models:
        if model_name in {"spline", "monet"}:
            base_dataset.transform = T.Cartesian(cat=False)
            rank_zero_print("   • Applied transforms.Cartesian(cat=False) for pseudo-coordinates.")

        # Detect edge_dim once (same across splits):
        if len(dm.datasets[0][0]) > 0:
            probe = dm.datasets[0][0][0]
            args.edge_dim = (int(probe.edge_attr.size(-1))
                            if getattr(probe, "edge_attr", None) is not None else None)
            if model_name in {"cgcnn", "ecc", "spline", "monet"}:
                rank_zero_print(f"   • Detected edge_dim={args.edge_dim} for {model_name}.")
        else:
            args.edge_dim = None

        for seed_idx, seed in enumerate(args.seeds):
            pl.seed_everything(seed, workers=True)

            train_loader = dm.train_dataloader(seed_idx)
            val_loader   = dm.val_dataloader(seed_idx)
            test_loader  = dm.test_dataloader(seed_idx)

            # Log steps/epoch (now fixed thanks to static sampler):
            try:
                # Note: In DDP, len(train_loader) returns the number of batches PER RANK.
                n_steps = len(train_loader)
                rank_zero_print(f"   • Seed {seed}: steps/epoch={n_steps} (per GPU) "
                      f"(cap {int(args.num_max)} {args.mode}s)")
            except Exception:
                pass

            dataset_properties = train_loader.dataset.dataset
            num_classes = dataset_properties.num_classes
            in_dim      = dataset_properties.num_node_features
            rank_zero_print(f"   • Seed {seed}: {num_classes} classes, {in_dim} input feats")

            logger = TensorBoardLogger(
                save_path / "tb_logs" / model_name,
                name=f"{model_name}_seed{seed}"
            )
            ckpt_dir = save_path / model_name / f"seed_{seed}"
            # Directory creation handled by ModelCheckpoint, but safe guarding for rank 0
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
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

            # Trainer Configuration
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=args.devices,
                # Explicitly use DDP Strategy
                strategy=DDPStrategy(find_unused_parameters=False),
                # CRITICAL: Disable Lightning's automatic distributed sampler replacement.
                # Our custom StaticBatchSampler handles the sharding manually.
                use_distributed_sampler=False,
                max_epochs=args.epochs,
                logger=logger,
                log_every_n_steps=args.log_every_n_steps,
                callbacks=[ckpt_top1, ckpt_f1, stopper],
                deterministic=True,
            )


            lit = LitGNN(args, num_classes, in_dim)
            # Initialize model parameters for DDP
            if trainer.is_global_zero:
                print("   ⚡️ Initializing model parameters with a dummy forward pass for DDP...")
            try:
                # Determine the target device for this DDP process
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{local_rank}")
                # Move model to the correct device
                lit.to(device)
                # Get a single batch and move it to the device
                dummy_batch = next(iter(train_loader))
                dummy_batch.to(device)
                # Perform the forward pass to initialize parameters
                with torch.no_grad():
                    lit(dummy_batch)
            except StopIteration:
                # Handle the case of an empty dataloader
                if trainer.is_global_zero:
                    print("   ⚠️ Train loader is empty, skipping dummy forward pass.")
            # Move model back to CPU; Lightning will handle placement from here.
            lit.cpu()


            rank_zero_print(f"   🚀 Starting training...")
            trainer.fit(lit, train_loader, val_loader)

            rank_zero_print(f"   🧪 Starting testing...")
            best_model_path = ckpt_f1.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                trainer.test(ckpt_path=best_model_path, dataloaders=test_loader)
            else:
                rank_zero_print(f"   ⚠️ No best model checkpoint found. Skipping testing.")


            # ---------- gather metrics ----------
            # Metrics are synchronized automatically, but we collect them on rank 0 for the summary CSV
            if trainer.is_global_zero:
                # Convert tensor metrics to float for CSV storage
                metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if v is not None}
                summaries.append({"seed": seed, **metrics})

            # Barrier to ensure all processes finish this seed before proceeding (optional, but safe)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # Reset transform so it doesn't leak:
        base_dataset.transform = None

        # ---------- write per-seed + summary CSV ----------
        # Write summary only on rank 0
        if int(os.environ.get("LOCAL_RANK", 0)) == 0 and summaries:
            df_path = save_path / f"{model_name}.csv"
            pd.DataFrame(summaries).to_csv(df_path, index=False)
            summarise_csv(df_path, df_path.with_name(df_path.stem + "_summary.csv"))
            print(f"✅  Finished {model_name}; results → {df_path}")


# ---------------------------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example args
    args = Namespace(
        root       = Path("/mnt/ssd/nhsg12m"),
        save_dir   = Path("/mnt/ssd/nhsg12m/HSG-12M/dev/baseline"),
        subset     = "one-band",
        models     = ['gcn'],#, 'mf', 'monet'],
        epochs     = 100,
        mode       = "edge",         # 'node' or 'edge'
        num_max    = 1e6,            # cap on total nodes/edges per batch
        skip_too_big = True,         # skip graphs larger than the cap
        dim_gnn    = 128,
        dim_mlp    = 1024,
        layers_gnn = 4,
        layers_mlp = 2,
        heads      = 1,
        dropout    = 0.,
        lr_init    = 1e-2,
        lr_min     = 1e-4,
        t0         = 34,
        t_mult     = 2,
        seeds                = [0],
        log_every_n_steps    = 5,
        early_stop_patience  = 20,
        edge_kernel          = 5,
        # Added defaults for arguments used in DataModule but missing in the original definition
        pack_strategy = "sorted_desc",
        num_workers = 0,
        pin_memory = True,
        persistent_workers = False,
        shuffle_batch_order = False,
        drop_last = False,
        devices = 2, # Set the number of devices
    )

    # ------------------------------------------------------------
    # 4. Convenience loop
    # ------------------------------------------------------------
    subsets_cfg = {
        "one-band":   dict(dim_gnn=128, dim_mlp=128, epochs=2, t0=2, t_mult=1),
        # "two-band":   dict(dim_gnn=256, dim_mlp=256,  epochs=1, t0=1, t_mult=1),
        # "topology":   dict(dim_gnn=512, dim_mlp=1500, epochs=1, t0=1, t_mult=1),
        # "three-band": dict(dim_gnn=512, dim_mlp=1500, epochs=1, t0=1, t_mult=1),
        # "all":        dict(dim_gnn=512, dim_mlp=1500, epochs=2, t0=1, t_mult=1),
    }

    # Run the experiments. PyTorch Lightning handles the DDP process launch.
    for subset, cfg in subsets_cfg.items():
        sweep_args = Namespace(**{**vars(args), "subset": subset, **cfg})
        run_experiment(sweep_args)