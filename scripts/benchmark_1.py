from argparse import Namespace
from pathlib import Path
from hsg import run_experiment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = Namespace(
    root         = Path("/mnt/ssd/nhsg12m"),
    save_dir     = Path("/mnt/ssd/nhsg12m/baseline_results"),
    models       = ["gcn", "sage", "gat", "gin", "mf", "cgcnn", "monet"],
    mode         = "edge",         # 'node' or 'edge'
    num_max      = 2e5,            # cap on total nodes/edges per batch
    layers_gnn   = 4,
    layers_mlp   = 2,
    dropout      = 0.,
    lr_init      = 1e-2,
    lr_min       = 1e-4,
    heads        = 1,
    edge_kernel  = 5,
    # epochs     = 100,
    # t0         = 34,
    # t_mult     = 2,
    seeds        = [42, 624, 706],
    log_every_n_steps    = 5,
    early_stop_patience  = 10,
    # # defaults for arguments used in DataModule
    # drop_last = False,
    # pack_strategy = "sorted_desc",
    # shuffle_batch_order = False,
    # num_workers = 0,
    # pin_memory = True,
    # persistent_workers = False,
)

# ------------------------------------------------------------
# 4. Convenience loop
# ------------------------------------------------------------
subsets_cfg = {
    # "one-band":   dict(dim_gnn=128, dim_mlp=128, epochs=40, t0=40, t_mult=1),
    # "topology":   dict(dim_gnn=512, dim_mlp=1500, epochs=15, t0=15, t_mult=1),
    # "three-band": dict(dim_gnn=512, dim_mlp=1500, epochs=5, t0=5, t_mult=1, seeds=[42]),
    "two-band":   dict(dim_gnn=256, dim_mlp=256,  epochs=20, t0=20, t_mult=1),
    "all":        dict(dim_gnn=512, dim_mlp=1500, epochs=5, t0=5, t_mult=1, seeds=[42]),
}

# Run the experiments. PyTorch Lightning handles the DDP process launch.
for subset, cfg in subsets_cfg.items():
    sweep_args = Namespace(**{**vars(args), "subset": subset, **cfg})
    run_experiment(sweep_args)