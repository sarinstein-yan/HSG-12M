import os
import time
import argparse

import numpy as np
import sympy as sp
import poly2graph as p2g
import pickle
from joblib import Parallel, delayed

def generate_dataset(
    class_idx: int,
    class_data: str = "./ParamTable.npz",
    save_dir: str = "./poly_class_dataset",
    num_partition: int = 10,
    short_edge_threshold: int = 30,
) -> None:
    """
    Generate and save spectral graph data for one polynomial class.

    Reads 'polys' and 'param_vals' from class_data NPZ, splits parameters
    into partitions, builds CharPolyClass, and stores pickled graphs.
    """
    # Define working symbols and parameter dict
    k, z, E, a, b = sp.symbols('k z E a b')
    params = {a, b}

    # Load param table with all parameter combinations
    data = np.load(class_data, allow_pickle=True)
    polys = data['polys']       # array of expression strings
    metas = data['metas']       # array of metadata dicts
    param1_vals, param2_vals = data['param_vals']

    # Split parameter sweeps into partitions
    p1_parts = np.array_split(param1_vals, num_partition)
    p2_parts = np.array_split(param2_vals, num_partition)

    print(f"[Class {class_idx}]: {polys[class_idx]}")
    cp = p2g.CharPolyClass(polys[class_idx], k, z, E, params)
    os.makedirs(save_dir, exist_ok=True)

    batcher = Parallel(n_jobs=-1, prefer='threads')
    for i, (v1, v2) in enumerate(zip(p1_parts, p2_parts)):
        print(f"[Class {class_idx}] Partition {i} - computing...")
        t0 = time.perf_counter()
        graphs, _ = cp.spectral_graph(
            {a: v1, b: v2}, 
            short_edge_threshold=short_edge_threshold,
        )

        # Save pickled graphs and their parameter values
        out = os.path.join(save_dir, f'class_{class_idx}_part_{i}.npz')
        graphs_ser = batcher(
            delayed(pickle.dumps)(g) for g in graphs
        )
        np.savez_compressed(
            out,
            graphs_pickle=graphs_ser,
            a_vals=v1,
            b_vals=v2,
        )
        dt = (time.perf_counter() - t0) / 60
        print(
            f"[Class {class_idx}] Partition {i} saved → {out}"
            f" ({len(graphs_ser)} graphs, took {dt:.4f} min)"
        )
    
    # Combine all partitions into a single NPZ
    all_pickles = []
    for i in range(num_partition):
        d = np.load(
            os.path.join(save_dir, f'class_{class_idx}_part_{i}.npz'), 
            allow_pickle=True
            )
        # .tolist() because np.savez loads object arrays as dtype=object
        all_pickles.extend(d['graphs_pickle'].tolist())
        os.remove(os.path.join(save_dir, f'class_{class_idx}_part_{i}.npz'))

    out = os.path.join(save_dir, f'class_{class_idx}.npz')
    np.savez_compressed(
        out,
        graphs_pickle=np.array(all_pickles, dtype=object),
        y=class_idx,
        a_vals=param1_vals,
        b_vals=param2_vals,
        **metas[class_idx],
    )
    print(f"[Class {class_idx}] Combined dataset saved → {out}")

def main():
    p = argparse.ArgumentParser(
        description="Generate spectral graph dataset for a given polynomial class."
    )
    p.add_argument(
        "--class_idx", required=True, type=int,
        help="Which class to process (0-1400)."
    )
    p.add_argument(
        "--class_data", default="./ParamTable.npz",
        help="Directory containing ParamTable.npz"
    )
    p.add_argument(
        "--save_dir", default="./poly_class_dataset",
        help="Where to write the .npz output files."
    )
    p.add_argument(
        "--num_partition", type=int, default=10,
        help="How many chunks to split the parameter arrays into."
    )
    p.add_argument(
        "--short_edge_threshold", type=int, default=30,
        help="Threshold for short edges in the graph."
    )
    args = p.parse_args()

    generate_dataset(
        args.class_idx,
        args.class_data,
        args.save_dir,
        args.num_partition,
        args.short_edge_threshold
    )

if __name__ == "__main__":
    main()