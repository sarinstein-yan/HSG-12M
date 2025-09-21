import time
from pathlib import Path
import torch
import pandas as pd
from hsg import Config, run_experiment

torch.set_float32_matmul_precision("medium")

# --- Sweep Configuration ---
DATA_ROOT = "/mnt/ssd/nhsg12m"
SAVE_DIR = "/mnt/ssd/nhsg12m/HSG-12M/dev/baseline_sweep"

SUBSETS = ["one-band", "two-band", "three-band", "topology", "all"]
MODEL_NAMES = ["mf", "gcn", "sage", "gat", "gin", "cgcnn", "monet"]
SEEDS = [42, 2025, 666]
EPOCHS = 1
BATCH_SIZE = 3600

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
# FOR TESTING
for subset in ["one-band", "all", "two-band", "three-band", "topology"]:
    for model_name in MODEL_NAMES:
        for seed in SEEDS[:2]:
            # Create config for the current run
            cfg = Config(
                data_root=DATA_ROOT,
                save_dir=SAVE_DIR,
                subset=subset,
                seed=seed,
                model_name=model_name,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                dim_h_gnn=DIM_H_GNN[subset][model_name],
                dim_h_mlp=DIM_H_MLP[subset],
            )

            try:
                results = run_experiment(cfg)
                # Add config details for easy grouping later
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
                print(f"‼️ ERROR running {model_name} on {subset} with seed {seed}: {e}")
                # Optionally, log the error to a file
                with open(Path(SAVE_DIR) / "error_log.txt", "a") as f:
                    f.write(f"[{time.ctime()}] ERROR on {model_name}/{subset}/seed{seed}: {e}\n")
                continue

# --- Aggregate and Summarize Final Results ---
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