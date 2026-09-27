"""Large-Scale Benchmark Pipeline for Criteo Uplift (~13.98M observations).

Optimized specifically for the Power PC hardware architecture:
- Intel Core i9-14900K (24C / 32T)
- NVIDIA GeForce RTX 4090 (24GB VRAM)
- 64GB DDR5 RAM

Tests Hypothesis H7:
Do the core causal-targeting findings (Response vs Uplift ranking discrepancy,
efficiency gains) generalize to massive advertising incrementality experiments?
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from src.utils.hardware import get_hardware_info, get_xgboost_device_params
from src.models.response_baselines import ResponseBaseline
from src.models.meta_learners import TLearner, SLearner
from src.evaluation.metrics import compute_qini_score, compute_uplift_curve
from src.visualization.plots_uplift import plot_uplift_curves


def run_criteo_benchmark(
    use_full: bool = False,
    sample_size: int = 500_000,
    target_outcome: str = "conversion",
    device: str = "auto",
):
    """Execute large-scale Criteo benchmark on GPU workstation."""
    from src.data.download_criteo import download_criteo, extract_criteo_subsample, DEFAULT_RAW_DEST

    hw = get_hardware_info()
    print("=" * 80)
    print("CRITEO UPLIFT EXPERIMENTAL BENCHMARK (POWER PC ACCELERATION)")
    print(f"Hardware: {hw['cpu_count']} CPU Threads | GPU: {hw['gpu_name']} ({hw['gpu_mem_gb']} GB VRAM)")
    print("=" * 80)

    raw_path = Path(DEFAULT_RAW_DEST)
    sample_path = Path("data/processed/criteo_sample_500k.parquet")

    # Step 1: Ensure dataset is downloaded
    if not raw_path.exists() and not (not use_full and sample_path.exists()):
        print("\n[INFO] Criteo raw dataset not found. Downloading compressed archive (~300MB)...")
        download_criteo(dest_path=str(raw_path))

    start_t = time.time()
    if use_full:
        print(f"\n[1/4] Loading FULL 13.98 Million observation Criteo dataset from {raw_path}...")
        df = pd.read_csv(raw_path, compression="gzip", dtype={f"f{i}": np.float32 for i in range(12)})
    else:
        if not sample_path.exists():
            print(f"\n[INFO] Extracting {sample_size:,} row sample to {sample_path}...")
            extract_criteo_subsample(source_gz=str(raw_path), output_csv=str(sample_path), sample_size=sample_size)
        print(f"\n[1/4] Loading {sample_size:,} sample from {sample_path}...")
        df = pd.read_parquet(sample_path)

    feature_cols = [f"f{i}" for i in range(12)]
    X = df[feature_cols].astype(np.float32)
    T = df["treatment"].astype(np.int32)
    y = df[target_outcome].astype(np.int32)

    n_total = len(df)
    n_train = int(n_total * 0.70)

    X_tr, X_te = X.iloc[:n_train], X.iloc[n_train:]
    T_tr, T_te = T.iloc[:n_train], T.iloc[n_train:]
    y_tr, y_te = y.iloc[:n_train], y.iloc[n_train:]

    print(f"Dataset split: Train N = {len(X_tr):,}, Test N = {len(X_te):,}")
    print(f"Treatment share: {T.mean():.2%}, Conversion rate: {y.mean():.4%}")

    # 2. Train Response vs Uplift Models on RTX 4090
    print("\n[2/4] Training models using GPU hardware acceleration (RTX 4090)...")
    t0 = time.time()
    resp_model = ResponseBaseline(model_type="xgboost", device=device).fit(X_tr, y_tr)
    print(f"  Response model trained in {time.time() - t0:.2f}s")

    t1 = time.time()
    uplift_model = TLearner(is_classification=True, device=device).fit(X_tr, T_tr, y_tr)
    print(f"  T-Learner uplift model trained in {time.time() - t1:.2f}s")

    # 3. Score Test Set
    print("\n[3/4] Generating predictions on held-out test data...")
    p_resp = resp_model.predict_proba(X_te)
    tau_uplift = uplift_model.predict_cate(X_te)

    # Hypothesis H1 on Criteo
    rho, pval = stats.spearmanr(p_resp, tau_uplift)
    top10_n = int(0.10 * len(X_te))
    top10_resp = set(np.argsort(p_resp)[::-1][:top10_n])
    top10_up = set(np.argsort(tau_uplift)[::-1][:top10_n])
    overlap = len(top10_resp.intersection(top10_up)) / top10_n

    print(f"\n[4/4] Generalization Results (Criteo Benchmark):")
    print(f"  Spearman rho(Response, CATE Uplift): {rho:.4f} (p = {pval:.2e})")
    print(f"  Top-10% Targeted Customer Overlap:  {overlap:.1%}")

    q_resp = compute_qini_score(y_te.values, T_te.values, p_resp)
    q_uplift = compute_qini_score(y_te.values, T_te.values, tau_uplift)

    results_table = [
        {"Dataset": "Criteo Uplift", "Mode": "Full (14M)" if use_full else f"Sample ({sample_size:,})", "Model": "Response (XGBoost)", "Qini Score": q_resp},
        {"Dataset": "Criteo Uplift", "Mode": "Full (14M)" if use_full else f"Sample ({sample_size:,})", "Model": "T-Learner Uplift", "Qini Score": q_uplift},
    ]
    print("\n" + tabulate(results_table, headers="keys", tablefmt="github", floatfmt=".4f"))

    # Plot Criteo Curve
    plot_uplift_curves(
        y=y_te.values,
        treatment=T_te.values,
        models_scores={"Response (XGBoost)": p_resp, "T-Learner Uplift": tau_uplift},
        outcome_name=f"Criteo {target_outcome.capitalize()}",
        output_path="outputs/figures/figure11_criteo_uplift_curves.png"
    )

    out_file = Path("outputs/tables/table11_criteo_replication.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write("### Table 11: Large-Scale External Replication (Criteo Uplift Benchmark)\n\n")
        f.write(tabulate(results_table, headers="keys", tablefmt="github", floatfmt=".4f") + "\n\n")
        f.write(f"- **Spearman Rank Correlation**: rho = {rho:.4f} (p = {pval:.2e})\n")
        f.write(f"- **Top-10% Cohort Overlap**: {overlap:.1%}\n")
        f.write(f"- **Total Rows Evaluated**: {n_total:,}\n")

    print(f"\n[COMPLETE] Criteo benchmark finished in {time.time() - start_t:.2f} seconds!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Criteo Uplift Benchmark on RTX 4090")
    parser.add_argument("--full", action="store_true", help="Run on full 13.98M observation dataset (Power PC recommended)")
    parser.add_argument("--sample-size", type=int, default=500_000, help="Sample size if not running full dataset (default: 500k)")
    parser.add_argument("--outcome", type=str, default="conversion", choices=["conversion", "visit"], help="Target outcome")
    args = parser.parse_args()

    run_criteo_benchmark(use_full=args.full, sample_size=args.sample_size, target_outcome=args.outcome)
