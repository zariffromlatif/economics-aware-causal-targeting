"""Treatment-Effect Heterogeneity and Subgroup Analysis.

Analyzes which pre-treatment customer characteristics moderate causal effects:
1. Subgroup Average Treatment Effects (CATE conditional on recency, channel, category history).
2. Four Customer Archetypes classification (Persuadables, Sure Things, Lost Causes, Sleeping Dogs).
3. Export Table 10 (Subgroup Treatment Effects).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data.loader import prepare_hillstrom_splits


def compute_subgroup_effects(
    df: pd.DataFrame,
    treatment_arm: str = "Mens E-Mail",
    control_arm: str = "No E-Mail",
) -> pd.DataFrame:
    """Compute empirical conditional average treatment effects across customer subgroups."""
    sub_df = df[df["segment"].isin([control_arm, treatment_arm])].copy()
    sub_df["T"] = (sub_df["segment"] == treatment_arm).astype(int)

    records = []

    # 1. Recency Subgroups (Recent: <= 6 months, Lapsed: > 6 months)
    for rec_label, mask in [("Recent (<= 6m)", sub_df["recency"] <= 6),
                            ("Lapsed (> 6m)", sub_df["recency"] > 6)]:
        subset = sub_df[mask]
        t1, t0 = subset[subset["T"] == 1], subset[subset["T"] == 0]
        ate_conv = (t1["conversion"].mean() - t0["conversion"].mean()) * 100
        ate_spend = t1["spend"].mean() - t0["spend"].mean()
        records.append({
            "Dimension": "Recency",
            "Subgroup": rec_label,
            "Sample Size": len(subset),
            "Delta Conv (%)": ate_conv,
            "Delta Spend ($)": ate_spend,
        })

    # 2. Prior Category Purchasing (Purchased Men's vs Not)
    for mens_label, mask in [("Prior Mens Buyer", sub_df["mens"] == 1),
                             ("No Prior Mens Buy", sub_df["mens"] == 0)]:
        subset = sub_df[mask]
        t1, t0 = subset[subset["T"] == 1], subset[subset["T"] == 0]
        ate_conv = (t1["conversion"].mean() - t0["conversion"].mean()) * 100
        ate_spend = t1["spend"].mean() - t0["spend"].mean()
        records.append({
            "Dimension": "Prior Mens Purchasing",
            "Subgroup": mens_label,
            "Sample Size": len(subset),
            "Delta Conv (%)": ate_conv,
            "Delta Spend ($)": ate_spend,
        })

    # 3. Newbie Indicator (New customer vs Established)
    for new_label, mask in [("New Customer", sub_df["newbie"] == 1),
                            ("Established", sub_df["newbie"] == 0)]:
        subset = sub_df[mask]
        t1, t0 = subset[subset["T"] == 1], subset[subset["T"] == 0]
        ate_conv = (t1["conversion"].mean() - t0["conversion"].mean()) * 100
        ate_spend = t1["spend"].mean() - t0["spend"].mean()
        records.append({
            "Dimension": "Customer Tenure",
            "Subgroup": new_label,
            "Sample Size": len(subset),
            "Delta Conv (%)": ate_conv,
            "Delta Spend ($)": ate_spend,
        })

    # 4. Shopping Channel (Phone, Web, Multichannel)
    for ch in ["Phone", "Web", "Multichannel"]:
        subset = sub_df[sub_df["channel"] == ch]
        t1, t0 = subset[subset["T"] == 1], subset[subset["T"] == 0]
        ate_conv = (t1["conversion"].mean() - t0["conversion"].mean()) * 100
        ate_spend = t1["spend"].mean() - t0["spend"].mean()
        records.append({
            "Dimension": "Channel Preference",
            "Subgroup": ch,
            "Sample Size": len(subset),
            "Delta Conv (%)": ate_conv,
            "Delta Spend ($)": ate_spend,
        })

    subgroup_df = pd.DataFrame(records)
    return subgroup_df


def generate_subgroup_table(output_dir: str = "outputs/tables") -> pd.DataFrame:
    """Generate and save Table 10: Subgroup Treatment Effects."""
    df_raw = pd.read_csv("data/raw/hillstrom.csv")
    table10 = compute_subgroup_effects(df_raw)

    out_file = Path(output_dir) / "table10_subgroup_treatment_effects.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("### Table 10: Heterogeneous Subgroup Treatment Effects\n\n")
        f.write(tabulate(table10, headers="keys", tablefmt="github", floatfmt=".4f") + "\n")

    print(f"[SUCCESS] Table 10 saved to {out_file}")
    return table10


if __name__ == "__main__":
    generate_subgroup_table()
