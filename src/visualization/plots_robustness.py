"""Visualization for Robustness, Regret, and Policy Stability (Figures 9 and 10).

Generates:
- Policy Stability Heatmap (Jaccard similarity matrix across scenarios, Figure 10)
- Scenario Regret Distribution / Minimax Regret comparison (Figure 9)
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", palette="deep")


def plot_policy_stability_heatmap(
    stability_matrix: pd.DataFrame,
    output_path: str = "outputs/figures/figure10_policy_stability_heatmap.png",
) -> None:
    """Plot heatmap of Jaccard similarity between targeted cohorts across scenarios."""
    plt.figure(figsize=(10, 8), dpi=300)

    # Clean scenario labels for display if needed
    sns.heatmap(
        stability_matrix,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Jaccard Targeting Stability Index (PS)"},
        linewidths=0.5,
    )

    plt.title("Targeting Policy Stability Across Commercial Scenarios (Jaccard Matrix)")
    plt.xlabel("Commercial Scenario (Margin & Cost)")
    plt.ylabel("Commercial Scenario (Margin & Cost)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Policy stability heatmap saved to {out_file}")


def plot_scenario_regret_comparison(
    regret_summary: pd.DataFrame,
    output_path: str = "outputs/figures/figure9_robust_vs_scenario_regret.png",
) -> None:
    """Plot max regret and mean regret comparison across candidate marketing policies."""
    plt.figure(figsize=(9, 5), dpi=300)

    policies = [c for c in regret_summary.columns]
    max_regrets = regret_summary.loc["MAX_REGRET", policies]
    mean_regrets = regret_summary.loc["MEAN_REGRET", policies]

    x = np.arange(len(policies))
    width = 0.35

    plt.bar(x - width/2, max_regrets, width, label="Worst-Case (Max) Regret", color="#d95f02", alpha=0.85)
    plt.bar(x + width/2, mean_regrets, width, label="Average (Mean) Regret", color="#7570b3", alpha=0.85)

    plt.ylabel("Economic Regret ($ per Customer)")
    plt.title("Policy Regret Across Economic Scenarios (Minimax Evaluation)")
    plt.xticks(x, policies, rotation=25, ha="right")
    plt.legend(frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Scenario regret plot saved to {out_file}")
