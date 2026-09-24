"""Visualization for Economic Policy Frontiers and Budget Curves (Figures 5, 6, 7, 8).

Generates:
- Policy Value vs Targeting Coverage (Figure 5)
- Policy Value vs Campaign Unit Cost (Figure 6)
- Budget-Performance Efficient Frontier (Figure 8)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
})


def plot_policy_value_vs_coverage(
    coverage_fractions: List[float],
    policy_values_by_coverage: Dict[str, List[float]],
    output_path: str = "outputs/figures/figure5_policy_value_vs_coverage.png",
) -> None:
    """Plot net economic policy value as a function of contact coverage."""
    plt.figure(figsize=(9, 6), dpi=300)

    for pol_name, vals in policy_values_by_coverage.items():
        plt.plot(np.array(coverage_fractions) * 100, vals, marker="o", markersize=4, label=pol_name)

    plt.axhline(0, color="black", linestyle=":", alpha=0.6)
    plt.title("Net Economic Policy Value vs Campaign Coverage (Held-Out Test Set)")
    plt.xlabel("Campaign Contact Coverage (% of Total Customer Base)")
    plt.ylabel("Net Policy Value ($ per Customer in Population)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Policy coverage plot saved to {out_file}")


def plot_cost_sensitivity(
    cost_grid: List[float],
    policy_values_by_cost: Dict[str, List[float]],
    output_path: str = "outputs/figures/figure6_policy_value_vs_cost.png",
) -> None:
    """Plot net policy value as unit campaign contact cost increases."""
    plt.figure(figsize=(9, 6), dpi=300)

    for pol_name, vals in policy_values_by_cost.items():
        plt.plot(cost_grid, vals, marker="s", markersize=4, label=pol_name)

    plt.axhline(0, color="black", linestyle=":", alpha=0.6)
    plt.title("Sensitivity to Campaign Contact Cost c (Fixed Margin M = 50%)")
    plt.xlabel("Campaign Unit Contact Cost ($)")
    plt.ylabel("Net Economic Policy Value ($ per Customer)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Cost sensitivity plot saved to {out_file}")
