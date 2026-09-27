"""Multi-Arm Personalized Treatment Allocation and Policy Evaluation (Experiment B).

Evaluates personalized campaign allocation across the 3-arm Hillstrom experiment:
A in {0: 'No E-Mail', 1: 'Mens E-Mail', 2: 'Womens E-Mail'}

Tests Hypothesis H7:
Does customer-specific treatment selection create additional economic value
compared with applying the same uniform campaign to all targeted customers?
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from src.models.revenue_models import HurdleRevenueUplift
from src.evaluation.inference import paired_bootstrap_policy_comparison

sns.set_theme(style="whitegrid", palette="deep")


class MultiArmPersonalizedPolicy:
    """Manages multi-arm CATE models and personalized targeting decisions."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.hurdle_arm1 = HurdleRevenueUplift(random_state=random_state)  # Mens vs Control
        self.hurdle_arm2 = HurdleRevenueUplift(random_state=random_state)  # Womens vs Control

    def fit(self, X_train: pd.DataFrame, T_train: pd.Series, y_conv: pd.Series, y_spend: pd.Series) -> "MultiArmPersonalizedPolicy":
        """Fit separate uplift models for each active intervention arm relative to control."""
        # Arm 1: Mens (T=1) vs Control (T=0)
        mask_arm1 = T_train.isin([0, 1])
        X1 = X_train.loc[mask_arm1].reset_index(drop=True)
        T1 = T_train.loc[mask_arm1].reset_index(drop=True)
        yc1 = y_conv.loc[mask_arm1].reset_index(drop=True)
        ys1 = y_spend.loc[mask_arm1].reset_index(drop=True)
        self.hurdle_arm1.fit(X1, T1, yc1, ys1)

        # Arm 2: Womens (T=2) vs Control (T=0)
        mask_arm2 = T_train.isin([0, 2])
        X2 = X_train.loc[mask_arm2].reset_index(drop=True)
        T2 = (T_train.loc[mask_arm2] == 2).astype(int).reset_index(drop=True)
        yc2 = y_conv.loc[mask_arm2].reset_index(drop=True)
        ys2 = y_spend.loc[mask_arm2].reset_index(drop=True)
        self.hurdle_arm2.fit(X2, T2, yc2, ys2)

        return self

    def predict_incremental_revenues(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict expected incremental spend for each active treatment arm."""
        delta_r1 = self.hurdle_arm1.predict_incremental_revenue(X)
        delta_r2 = self.hurdle_arm2.predict_incremental_revenue(X)
        return delta_r1, delta_r2

    def compute_utilities(
        self,
        X: pd.DataFrame,
        margin: float = 0.50,
        cost_arm1: float = 0.25,
        cost_arm2: float = 0.25,
    ) -> Dict[int, np.ndarray]:
        """Calculate individual net utility for each treatment option."""
        delta_r1, delta_r2 = self.predict_incremental_revenues(X)
        n = len(X)
        return {
            0: np.zeros(n),
            1: delta_r1 * margin - cost_arm1,
            2: delta_r2 * margin - cost_arm2,
        }

    def allocate_personalized(
        self,
        X: pd.DataFrame,
        margin: float = 0.50,
        cost_arm1: float = 0.25,
        cost_arm2: float = 0.25,
        budget_fraction: float | None = 0.20,
    ) -> np.ndarray:
        """Assign treatment arm to each customer maximizing economic utility.
        
        If budget_fraction is provided, limits total contacted (arm != 0) to top fraction.
        """
        u = self.compute_utilities(X, margin, cost_arm1, cost_arm2)
        u1, u2 = u[1], u[2]
        n = len(X)

        best_active_arm = np.where(u1 >= u2, 1, 2)
        best_active_utility = np.maximum(u1, u2)

        if budget_fraction is None:
            # Unconstrained threshold targeting
            decisions = np.where(best_active_utility > 0, best_active_arm, 0)
        else:
            # Capacity-constrained targeting (contact top K customers)
            k = int(np.round(n * budget_fraction))
            decisions = np.zeros(n, dtype=int)
            top_k_indices = np.argsort(best_active_utility)[::-1][:k]
            decisions[top_k_indices] = best_active_arm[top_k_indices]

        return decisions


def evaluate_multiarm_policy_ipw(
    y_spend: np.ndarray,
    treatment: np.ndarray,
    policy_actions: np.ndarray,
    margin: float = 0.50,
    costs: Dict[int, float] | None = None,
    propensity: float = 1.0 / 3.0,
) -> Dict[str, float]:
    """Evaluate 3-arm policy on randomized held-out test data via IPW."""
    n = len(y_spend)
    costs = costs or {0: 0.0, 1: 0.25, 2: 0.25}

    match = (treatment == policy_actions)
    gross_spend_est = float(np.sum(match * y_spend / propensity) / n)

    # Average cost per customer in population
    total_cost = sum(np.sum(policy_actions == a) * costs.get(a, 0.0) for a in [0, 1, 2])
    mean_cost = float(total_cost / n)

    net_value = gross_spend_est * margin - mean_cost

    return {
        "gross_spend": gross_spend_est,
        "mean_cost": mean_cost,
        "net_value": net_value,
        "pct_treated": float(np.mean(policy_actions != 0) * 100),
        "pct_arm1": float(np.mean(policy_actions == 1) * 100),
        "pct_arm2": float(np.mean(policy_actions == 2) * 100),
    }


def plot_customer_treatment_allocation(
    X_test: pd.DataFrame,
    policy_actions: np.ndarray,
    output_path: str = "outputs/figures/figure12_customer_treatment_allocation.png",
) -> None:
    """Plot customer treatment allocation across historical category purchasing (Figure 12)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Overall action distribution
    arm_labels = ["No E-Mail (Control)", "Mens E-Mail", "Womens E-Mail"]
    counts = [np.sum(policy_actions == a) for a in [0, 1, 2]]
    colors = ["#7f7f7f", "#1f77b4", "#e377c2"]

    axes[0].bar(arm_labels, counts, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Customer Allocation by Assigned Action (Top 20% Budget)")
    axes[0].set_ylabel("Number of Customers Assigned")
    for i, count in enumerate(counts):
        axes[0].text(i, count + max(counts)*0.02, f"{count:,}\n({count/len(policy_actions):.1%})", ha="center")

    # 2. Cross-tabulation by prior purchase history (mens vs womens buyer)
    df_plot = pd.DataFrame({
        "action": policy_actions,
        "mens_buyer": X_test["mens"].values,
        "womens_buyer": X_test["womens"].values,
    })

    df_plot["category_history"] = "Neither"
    df_plot.loc[(df_plot["mens_buyer"] == 1) & (df_plot["womens_buyer"] == 0), "category_history"] = "Men's Only"
    df_plot.loc[(df_plot["mens_buyer"] == 0) & (df_plot["womens_buyer"] == 1), "category_history"] = "Women's Only"
    df_plot.loc[(df_plot["mens_buyer"] == 1) & (df_plot["womens_buyer"] == 1), "category_history"] = "Both Categories"

    # Filter to contacted customers only
    contacted = df_plot[df_plot["action"] != 0].copy()
    contacted["action_name"] = contacted["action"].map({1: "Mens E-Mail", 2: "Womens E-Mail"})

    ct = pd.crosstab(contacted["category_history"], contacted["action_name"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=axes[1], color=["#1f77b4", "#e377c2"], edgecolor="black", alpha=0.85)
    axes[1].set_title("Intervention Choice by Prior Purchase Affinity")
    axes[1].set_ylabel("Share of Contacted Customers (%)")
    axes[1].set_xlabel("Historical Customer Purchase Segment")
    axes[1].legend(title="Assigned Intervention", loc="upper right")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Multi-arm allocation plot saved to {out_file}")
