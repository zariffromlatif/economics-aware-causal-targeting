"""Visualization for Qini and Uplift Curves (Figures 3 and 4).

Generates publication-quality charts comparing:
- Response prediction ranking vs Causal Uplift (T-Learner, X-Learner, DR-Learner).
- Cumulative conversion gain and incremental revenue gain.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.metrics import compute_uplift_curve, compute_qini_score

# Set aesthetic defaults suitable for academic publications
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
})


def plot_uplift_curves(
    y: np.ndarray,
    treatment: np.ndarray,
    models_scores: Dict[str, np.ndarray],
    outcome_name: str = "Conversion",
    output_path: str = "outputs/figures/figure3_uplift_curves.png",
) -> None:
    """Plot cumulative uplift/Qini curves for multiple models against random targeting."""
    plt.figure(figsize=(9, 6), dpi=300)

    # Plot random baseline once
    first_scores = next(iter(models_scores.values()))
    fractions, _, random_curve = compute_uplift_curve(y, treatment, first_scores)
    plt.plot(fractions * 100, random_curve, "--", color="gray", label="Random Policy (Baseline)", alpha=0.8)

    palette = sns.color_palette("tab10", len(models_scores))

    for idx, (model_name, scores) in enumerate(models_scores.items()):
        fracs, qini, _ = compute_uplift_curve(y, treatment, scores)
        q_score = compute_qini_score(y, treatment, scores)
        label_txt = f"{model_name} (Qini: {q_score:+.2f})"
        plt.plot(fracs * 100, qini, label=label_txt, color=palette[idx])

    plt.title(f"Cumulative Incremental {outcome_name} Uplift Curve (Held-Out Test Set)")
    plt.xlabel("Percentage of Customers Contacted (%)")
    plt.ylabel(f"Cumulative Incremental {outcome_name}")
    plt.xlim(0, 100)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Uplift curve saved to {out_file}")
