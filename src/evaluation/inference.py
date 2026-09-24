"""Statistical Inference and Paired Bootstrap Hypothesis Testing.

Implements non-parametric paired bootstrap procedures to compute:
1. 95% Confidence Intervals for policy values V(pi).
2. Paired policy difference tests Delta V = V(pi_A) - V(pi_B).
3. Empirical p-values testing H0: Delta V <= 0 vs H1: Delta V > 0.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import estimate_net_economic_value


def paired_bootstrap_policy_comparison(
    spend: np.ndarray,
    treatment: np.ndarray,
    policy_a: np.ndarray,
    policy_b: np.ndarray,
    margin: float = 0.50,
    cost: float = 0.25,
    propensity: float = 0.5,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform paired bootstrap test between policy A and policy B."""
    rng = np.random.default_rng(seed)
    n = len(spend)

    val_a_point = estimate_net_economic_value(spend, treatment, policy_a, margin, cost, propensity)
    val_b_point = estimate_net_economic_value(spend, treatment, policy_b, margin, cost, propensity)
    diff_point = val_a_point - val_b_point

    boot_a = np.empty(n_bootstraps)
    boot_b = np.empty(n_bootstraps)
    boot_diff = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        v_a = estimate_net_economic_value(
            spend[boot_idx], treatment[boot_idx], policy_a[boot_idx],
            margin, cost, propensity
        )
        v_b = estimate_net_economic_value(
            spend[boot_idx], treatment[boot_idx], policy_b[boot_idx],
            margin, cost, propensity
        )
        boot_a[b] = v_a
        boot_b[b] = v_b
        boot_diff[b] = v_a - v_b

    # Percentile confidence intervals
    ci_low_pct = 100 * (alpha / 2.0)
    ci_high_pct = 100 * (1.0 - alpha / 2.0)

    ci_diff = (np.percentile(boot_diff, ci_low_pct), np.percentile(boot_diff, ci_high_pct))
    ci_a = (np.percentile(boot_a, ci_low_pct), np.percentile(boot_a, ci_high_pct))
    ci_b = (np.percentile(boot_b, ci_low_pct), np.percentile(boot_b, ci_high_pct))

    # Two-sided empirical p-value for H0: diff == 0
    p_val = 2.0 * min(np.mean(boot_diff <= 0), np.mean(boot_diff >= 0))
    p_val = min(1.0, max(1.0 / n_bootstraps, p_val))

    return {
        "point_diff": diff_point,
        "ci_diff": ci_diff,
        "val_a": val_a_point,
        "ci_a": ci_a,
        "val_b": val_b_point,
        "ci_b": ci_b,
        "p_value": p_val,
        "significant": (ci_diff[0] > 0 or ci_diff[1] < 0),
    }
