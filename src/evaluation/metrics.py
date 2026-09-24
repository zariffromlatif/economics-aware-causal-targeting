"""Uplift Evaluation Metrics and Off-Policy Value Estimators.

Implements:
1. Qini Curve and Area Under the Qini Curve (AUQC / Qini score).
2. Area Under the Uplift Curve (AUUC).
3. Held-out Off-Policy Value Estimation using Inverse Propensity Weighting (IPW)
   and Doubly Robust (AIPW) scoring for randomized trial data.
4. Net Economic Policy Value accounting for contribution margins and contact costs.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd


def compute_uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative uplift and Qini curves.
    
    Args:
        y: Observed outcomes (0/1 conversion or continuous spend).
        treatment: Observed treatment indicator (0 or 1).
        scores: Predicted priority score used to rank customers descendingly.
        n_bins: Number of evaluation cutoffs.

    Returns:
        fractions: Array of targeted population fractions in [0, 1].
        qini_curve: Cumulative incremental gain.
        random_curve: Expected gain under uniform random targeting.
    """
    order = np.argsort(scores)[::-1]
    y_sort = y[order]
    t_sort = treatment[order]
    n = len(y)

    cum_n = np.arange(1, n + 1)
    cum_t = np.cumsum(t_sort)
    cum_c = cum_n - cum_t

    cum_y_t = np.cumsum(y_sort * t_sort)
    cum_y_c = np.cumsum(y_sort * (1 - t_sort))

    # Avoid division by zero
    scale = np.where(cum_c > 0, cum_t / np.maximum(cum_c, 1), 0.0)
    # Qini cumulative gain: Y_t - Y_c * (N_t / N_c)
    qini_vals = cum_y_t - (cum_y_c * scale)

    # Subsample to n_bins points for plotting and integration
    bin_indices = np.linspace(0, n - 1, n_bins, dtype=int)
    fractions = (bin_indices + 1) / n
    qini_sampled = qini_vals[bin_indices]

    # Prepend 0
    fractions = np.insert(fractions, 0, 0.0)
    qini_sampled = np.insert(qini_sampled, 0, 0.0)

    # Theoretical random baseline connects (0, 0) to (1, Qini(1))
    total_gain = qini_sampled[-1]
    random_sampled = fractions * total_gain

    return fractions, qini_sampled, random_sampled


def compute_qini_score(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute Area Under the Qini Curve minus random baseline area."""
    fractions, qini, random_b = compute_uplift_curve(y, treatment, scores)
    area_qini = np.trapezoid(qini, fractions)
    area_random = np.trapezoid(random_b, fractions)
    return float(area_qini - area_random)


def estimate_policy_value_ipw(
    y: np.ndarray,
    treatment: np.ndarray,
    policy_actions: np.ndarray,
    propensity: float = 0.5,
) -> float:
    """Estimate expected outcome of policy pi using Inverse Propensity Weighting (IPW).
    
    V(pi) = (1 / N) * sum_i [ 1(T_i == pi(X_i)) * Y_i / P(T_i == pi(X_i)) ]
    In Hillstrom binary comparison, P(T=1) = P(T=0) = propensity = 0.5.
    """
    n = len(y)
    match = (treatment == policy_actions)
    prob_match = np.where(policy_actions == 1, propensity, 1.0 - propensity)
    return float(np.sum(match * y / prob_match) / n)


def estimate_net_economic_value(
    spend: np.ndarray,
    treatment: np.ndarray,
    policy_actions: np.ndarray,
    margin: float = 0.50,
    cost: float = 0.25,
    propensity: float = 0.5,
) -> float:
    """Estimate net economic policy value (gross contribution margin minus contact costs).
    
    Net Value = V_Spend(pi) * Margin - Mean_Cost(pi)
    """
    gross_spend_value = estimate_policy_value_ipw(spend, treatment, policy_actions, propensity)
    mean_cost = float(np.mean(policy_actions * cost))
    return float(gross_spend_value * margin - mean_cost)
