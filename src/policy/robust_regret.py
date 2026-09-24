"""Robust Policy Learning and Minimax Regret Analysis.

Evaluates marketing policy performance under economic uncertainty:
1. Minimax Regret:
   R(pi; theta) = V(pi*_theta; theta) - V(pi; theta)
   pi_R = argmin_pi max_{theta in Theta} R(pi; theta)
2. Policy Stability:
   Jaccard similarity of selected cohorts across commercial scenarios s_1 and s_2:
   PS(s_1, s_2) = |T_{s_1} cap T_{s_2}| / |T_{s_1} cup T_{s_2}|
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.economics.scenarios import EconomicScenario


def compute_jaccard_similarity(decision_a: np.ndarray, decision_b: np.ndarray) -> float:
    """Calculate Jaccard targeting stability between two binary targeting vectors."""
    idx_a = set(np.where(decision_a > 0)[0])
    idx_b = set(np.where(decision_b > 0)[0])

    if not idx_a and not idx_b:
        return 1.0  # Both select nobody
    union_len = len(idx_a.union(idx_b))
    if union_len == 0:
        return 1.0
    return len(idx_a.intersection(idx_b)) / union_len


def compute_policy_stability_matrix(
    decisions_by_scenario: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity matrix across economic scenarios."""
    scenarios = list(decisions_by_scenario.keys())
    m = len(scenarios)
    matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            matrix[i, j] = compute_jaccard_similarity(
                decisions_by_scenario[scenarios[i]],
                decisions_by_scenario[scenarios[j]],
            )

    return pd.DataFrame(matrix, index=scenarios, columns=scenarios)


def compute_regret_table(
    policy_values: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compute regret matrix and identify the Minimax Regret policy.
    
    Args:
        policy_values: Nested dict mapping scenario_id -> {policy_name: value}.

    Returns:
        values_df: Table of policy values across scenarios.
        regret_df: Table of regrets across scenarios, with Max Regret column.
        best_robust_policy: Name of the policy that minimizes maximum regret.
    """
    values_df = pd.DataFrame(policy_values).T  # rows: scenarios, cols: policies

    # For each scenario (row), find optimal policy value V*(theta)
    optimal_values = values_df.max(axis=1)

    # Regret = V*(theta) - V(pi; theta)
    regret_df = values_df.apply(lambda col: optimal_values - col, axis=0)

    # Max regret per policy across all scenarios
    max_regrets = regret_df.max(axis=0)
    best_robust_policy = str(max_regrets.idxmin())

    # Add summary row to regret table
    regret_summary = regret_df.copy()
    regret_summary.loc["MAX_REGRET"] = max_regrets
    regret_summary.loc["MEAN_REGRET"] = regret_df.mean(axis=0)

    return values_df, regret_summary, best_robust_policy
