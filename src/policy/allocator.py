"""Marketing Policy Allocation and Knapsack Optimization.

Implements targeting policies:
1. Top-K Ranking Policy: Select top fraction alpha or K customers ranked by a priority score.
2. Budget-Constrained Knapsack Solver: Solves resource-constrained targeting
   max sum_i U_i(a_i)  s.t.  sum_i c_{a_i} <= B
"""

from typing import Union, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pulp


def policy_top_k(scores: np.ndarray, budget_count: int) -> np.ndarray:
    """Generate binary targeting decisions by selecting top `budget_count` scores.
    
    Returns binary array pi in {0, 1}^N.
    """
    n = len(scores)
    k = min(max(0, budget_count), n)
    decisions = np.zeros(n, dtype=int)
    if k == 0:
        return decisions
    # Find indices of top k scores
    top_indices = np.argsort(scores)[::-1][:k]
    decisions[top_indices] = 1
    return decisions


def policy_top_fraction(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Select top fraction alpha in [0, 1] of customers ranked by score."""
    assert 0.0 <= fraction <= 1.0, f"Fraction must be in [0, 1], got {fraction}"
    budget_count = int(np.round(len(scores) * fraction))
    return policy_top_k(scores, budget_count)


def solve_multiarm_knapsack(
    utilities: Dict[int, np.ndarray],
    costs: Dict[int, float],
    total_budget: float,
    time_limit_sec: int = 15,
) -> np.ndarray:
    """Solve multiple-choice knapsack allocation for multi-arm campaign targeting.
    
    Args:
        utilities: Dict mapping action a in {0, 1, ...} to utility vector U_i(a).
        costs: Dict mapping action a to contact cost c_a (c_0 = 0).
        total_budget: Total spending budget B.
        time_limit_sec: Solver time limit.

    Returns:
        Action assignment array pi in {0, 1, ...}^N.
    """
    n = len(next(iter(utilities.values())))
    arms = [a for a in utilities.keys() if a != 0]

    # Fast heuristic greedy sort if 1 treatment arm
    if len(arms) == 1:
        arm = arms[0]
        c_a = costs.get(arm, 0.0)
        u_arr = utilities[arm]
        if c_a <= 0:
            return (u_arr > 0).astype(int) * arm
        k = int(np.floor(total_budget / c_a))
        top_k_idx = np.argsort(u_arr)[::-1][:k]
        assignment = np.zeros(n, dtype=int)
        # Only assign if utility is positive or forced by budget
        assignment[top_k_idx] = arm
        return assignment

    # Multi-arm Integer Linear Program via PuLP
    prob = pulp.LpProblem("Marketing_Campaign_Knapsack", pulp.LpMaximize)

    # Decision variables x_{i, a}
    x_vars = {}
    for i in range(n):
        for a in arms:
            x_vars[(i, a)] = pulp.LpVariable(f"x_{i}_{a}", cat="Binary")

    # Objective: maximize total utility
    prob += pulp.lpSum(utilities[a][i] * x_vars[(i, a)] for i in range(n) for a in arms)

    # Budget constraint: sum_i sum_a c_a * x_{i, a} <= B
    prob += pulp.lpSum(costs[a] * x_vars[(i, a)] for i in range(n) for a in arms) <= total_budget

    # At most one treatment per customer: sum_a x_{i, a} <= 1
    for i in range(n):
        prob += pulp.lpSum(x_vars[(i, a)] for a in arms) <= 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    prob.solve(solver)

    assignment = np.zeros(n, dtype=int)
    for i in range(n):
        for a in arms:
            if pulp.value(x_vars[(i, a)]) == 1:
                assignment[i] = a
                break

    return assignment
