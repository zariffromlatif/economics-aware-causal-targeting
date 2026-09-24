"""Economic Utility and Net Campaign Value Calculation.

Implements the economic integration layer:
U_i(a) = Delta R_{i,a} * M - c_a

Where:
- Delta R_{i,a}: Predicted incremental revenue for customer i from action a
- M: Contribution margin rate (e.g. 0.20 to 0.80)
- c_a: Direct unit campaign cost of contact action a (c_0 = 0)
"""

from typing import Union
import numpy as np
import pandas as pd


def compute_incremental_utility(
    delta_revenue: Union[np.ndarray, pd.Series],
    margin: float = 0.50,
    cost: float = 0.25,
) -> np.ndarray:
    """Compute individual incremental economic utility for binary intervention.
    
    U_i = Delta R_i * M - c
    """
    assert 0.0 < margin <= 1.0, f"Margin must be in (0, 1], got {margin}"
    assert cost >= 0.0, f"Cost must be non-negative, got {cost}"
    delta_r = np.asarray(delta_revenue, dtype=float)
    return delta_r * margin - cost


def compute_multiarm_utility(
    delta_revenue_arms: dict[int, np.ndarray],
    cost_arms: dict[int, float],
    margin: float = 0.50,
) -> dict[int, np.ndarray]:
    """Compute individual incremental economic utility for multi-arm treatments.
    
    Control arm (0) has identically 0 incremental revenue and 0 cost: U_i(0) = 0.
    """
    utilities = {0: np.zeros_like(next(iter(delta_revenue_arms.values())))}
    for arm, delta_r in delta_revenue_arms.items():
        if arm == 0:
            continue
        c_a = cost_arms.get(arm, 0.0)
        utilities[arm] = np.asarray(delta_r, dtype=float) * margin - c_a
    return utilities
