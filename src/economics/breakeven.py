"""Break-Even Campaign Cost Analysis.

Derives the maximum contact cost c^*_{i,a} that can be incurred for each customer
before the marketing contact becomes economically negative:
c^*_{i,a} = Delta R_{i,a} * M
"""

from typing import Union, Dict, Any
import numpy as np
import pandas as pd


def compute_breakeven_costs(
    delta_revenue: Union[np.ndarray, pd.Series],
    margin: float = 0.50,
) -> np.ndarray:
    """Calculate individual customer break-even contact cost.
    
    c*_i = Delta R_i * M
    """
    assert 0.0 < margin <= 1.0, f"Margin must be in (0, 1], got {margin}"
    delta_r = np.asarray(delta_revenue, dtype=float)
    return delta_r * margin


def summarize_breakeven_distribution(
    breakeven_costs: np.ndarray,
    cost_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Summarize percentiles and viable targeting fractions under various unit costs."""
    be = np.asarray(breakeven_costs, dtype=float)
    thresholds = cost_thresholds or [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

    records = []
    total_n = len(be)
    for c in thresholds:
        viable_count = int(np.sum(be >= c))
        records.append({
            "Campaign Cost ($)": c,
            "Viable Customers": viable_count,
            "Viable Share (%)": (viable_count / total_n) * 100.0,
            "Expected Profit if Viable Treated ($)": float(np.sum(be[be >= c] - c)),
        })

    return pd.DataFrame(records)
