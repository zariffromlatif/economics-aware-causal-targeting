"""Economic Scenario Grid Definitions.

Defines plausible commercial scenarios Theta = M x C to evaluate policy sensitivity,
stability, and minimax regret without fabricating an arbitrary single profit number.
"""

from dataclasses import dataclass
from typing import List, Tuple
import itertools


@dataclass(frozen=True)
class EconomicScenario:
    """Represents a distinct commercial cost-margin scenario theta = (M, c)."""
    scenario_id: str
    margin: float
    cost: float
    description: str = ""


DEFAULT_MARGINS = [0.20, 0.40, 0.60, 0.80]
DEFAULT_COSTS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]


def build_scenario_grid(
    margins: List[float] | None = None,
    costs: List[float] | None = None,
) -> List[EconomicScenario]:
    """Generate Cartesian product of margin and cost parameters."""
    m_list = margins or DEFAULT_MARGINS
    c_list = costs or DEFAULT_COSTS

    scenarios = []
    for m, c in itertools.product(m_list, c_list):
        s_id = f"M{int(m*100)}_C{int(c*100):03d}"
        desc = f"Margin: {m:.0%}, Cost: ${c:.2f}"
        scenarios.append(EconomicScenario(scenario_id=s_id, margin=m, cost=c, description=desc))

    return scenarios
