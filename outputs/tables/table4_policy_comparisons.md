### Table 4: Policy Value and Bootstrap Inference

|    | Policy                    | Targeted (%)   |   Expected Conv Rate (%) |   Expected Gross Spend ($) |   Net Economic Value ($) |
|----|---------------------------|----------------|--------------------------|----------------------------|--------------------------|
|  0 | Random Policy             | 20.0%          |                   0.6805 |                     0.6812 |                   0.2906 |
|  1 | Response Policy (XGB)     | 20.0%          |                   0.5866 |                     0.7037 |                   0.3019 |
|  2 | Uplift Policy (T-Learner) | 20.0%          |                   0.6570 |                     0.8974 |                   0.3987 |
|  3 | Revenue Uplift Policy     | 20.0%          |                   0.6336 |                     0.8797 |                   0.3898 |
|  4 | Profit-Aware Policy       | 20.0%          |                   0.6336 |                     0.8797 |                   0.3898 |

**Bootstrap Test (Revenue Uplift vs Response)**: Delta V = $+0.0880 (95% CI: [$-0.0608, $+0.2370], p = 0.2480)
