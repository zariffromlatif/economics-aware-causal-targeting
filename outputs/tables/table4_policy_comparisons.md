### Table 4: Policy Value and Bootstrap Inference

|    | Policy                    | Targeted (%)   |   Expected Conv Rate (%) |   Expected Gross Spend ($) |   Net Economic Value ($) |
|----|---------------------------|----------------|--------------------------|----------------------------|--------------------------|
|  0 | Random Policy             | 20.0%          |                   0.6805 |                     0.6812 |                   0.2906 |
|  1 | Response Policy (XGB)     | 20.0%          |                   0.5163 |                     0.6817 |                   0.2908 |
|  2 | Uplift Policy (T-Learner) | 20.0%          |                   0.6336 |                     0.8869 |                   0.3934 |
|  3 | Revenue Uplift Policy     | 20.0%          |                   0.5866 |                     0.8160 |                   0.3580 |
|  4 | Profit-Aware Policy       | 20.0%          |                   0.5866 |                     0.8160 |                   0.3580 |

**Bootstrap Test (Revenue Uplift vs Response)**: Delta V = $+0.0672 (95% CI: [$-0.0706, $+0.1928], p = 0.3240)
