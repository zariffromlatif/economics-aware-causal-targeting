### Table 7: Multi-Arm Policy Evaluation and Personalization Advantage

|    | Policy                                | Targeted (%)   | Mens (%)   | Womens (%)   |   Gross Spend ($) |   Mean Cost ($) |   Net Policy Value ($) |
|----|---------------------------------------|----------------|------------|--------------|-------------------|-----------------|------------------------|
|  0 | Baseline (No Contact)                 | 0.0%           | 0.0%       | 0.0%         |            0.5661 |          0.0000 |                 0.2831 |
|  1 | Uniform Mens E-Mail (Global Best)     | 20.0%          | 20.0%      | 0.0%         |            0.8150 |          0.0500 |                 0.3575 |
|  2 | Uniform Womens E-Mail                 | 20.0%          | 0.0%       | 20.0%        |            0.7252 |          0.0500 |                 0.3126 |
|  3 | Personalized Causal Policy (Proposed) | 20.0%          | 11.8%      | 8.2%         |            0.6230 |          0.0500 |                 0.2615 |

**Hypothesis H7 Test (Personalized vs Uniform Mens)**:
- Point Estimate Difference: $-0.0960
- 95% Bootstrap CI: [$-0.2886, $+0.0635]
- Empirical p-value: 0.2720
