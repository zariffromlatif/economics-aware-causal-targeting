### Table 7: Multi-Arm Policy Evaluation and Personalization Advantage

|    | Policy                                | Targeted (%)   | Mens (%)   | Womens (%)   |   Gross Spend ($) |   Mean Cost ($) |   Net Policy Value ($) |
|----|---------------------------------------|----------------|------------|--------------|-------------------|-----------------|------------------------|
|  0 | Baseline (No Contact)                 | 0.0%           | 0.0%       | 0.0%         |            0.5661 |          0.0000 |                 0.2831 |
|  1 | Uniform Mens E-Mail (Global Best)     | 20.0%          | 20.0%      | 0.0%         |            0.8787 |          0.0500 |                 0.3893 |
|  2 | Uniform Womens E-Mail                 | 20.0%          | 0.0%       | 20.0%        |            0.7369 |          0.0500 |                 0.3185 |
|  3 | Personalized Causal Policy (Proposed) | 20.0%          | 11.5%      | 8.5%         |            0.6802 |          0.0500 |                 0.2901 |

**Hypothesis H7 Test (Personalized vs Uniform Mens)**:
- Point Estimate Difference: $-0.0992
- 95% Bootstrap CI: [$-0.2563, $+0.0565]
- Empirical p-value: 0.2320
