# Economics-Aware Causal Marketing Targeting  
## Robust Policy Learning for Incremental Revenue Under Budget and Campaign-Cost Uncertainty

### Proposed Interdisciplinary Research Project for Computer Science and Business Administration

---

## 1. Proposed Research Title

### Primary title
**Economics-Aware Causal Marketing Targeting: Robust Policy Learning for Incremental Revenue Under Budget and Campaign-Cost Uncertainty**

### Alternative title for journal submission
**From Prediction to Intervention: Robust Uplift-Based Marketing Policies for Incremental Revenue and Budget-Constrained Customer Targeting**

### Short working title
**Robust Profit-Aware Uplift Marketing**

---

## 2. Abstract
Traditional marketing analytics frequently predicts which customers are likely to purchase, respond, or engage, but high response probability does not necessarily imply that a marketing intervention caused the observed outcome. Customers who would purchase without an intervention may consume campaign resources without generating incremental value. Uplift modeling addresses this problem by estimating heterogeneous treatment effects and identifying customers whose behavior is causally influenced by an intervention.

However, an important decision gap remains between estimating treatment effects and selecting a commercially defensible intervention policy. Existing research has separately studied uplift modeling, revenue/profit uplift, budget-constrained incentive allocation, and heterogeneous treatment effects, but these strands are not always evaluated together under a common economic decision framework. Recent work has also shown that evaluation of uplift models is sensitive to metric choice and statistical variance, strengthening the need for rigorous held-out policy evaluation.

This study proposes an empirical framework that integrates causal uplift estimation, incremental-revenue modeling, campaign-cost sensitivity, budget-constrained policy optimization, and robustness analysis. The primary empirical setting will use the Hillstrom MineThatData email experiment, containing 64,000 customers randomly assigned to men's email, women's email, or no-email control groups, with post-campaign visit, conversion, and spending outcomes.

The study will compare conventional response models with uplift and causal estimators, evaluate their policies using held-out randomized data, and determine how optimal targeting changes as campaign cost, contribution margin, and campaign capacity vary. An optional external validation will use the unbiased Criteo Uplift Prediction Dataset, containing approximately 14 million observations from randomized incrementality experiments.

The study does not claim to invent a new causal estimator. Instead, its contribution is an integrated, reproducible decision framework for determining **who should be treated, with which intervention, under what economic conditions, and with what degree of policy robustness**.

---

## 3. Background and Research Motivation
Marketing organizations have traditionally used predictive models to rank customers according to their probability of responding:
\[
X \rightarrow P(Y=1|X) \rightarrow \text{Target highest-probability customers}
\]

This creates a fundamental problem:
- **Customer A:** will purchase whether contacted or not.
- **Customer B:** will purchase only because of the marketing intervention.

A conventional response model assigns both high response probability, but only Customer B generates incremental value.

Uplift modeling estimates the conditional average treatment effect (CATE):
\[
\tau(X) = \mathbb{E}[Y(1) - Y(0) | X]
\]
Shifting the objective from *"Who is likely to respond?"* to *"Who responds because of the intervention?"*.

Furthermore, even with positive uplift, contacting a customer is unprofitable if:
\[
\text{incremental value} < \text{campaign cost}
\]
Hence, marketing targeting must be formulated as a **policy-learning and resource-allocation problem**, not solely a prediction problem.

---

## 4. Problem Formulation & Research Questions

### Central Research Question
**How can causal uplift estimation be converted into a robust, budget-constrained marketing policy that maximizes incremental economic value under uncertain campaign economics?**

### Secondary Research Questions
- **RQ1 (Prediction vs. Causality):** How different are customer rankings produced by conventional response models and causal uplift models?
- **RQ2 (Economic Value):** Does targeting customers according to incremental response generate greater incremental revenue than targeting customers according to response probability alone?
- **RQ3 (Budget Constraints):** How does the optimal targeting policy change as campaign capacity and treatment cost change?
- **RQ4 (Economic Uncertainty):** How sensitive is the optimal policy to assumptions about contribution margin and intervention cost?
- **RQ5 (Robustness):** Can a policy that is robust across multiple plausible economic scenarios outperform policies optimized for a single assumed scenario in worst-case or regret-based performance?
- **RQ6 (Treatment Personalization):** When multiple campaign treatments are available, does customer-specific treatment selection create additional economic value compared with applying the same campaign to all targeted customers?
- **RQ7 (Generalization):** Do the main causal-targeting findings remain directionally consistent on a second randomized marketing benchmark?

---

## 5. Economic & Robust Policy Decision Framework

### Economic Utility
For customer $i$ and treatment action $a$:
\[
U_i(a) = \widehat{\Delta R}_{i,a} \cdot M - c_a
\]
where:
- $\widehat{\Delta R}_{i,a}$: Estimated incremental revenue from action $a$ relative to control ($a=0$).
- $M$: Product / category contribution margin rate ($M \in [0.20, 0.80]$).
- $c_a$: Unit cost of delivering campaign contact $a$.

### Break-Even Contact Cost
\[
c^*_{i,a} = \widehat{\Delta R}_{i,a} \cdot M
\]

### Budget-Constrained Policy Allocation
\[
\max_{a_1, \dots, a_N} \sum_{i=1}^N U_i(a_i) \quad \text{s.t.} \quad \sum_{i=1}^N c_{a_i} \le B
\]

### Minimax Regret Policy
For scenarios $\theta = (M, c) \in \Theta$:
\[
\pi_R = \arg\min_{\pi} \max_{\theta \in \Theta} \left[ V(\pi^*_\theta; \theta) - V(\pi; \theta) \right]
\]

### Policy Stability (Jaccard Index)
\[
PS(s_1, s_2) = \frac{|T_{s_1} \cap T_{s_2}|}{|T_{s_1} \cup T_{s_2}|}
\]

---

## 6. Datasets

1. **Hillstrom MineThatData (Primary Benchmark)**:
   - 64,000 customers randomly assigned to:
     - Men's email ($N \approx 21,307$)
     - Women's email ($N \approx 21,364$)
     - Control / No email ($N \approx 21,329$)
   - Pre-treatment covariates: `recency`, `history`, `history_segment`, `mens`, `womens`, `zip_code`, `newbie`, `channel`.
   - Post-treatment outcomes: `visit`, `conversion`, `spend`.

2. **Criteo Uplift Prediction Dataset (Secondary Benchmark)**:
   - ~13.98M randomized observations from ad incrementality experiments.
   - Used for large-scale external validation of conversion uplift policies.

---

## 7. Model Taxonomy & Policy Baselines

| ID | Model / Policy | Description |
|---|---|---|
| A1 | Logistic Regression | Baseline response probability $P(Y=1|X)$ |
| A2 | XGBoost Classifier | Nonlinear response probability $P(Y=1|X)$ |
| B1 | S-Learner | Single model with treatment indicator feature |
| B2 | T-Learner | Twin models $\hat{\mu}_1(X) - \hat{\mu}_0(X)$ |
| B3 | X-Learner | Imputed counterfactual residual regression |
| B4 | Causal Forest / GRF | Nonparametric heterogeneous treatment effect estimation |
| C  | Doubly Robust Learner | Cross-fitted orthogonal score estimation (AIPW) |
| Pol-Rand | Random Targeting | Uniform random contact allocation |
| Pol-Resp | Response Policy | Rank by predicted conversion probability |
| Pol-ATE  | Global Best Arm | Assign single best treatment unconditionally |
| Pol-Uplift | CATE Uplift Policy | Rank by conversion CATE $\hat{\tau}^C(X)$ |
| Pol-Rev  | Revenue Uplift Policy | Rank by incremental revenue $\hat{\tau}^R(X)$ |
| Pol-Econ | Profit-Aware Policy | Rank by net economic utility $U_i(a)$ |
| Pol-Bgt  | Knapsack Policy | Optimal multiple-choice allocation under budget $B$ |
| Pol-Rob  | Minimax Regret Policy | Robust allocation minimizing worst-case regret |

---

## 8. Target Publication Outlets
1. **Primary Target**: *Journal of Marketing Analytics* (Springer, Impact Factor: 5.0, Scopus Q1)
2. **Secondary Target**: *Electronic Commerce Research and Applications* (Elsevier, Scopus Q1)
3. **Stretch Target**: *Journal of Business Research* (Elsevier, Scopus Q1)
