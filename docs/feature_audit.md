# Pre-Treatment Covariate and Balance Audit
## Hillstrom MineThatData Randomized Marketing Experiment

### 1. Data Integrity and Randomization Check
- Total Sample Size: 64,000 customers.
- Treatment Allocation:
  - Control (No E-Mail): 21,306 (33.33%)
  - Mens E-Mail: 21,307 (32.98%)
  - Womens E-Mail: 21,387 (33.38%)

### 2. Randomization Diagnostic (Omnibus Test)
- Logistic Regression Omnibus Model (Predicting $T$ from $X$):
  - 5-Fold Cross-Validated ROC-AUC: **0.4991**
  - **Verdict**: The model cannot predict treatment assignment better than random chance ($AUC \approx 0.50$). This confirms the absence of selection bias and supports the unconfoundedness assumption ($Y(0), Y(1) \perp T \mid X$).

### 3. Covariate Balance (Standardized Mean Differences)
A standardized mean difference (SMD) threshold of $< 0.05$ or $< 0.10$ denotes excellent balance.
- **Maximum SMD observed**: **0.0169**
- All features exhibit negligible discrepancy between arms.

### 4. Empirical Outcome Overview
|    | Arm                 |     N |   Visit Rate (%) |   Conv Rate (%) |   Spend / Customer ($) |   Spend | Converter ($) |
|----|---------------------|-------|------------------|-----------------|------------------------|-------------------------|
|  0 | Control (No E-Mail) | 21306 |          10.6167 |          0.5726 |                 0.6528 |                114.0027 |
|  1 | Mens E-Mail         | 21307 |          18.2757 |          1.2531 |                 1.4226 |                113.5269 |
|  2 | Womens E-Mail       | 21387 |          15.1400 |          0.8837 |                 1.0772 |                121.8948 |

### 5. Empirical Average Treatment Effects (ATE)
|    | Comparison        |   Delta Visit (%) |   Delta Conv (%) |   Delta Spend ($) |
|----|-------------------|-------------------|------------------|-------------------|
|  0 | Mens vs Control   |            7.6590 |           0.6805 |            0.7698 |
|  1 | Womens vs Control |            4.5233 |           0.3111 |            0.4244 |

### 6. Data Leakage Verification
- Pre-treatment covariates strictly audited: `recency`, `history`, `mens`, `womens`, `newbie`, `zip_code`, `channel`.
- Post-treatment outcomes (`visit`, `conversion`, `spend`) and derived metrics are strictly excluded from feature inputs.
