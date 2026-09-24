# Economics-Aware Causal Marketing Targeting
## Robust Policy Learning for Incremental Revenue Under Budget and Campaign-Cost Uncertainty

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Research Prototype](https://img.shields.io/badge/status-active--research-success.svg)]()
[![Target: JMA / ECRA / JBR](https://img.shields.io/badge/target-Q1%20Journals-purple.svg)]()

### Interdisciplinary Senior Thesis (Computer Science & Business Administration)
**Institution:** BRAC University

---

## 1. Project Overview

Traditional marketing analytics targets customers with high predicted response probabilities ($P(Y=1|X)$). However, customers who would purchase without an intervention ("Sure Things") consume marketing resources without generating incremental value. Uplift modeling addresses this by estimating the Conditional Average Treatment Effect (CATE):
$$\tau(X) = \mathbb{E}[Y(1) - Y(0) | X]$$

While algorithmic uplift estimation has advanced, an important managerial gap persists: **how to translate treatment effects into robust, budget-constrained policies when campaign economics (contribution margin $M$, contact cost $c$) are uncertain**.

This project provides an end-to-end, reproducible framework that bridges:
$$\boxed{\text{Prediction}} \longrightarrow \boxed{\text{Causal Uplift}} \longrightarrow \boxed{\text{Incremental Spend}} \longrightarrow \boxed{\text{Economic Utility}} \longrightarrow \boxed{\text{Budget-Constrained Optimization}} \longrightarrow \boxed{\text{Robust Minimax Regret Policy}}$$

---

## 2. Key Research Questions

- **RQ1 (Prediction vs. Causality):** How different are customer rankings produced by conventional response models and causal uplift models?
- **RQ2 (Economic Value):** Does targeting customers according to incremental response generate greater incremental revenue than targeting by response probability alone?
- **RQ3 (Budget Constraints):** How does the optimal targeting policy change as campaign capacity and treatment cost change?
- **RQ4 (Economic Uncertainty):** How sensitive is the optimal policy to assumptions about contribution margin and intervention cost?
- **RQ5 (Robustness):** Can a minimax-regret policy maintain acceptable performance across uncertain cost scenarios without sharp performance degradation?
- **RQ6 (Treatment Personalization):** In multi-arm settings (Men's vs Women's email vs Control), does individualized treatment selection create additional economic value?
- **RQ7 (Generalization):** Do the core causal targeting insights replicate on large-scale benchmarks (Criteo)?

---

## 3. Repository Layout

```text
├── configs/                   # Experiment and scenario grid configurations
├── data/
│   ├── raw/                   # Raw immutable datasets (Hillstrom, Criteo)
│   └── processed/             # Leakage-audited train/validation/test splits
├── docs/
│   ├── PROPOSAL.md            # Complete theoretical thesis proposal
│   └── feature_audit.md       # Pre-treatment covariate balance & audit report
├── notebooks/                 # Exploratory data analysis & interactive walkthroughs
├── outputs/
│   ├── figures/               # High-res publication charts (PDF, PNG, SVG)
│   ├── tables/                # LaTeX and Markdown tables
│   └── logs/                  # Training logs & experiment tracking
├── src/
│   ├── data/                  # Ingestion, validation, and leakage auditing
│   ├── models/                # Response baselines & causal meta-learners (S/T/X/DR)
│   ├── economics/             # Economic utility, break-even cost, and scenarios
│   ├── policy/                # Top-K ranking, knapsack budget solver, robust minimax
│   ├── evaluation/            # Off-policy evaluation (AIPW), AUUC, Qini, bootstrap CI
│   └── visualization/         # Uplift curves, budget frontiers, stability heatmaps
└── pyproject.toml             # Project dependency specification
```

---

## 4. Setup & Quickstart

```bash
# Clone the repository
git clone <repo-url>
cd "Economics-Aware Causal Marketing Targeting"

# Set up virtual environment using uv or standard venv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

---

## 5. License & Citation
Academic research prototype developed for BRAC University CSE/BBA thesis research.
