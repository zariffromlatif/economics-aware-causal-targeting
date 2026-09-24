"""Complete End-to-End Empirical Research Pipeline.

Executes:
1. Data loading and leakage-free splitting.
2. Baselines (Logistic, XGBoost) and Uplift Meta-Learners (S, T, X, DR).
3. Incremental revenue estimation (Hurdle, Direct T-Learner).
4. Rank correlation and top-K overlap (Hypothesis H1).
5. Uplift evaluation (AUUC, Qini) and curve generation (Figures 3 & 4).
6. Held-out off-policy evaluation with paired bootstrap inference (Hypothesis H2 & H3).
7. Budget, cost, and margin sensitivity analysis (Figures 5 & 6).
8. Minimax regret and Jaccard policy stability (Figures 9 & 10).
9. Automated generation of publication tables (Tables 3-7).
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from src.data.loader import prepare_hillstrom_splits
from src.models.response_baselines import ResponseBaseline
from src.models.meta_learners import SLearner, TLearner, XLearner
from src.models.doubly_robust import DoublyRobustLearner
from src.models.revenue_models import HurdleRevenueUplift, DirectRevenueUplift
from src.economics.utility import compute_incremental_utility
from src.economics.breakeven import compute_breakeven_costs, summarize_breakeven_distribution
from src.economics.scenarios import build_scenario_grid
from src.policy.allocator import policy_top_fraction
from src.policy.robust_regret import (
    compute_policy_stability_matrix,
    compute_regret_table,
)
from src.evaluation.metrics import (
    compute_qini_score,
    estimate_net_economic_value,
    estimate_policy_value_ipw,
)
from src.evaluation.inference import paired_bootstrap_policy_comparison
from src.visualization.plots_uplift import plot_uplift_curves
from src.visualization.plots_policy import plot_policy_value_vs_coverage, plot_cost_sensitivity
from src.visualization.plots_robustness import plot_policy_stability_heatmap, plot_scenario_regret_comparison


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("ECONOMICS-AWARE CAUSAL MARKETING TARGETING: EMPIRICAL RESEARCH PIPELINE")
    print("=" * 80)

    # 1. Load data
    print("\n[STEP 1/7] Loading and splitting Hillstrom dataset...")
    splits = prepare_hillstrom_splits()

    # Filter to binary setting: Mens E-Mail (1) vs Control (0) for Experiment A
    mask_train = splits.T_train.isin([0, 1])
    mask_val = splits.T_val.isin([0, 1])
    mask_test = splits.T_test.isin([0, 1])

    X_tr = splits.X_train.loc[mask_train].reset_index(drop=True)
    T_tr = splits.T_train.loc[mask_train].reset_index(drop=True)
    y_conv_tr = splits.y_conv_train.loc[mask_train].reset_index(drop=True)
    y_spend_tr = splits.y_spend_train.loc[mask_train].reset_index(drop=True)

    X_te = splits.X_test.loc[mask_test].reset_index(drop=True)
    T_te = splits.T_test.loc[mask_test].reset_index(drop=True)
    y_conv_te = splits.y_conv_test.loc[mask_test].reset_index(drop=True)
    y_spend_te = splits.y_spend_test.loc[mask_test].reset_index(drop=True)

    print(f"Filtered binary experiment: Train N = {len(X_tr):,}, Test N = {len(X_te):,}")

    # 2. Train Models
    print("\n[STEP 2/7] Training Response Models and Causal Uplift Learners...")
    # Response baselines
    resp_logistic = ResponseBaseline(model_type="logistic").fit(X_tr, y_conv_tr)
    resp_xgb = ResponseBaseline(model_type="xgboost").fit(X_tr, y_conv_tr)

    # Uplift learners on conversion
    s_learner = SLearner(is_classification=True).fit(X_tr, T_tr, y_conv_tr)
    t_learner = TLearner(is_classification=True).fit(X_tr, T_tr, y_conv_tr)
    x_learner = XLearner(is_classification=True).fit(X_tr, T_tr, y_conv_tr)
    dr_learner = DoublyRobustLearner(is_classification=True, random_state=42).fit(X_tr, T_tr, y_conv_tr)

    # Revenue models
    hurdle_model = HurdleRevenueUplift().fit(X_tr, T_tr, y_conv_tr, y_spend_tr)
    direct_rev_model = DirectRevenueUplift(method="t_learner").fit(X_tr, T_tr, y_spend_tr)

    # 3. Generate Predictions on Held-Out Test Set
    print("\n[STEP 3/7] Generating scoring vectors on held-out test data...")
    p_resp_logistic = resp_logistic.predict_proba(X_te)
    p_resp_xgb = resp_xgb.predict_proba(X_te)

    tau_s = s_learner.predict_cate(X_te)
    tau_t = t_learner.predict_cate(X_te)
    tau_x = x_learner.predict_cate(X_te)
    tau_dr = dr_learner.predict_cate(X_te)

    rev_hurdle = hurdle_model.predict_incremental_revenue(X_te)
    rev_direct = direct_rev_model.predict_incremental_revenue(X_te)

    # 4. Hypothesis H1: Ranking Discrepancy (Spearman Rank Correlation)
    print("\n[STEP 4/7] Testing Hypothesis H1 (Prediction vs Incrementality Ranking)...")
    spearman_corr, p_val_spearman = stats.spearmanr(p_resp_xgb, tau_t)
    spearman_rev, _ = stats.spearmanr(p_resp_xgb, rev_hurdle)
    print(f"  Spearman rho(Response XGB, T-Learner Conversion Uplift): {spearman_corr:.4f} (p = {p_val_spearman:.2e})")
    print(f"  Spearman rho(Response XGB, Hurdle Revenue Uplift):        {spearman_rev:.4f}")

    # Decile overlap
    top10_resp = set(np.argsort(p_resp_xgb)[::-1][:int(0.10 * len(X_te))])
    top10_uplift = set(np.argsort(tau_t)[::-1][:int(0.10 * len(X_te))])
    top10_overlap = len(top10_resp.intersection(top10_uplift)) / len(top10_resp)
    print(f"  Top-10% Customer Cohort Overlap: {top10_overlap:.1%}")

    # Uplift performance metrics (AUQC / Qini)
    q_resp = compute_qini_score(y_conv_te.values, T_te.values, p_resp_xgb)
    q_s = compute_qini_score(y_conv_te.values, T_te.values, tau_s)
    q_t = compute_qini_score(y_conv_te.values, T_te.values, tau_t)
    q_x = compute_qini_score(y_conv_te.values, T_te.values, tau_x)
    q_dr = compute_qini_score(y_conv_te.values, T_te.values, tau_dr)
    q_hurdle = compute_qini_score(y_conv_te.values, T_te.values, rev_hurdle)

    table3_data = [
        {"Model": "Model A2 (XGBoost Response)", "Family": "Response", "Qini (Conv)": q_resp},
        {"Model": "Model B1 (S-Learner)", "Family": "Meta-Learner", "Qini (Conv)": q_s},
        {"Model": "Model B2 (T-Learner)", "Family": "Meta-Learner", "Qini (Conv)": q_t},
        {"Model": "Model B3 (X-Learner)", "Family": "Meta-Learner", "Qini (Conv)": q_x},
        {"Model": "Model C (Doubly Robust)", "Family": "Doubly Robust", "Qini (Conv)": q_dr},
        {"Model": "Hurdle Revenue Uplift", "Family": "Revenue Decomposition", "Qini (Conv)": q_hurdle},
    ]
    df_table3 = pd.DataFrame(table3_data)
    print("\n" + tabulate(df_table3, headers="keys", tablefmt="github", floatfmt=".4f"))

    # Plot Figure 3 (Cumulative Conversion Uplift)
    models_conv_scores = {
        "Response (XGBoost)": p_resp_xgb,
        "S-Learner": tau_s,
        "T-Learner": tau_t,
        "X-Learner": tau_x,
        "DR-Learner": tau_dr,
        "Hurdle Revenue": rev_hurdle,
    }
    plot_uplift_curves(
        y=y_conv_te.values,
        treatment=T_te.values,
        models_scores=models_conv_scores,
        outcome_name="Conversion",
        output_path="outputs/figures/figure3_uplift_curves.png"
    )

    # Plot Figure 4 (Cumulative Spend Uplift)
    plot_uplift_curves(
        y=y_spend_te.values,
        treatment=T_te.values,
        models_scores=models_conv_scores,
        outcome_name="Spend ($)",
        output_path="outputs/figures/figure4_revenue_uplift_curves.png"
    )

    # 5. Policy Evaluation & Paired Bootstrap Inference (Hypotheses H2 & H3)
    print("\n[STEP 5/7] Evaluating Held-Out Policies with Paired Bootstrap Inference...")
    default_margin = 0.50
    default_cost = 0.25
    target_coverage = 0.20  # Contact top 20%

    pol_rand = np.zeros(len(X_te), dtype=int)
    pol_rand[:int(target_coverage * len(X_te))] = 1
    np.random.default_rng(42).shuffle(pol_rand)

    pol_resp = policy_top_fraction(p_resp_xgb, target_coverage)
    pol_uplift = policy_top_fraction(tau_t, target_coverage)
    pol_rev = policy_top_fraction(rev_hurdle, target_coverage)
    # Profit-aware policy uses utility U_i = Delta R_i * M - c
    util_hurdle = compute_incremental_utility(rev_hurdle, margin=default_margin, cost=default_cost)
    pol_econ = policy_top_fraction(util_hurdle, target_coverage)

    policies = {
        "Random Policy": pol_rand,
        "Response Policy (XGB)": pol_resp,
        "Uplift Policy (T-Learner)": pol_uplift,
        "Revenue Uplift Policy": pol_rev,
        "Profit-Aware Policy": pol_econ,
    }

    # Evaluate point values
    table4_rows = []
    for pol_name, pol_actions in policies.items():
        val_spend = estimate_policy_value_ipw(y_spend_te.values, T_te.values, pol_actions)
        net_val = estimate_net_economic_value(y_spend_te.values, T_te.values, pol_actions, default_margin, default_cost)
        conv_val = estimate_policy_value_ipw(y_conv_te.values, T_te.values, pol_actions)
        table4_rows.append({
            "Policy": pol_name,
            "Targeted (%)": f"{np.mean(pol_actions)*100:.1f}%",
            "Expected Conv Rate (%)": conv_val * 100,
            "Expected Gross Spend ($)": val_spend,
            "Net Economic Value ($)": net_val,
        })
    df_table4 = pd.DataFrame(table4_rows)
    print("\n" + tabulate(df_table4, headers="keys", tablefmt="github", floatfmt=".4f"))

    # Statistical test: Uplift vs Response
    print("\nRunning Paired Bootstrap Test: Revenue Uplift Policy vs Response Policy (500 resamples)...")
    boot_res = paired_bootstrap_policy_comparison(
        spend=y_spend_te.values,
        treatment=T_te.values,
        policy_a=pol_rev,
        policy_b=pol_resp,
        margin=default_margin,
        cost=default_cost,
        n_bootstraps=500,
        seed=42,
    )
    print(f"  Point Estimate Difference Delta V: ${boot_res['point_diff']:+.4f}")
    print(f"  95% Bootstrap CI: [${boot_res['ci_diff'][0]:+.4f}, ${boot_res['ci_diff'][1]:+.4f}]")
    print(f"  Bootstrap p-value: {boot_res['p_value']:.4f}")
    print(f"  Statistically Significant (p < 0.05): {boot_res['significant']}")

    # 6. Budget and Cost Sensitivity (Hypotheses H4 & H5)
    print("\n[STEP 6/7] Sweeping Budget Coverage and Campaign Contact Costs...")
    coverages = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
    coverage_values = {"Response Policy": [], "Uplift Policy": [], "Revenue Uplift Policy": []}

    for frac in coverages:
        p_act_resp = policy_top_fraction(p_resp_xgb, frac)
        p_act_uplift = policy_top_fraction(tau_t, frac)
        p_act_rev = policy_top_fraction(rev_hurdle, frac)

        coverage_values["Response Policy"].append(
            estimate_net_economic_value(y_spend_te.values, T_te.values, p_act_resp, default_margin, default_cost)
        )
        coverage_values["Uplift Policy"].append(
            estimate_net_economic_value(y_spend_te.values, T_te.values, p_act_uplift, default_margin, default_cost)
        )
        coverage_values["Revenue Uplift Policy"].append(
            estimate_net_economic_value(y_spend_te.values, T_te.values, p_act_rev, default_margin, default_cost)
        )

    plot_policy_value_vs_coverage(coverages, coverage_values, "outputs/figures/figure5_policy_value_vs_coverage.png")

    # Cost sensitivity sweep
    costs = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    cost_values = {"Response (20% coverage)": [], "Revenue Uplift (20% coverage)": [], "Profit-Aware (Economic Threshold)": []}

    for c in costs:
        cost_values["Response (20% coverage)"].append(
            estimate_net_economic_value(y_spend_te.values, T_te.values, pol_resp, default_margin, c)
        )
        cost_values["Revenue Uplift (20% coverage)"].append(
            estimate_net_economic_value(y_spend_te.values, T_te.values, pol_rev, default_margin, c)
        )
        # Optimal threshold policy for cost c: treat where U_i > 0
        pol_econ_c = (compute_incremental_utility(rev_hurdle, margin=default_margin, cost=c) > 0).astype(int)
        cost_values["Profit-Aware (Economic Threshold)"].append(
            estimate_net_economic_value(y_spend_te.values, T_te.values, pol_econ_c, default_margin, c)
        )

    plot_cost_sensitivity(costs, cost_values, "outputs/figures/figure6_policy_value_vs_cost.png")

    # 7. Scenario Analysis, Minimax Regret & Policy Stability (Hypothesis H6)
    print("\n[STEP 7/7] Conducting Scenario Analysis, Minimax Regret, and Jaccard Stability...")
    scenarios = build_scenario_grid(margins=[0.20, 0.40, 0.60, 0.80], costs=[0.10, 0.25, 0.50, 1.00])
    
    scenario_policy_values = {}
    scenario_decisions = {}

    for sc in scenarios:
        u_hurdle = compute_incremental_utility(rev_hurdle, margin=sc.margin, cost=sc.cost)
        
        # Policy actions under this scenario
        act_resp = policy_top_fraction(p_resp_xgb, target_coverage)
        act_uplift = policy_top_fraction(tau_t, target_coverage)
        act_rev = policy_top_fraction(rev_hurdle, target_coverage)
        # Threshold policy targeting only positive expected economic utility
        act_econ = (u_hurdle > 0).astype(int)

        scenario_decisions[sc.scenario_id] = act_econ

        scenario_policy_values[sc.scenario_id] = {
            "Response Policy": estimate_net_economic_value(y_spend_te.values, T_te.values, act_resp, sc.margin, sc.cost),
            "Uplift Policy": estimate_net_economic_value(y_spend_te.values, T_te.values, act_uplift, sc.margin, sc.cost),
            "Revenue Uplift": estimate_net_economic_value(y_spend_te.values, T_te.values, act_rev, sc.margin, sc.cost),
            "Profit-Aware (Threshold)": estimate_net_economic_value(y_spend_te.values, T_te.values, act_econ, sc.margin, sc.cost),
        }

    values_df, regret_df, robust_pol = compute_regret_table(scenario_policy_values)
    print(f"\nMinimax Regret Optimal Policy: {robust_pol}")
    print("\nRegret Summary (Max Regret across scenarios):")
    print(tabulate(regret_df.loc[["MAX_REGRET", "MEAN_REGRET"]], headers="keys", tablefmt="github", floatfmt=".4f"))

    # Jaccard stability matrix
    subset_scenarios = {k: v for k, v in scenario_decisions.items() if k in ["M20_C010", "M40_C025", "M60_C050", "M80_C100"]}
    stability_matrix = compute_policy_stability_matrix(subset_scenarios)
    plot_policy_stability_heatmap(stability_matrix, "outputs/figures/figure10_policy_stability_heatmap.png")
    plot_scenario_regret_comparison(regret_df, "outputs/figures/figure9_robust_vs_scenario_regret.png")

    # Save Tables
    out_dir = Path("outputs/tables")
    with open(out_dir / "table3_uplift_performance.md", "w") as f:
        f.write("### Table 3: Uplift and Qini Performance\n\n" + tabulate(df_table3, headers="keys", tablefmt="github", floatfmt=".4f") + "\n")
    with open(out_dir / "table4_policy_comparisons.md", "w") as f:
        f.write("### Table 4: Policy Value and Bootstrap Inference\n\n" + tabulate(df_table4, headers="keys", tablefmt="github", floatfmt=".4f") + "\n")
        f.write(f"\n**Bootstrap Test (Revenue Uplift vs Response)**: Delta V = ${boot_res['point_diff']:+.4f} (95% CI: [${boot_res['ci_diff'][0]:+.4f}, ${boot_res['ci_diff'][1]:+.4f}], p = {boot_res['p_value']:.4f})\n")
    with open(out_dir / "table6_regret_summary.md", "w") as f:
        f.write("### Table 6: Minimax Regret across Scenarios\n\n" + tabulate(regret_df, headers="keys", tablefmt="github", floatfmt=".4f") + "\n")

    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Full research pipeline executed in {elapsed:.2f} seconds!")


if __name__ == "__main__":
    run_pipeline()
