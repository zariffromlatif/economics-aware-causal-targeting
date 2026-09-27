"""Multi-Arm Personalization Experiment Runner (Experiment B / Hypothesis H7).

Compares:
1. Uniform Baseline (Contact top 20% with Mens E-Mail only)
2. Uniform Alternative (Contact top 20% with Womens E-Mail only)
3. Personalized Causal Targeting (Assign each contacted customer to their individually optimal arm)

Computes IPW policy values on the 3-arm held-out test set and conducts paired bootstrap inference.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data.loader import prepare_hillstrom_splits
from src.policy.multi_arm import (
    MultiArmPersonalizedPolicy,
    evaluate_multiarm_policy_ipw,
    plot_customer_treatment_allocation,
)
from src.policy.allocator import policy_top_fraction


def run_multiarm_study():
    start_time = time.time()
    print("=" * 80)
    print("EXPERIMENT B: MULTI-ARM TREATMENT PERSONALIZATION (MENS VS WOMENS VS CONTROL)")
    print("=" * 80)

    # 1. Load full 3-arm splits
    print("\n[1/5] Loading 3-arm Hillstrom splits...")
    splits = prepare_hillstrom_splits()
    X_tr, T_tr = splits.X_train, splits.T_train
    yc_tr, ys_tr = splits.y_conv_train, splits.y_spend_train

    X_te, T_te = splits.X_test, splits.T_test
    yc_te, ys_te = splits.y_conv_test, splits.y_spend_test

    print(f"Train N = {len(X_tr):,}, Held-Out Test N = {len(X_te):,}")
    print(f"Test Arm Counts: Control: {np.sum(T_te == 0):,}, Mens: {np.sum(T_te == 1):,}, Womens: {np.sum(T_te == 2):,}")

    # 2. Fit Multi-Arm CATE Hurdle Models
    print("\n[2/5] Training Multi-Arm Causal Revenue Models...")
    multi_policy = MultiArmPersonalizedPolicy(random_state=42)
    multi_policy.fit(X_tr, T_tr, yc_tr, ys_tr)

    # 3. Formulate Candidate Policies (20% Budget Coverage)
    print("\n[3/5] Constructing Policies on Held-Out Test Data...")
    target_budget = 0.20
    margin = 0.50
    cost_arm1 = 0.25
    cost_arm2 = 0.25
    costs = {0: 0.0, 1: cost_arm1, 2: cost_arm2}

    # Policy A: No-contact baseline
    pol_zero = np.zeros(len(X_te), dtype=int)

    # Policy B: Uniform Global Best (Assign top 20% to Mens E-mail only)
    delta_r1, delta_r2 = multi_policy.predict_incremental_revenues(X_te)
    top20_mens_idx = policy_top_fraction(delta_r1, target_budget)
    pol_uniform_mens = np.where(top20_mens_idx == 1, 1, 0)

    # Policy C: Uniform Alternative (Assign top 20% to Womens E-mail only)
    top20_womens_idx = policy_top_fraction(delta_r2, target_budget)
    pol_uniform_womens = np.where(top20_womens_idx == 1, 2, 0)

    # Policy D: Personalized Policy (Assign each customer to their individual utility-maximizing arm)
    pol_personalized = multi_policy.allocate_personalized(
        X_te, margin=margin, cost_arm1=cost_arm1, cost_arm2=cost_arm2, budget_fraction=target_budget
    )

    # 4. Evaluate Held-Out Policy Values via 3-Arm IPW
    print("\n[4/5] Evaluating Policies on Held-Out Randomized Test Set...")
    policies = {
        "Baseline (No Contact)": pol_zero,
        "Uniform Mens E-Mail (Global Best)": pol_uniform_mens,
        "Uniform Womens E-Mail": pol_uniform_womens,
        "Personalized Causal Policy (Proposed)": pol_personalized,
    }

    eval_results = []
    for pol_name, actions in policies.items():
        res = evaluate_multiarm_policy_ipw(
            y_spend=ys_te.values,
            treatment=T_te.values,
            policy_actions=actions,
            margin=margin,
            costs=costs,
            propensity=1.0/3.0,
        )
        eval_results.append({
            "Policy": pol_name,
            "Targeted (%)": f"{res['pct_treated']:.1f}%",
            "Mens (%)": f"{res['pct_arm1']:.1f}%",
            "Womens (%)": f"{res['pct_arm2']:.1f}%",
            "Gross Spend ($)": res["gross_spend"],
            "Mean Cost ($)": res["mean_cost"],
            "Net Policy Value ($)": res["net_value"],
        })

    df_table7 = pd.DataFrame(eval_results)
    print("\n" + tabulate(df_table7, headers="keys", tablefmt="github", floatfmt=".4f"))

    # 5. Paired Bootstrap Hypothesis Test (H7: Personalized vs Uniform Best)
    print("\n[5/5] Running Paired Bootstrap Test: Personalized vs Uniform Mens (500 resamples)...")
    rng = np.random.default_rng(42)
    n = len(X_te)
    n_boot = 500

    boot_diffs = []
    val_pers_point = evaluate_multiarm_policy_ipw(ys_te.values, T_te.values, pol_personalized, margin, costs)["net_value"]
    val_unif_point = evaluate_multiarm_policy_ipw(ys_te.values, T_te.values, pol_uniform_mens, margin, costs)["net_value"]
    point_diff = val_pers_point - val_unif_point

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        v_p = evaluate_multiarm_policy_ipw(ys_te.values[idx], T_te.values[idx], pol_personalized[idx], margin, costs)["net_value"]
        v_u = evaluate_multiarm_policy_ipw(ys_te.values[idx], T_te.values[idx], pol_uniform_mens[idx], margin, costs)["net_value"]
        boot_diffs.append(v_p - v_u)

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_val = 2.0 * min(np.mean(np.array(boot_diffs) <= 0), np.mean(np.array(boot_diffs) >= 0))
    p_val = min(1.0, max(1.0 / n_boot, p_val))

    print(f"\nPersonalization Advantage Delta V: ${point_diff:+.4f}")
    print(f"95% Bootstrap CI: [${ci_low:+.4f}, ${ci_high:+.4f}]")
    print(f"Bootstrap p-value: {p_val:.4f}")

    # Generate Figure 12 (Allocation Breakdown)
    plot_customer_treatment_allocation(X_te, pol_personalized, "outputs/figures/figure12_customer_treatment_allocation.png")

    # Save Table 7
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table7_multiarm_personalization.md", "w") as f:
        f.write("### Table 7: Multi-Arm Policy Evaluation and Personalization Advantage\n\n")
        f.write(tabulate(df_table7, headers="keys", tablefmt="github", floatfmt=".4f") + "\n\n")
        f.write(f"**Hypothesis H7 Test (Personalized vs Uniform Mens)**:\n")
        f.write(f"- Point Estimate Difference: ${point_diff:+.4f}\n")
        f.write(f"- 95% Bootstrap CI: [${ci_low:+.4f}, ${ci_high:+.4f}]\n")
        f.write(f"- Empirical p-value: {p_val:.4f}\n")

    print(f"\n[COMPLETE] Multi-arm experiment finished in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    run_multiarm_study()
