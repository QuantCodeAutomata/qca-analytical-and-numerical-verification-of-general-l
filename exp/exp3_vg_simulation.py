"""
Experiment 3: Monte Carlo Replication of the Variance Gamma Bias-Reduction
Study — MLE vs Firth Estimators.

Reproduces the paper's simulation study (Figure 1):
- C = 0.1, T = 1, λ_+^true = 1, λ_-^true = 1.5
- Sample sizes n ∈ {50, 100, 200, 500}
- R = 5000 Monte Carlo replications
- Random seed = 42 (implementation choice; not specified by the paper)

Uses sufficient-statistic formulation: S_± ~ Gamma(shape=nTC, rate=λ_±^true).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vg_simulation import run_full_study, analytical_expectations


# ---------------------------------------------------------------------------
# Paper-specified constants (must not be changed)
# ---------------------------------------------------------------------------
C = 0.1
T = 1.0
LAMBDA_PLUS_TRUE = 1.0
LAMBDA_MINUS_TRUE = 1.5
SAMPLE_SIZES = [50, 100, 200, 500]
R = 5000
SEED = 42   # implementation choice (paper does not specify)


def run_simulation() -> pd.DataFrame:
    """Run the full MC study and return long-form results."""
    print(f"Running VG bias study: C={C}, T={T}, λ+={LAMBDA_PLUS_TRUE}, "
          f"λ-={LAMBDA_MINUS_TRUE}, R={R}, seed={SEED}")
    df = run_full_study(
        C=C, T=T,
        lambda_plus=LAMBDA_PLUS_TRUE,
        lambda_minus=LAMBDA_MINUS_TRUE,
        sample_sizes=SAMPLE_SIZES,
        R=R,
        seed=SEED,
    )
    return df


def compare_to_analytic(df: pd.DataFrame) -> pd.DataFrame:
    """Compare MC estimates to exact analytic expectations."""
    analytic = analytical_expectations(C, T, LAMBDA_PLUS_TRUE, LAMBDA_MINUS_TRUE, SAMPLE_SIZES)

    rows = []
    for _, arow in analytic.iterrows():
        n = int(arow["n"])
        param = arow["param"]
        # Find MC rows
        lbl_mle = f"MLE {param}"
        lbl_firth = f"Firth {param}"
        mc_mle = df[(df["n"] == n) & (df["estimator"] == lbl_mle)]
        mc_firth = df[(df["n"] == n) & (df["estimator"] == lbl_firth)]
        if mc_mle.empty or mc_firth.empty:
            continue
        rows.append({
            "n": n, "param": param, "shape_k": arow["shape_k"],
            "E_MLE_analytic": arow["E[MLE]"],
            "E_MLE_mc": float(mc_mle["mean"].iloc[0]),
            "E_Firth_analytic": arow["E[Firth]"],
            "E_Firth_mc": float(mc_firth["mean"].iloc[0]),
            "rel_bias_MLE_analytic": arow["rel_bias_MLE"],
            "rel_bias_MLE_mc": float(mc_mle["rel_bias"].iloc[0]),
            "rel_bias_Firth_analytic": arow["rel_bias_Firth"],
            "rel_bias_Firth_mc": float(mc_firth["rel_bias"].iloc[0]),
        })
    return pd.DataFrame(rows)


def plot_figure1(df: pd.DataFrame, save_path: str) -> None:
    """Reproduce Figure 1 from the paper: relative bias and RMSE vs n.

    Panel (a): relative bias.
    Panel (b): RMSE.

    Line styles:
        Firth: solid + square marker
        MLE: dashed + circle marker
    Colors:
        λ+: dark
        λ-: medium gray
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    styles = {
        "MLE λ+": dict(ls="--", marker="o", color="#2c2c2c", label="MLE λ+"),
        "MLE λ-": dict(ls="--", marker="o", color="#888888", label="MLE λ-"),
        "Firth λ+": dict(ls="-", marker="s", color="#1f4e79", label="Firth λ+"),
        "Firth λ-": dict(ls="-", marker="s", color="#5b9bd5", label="Firth λ-"),
    }

    for ax, metric, ylabel in [
        (axes[0], "rel_bias", "Relative Bias"),
        (axes[1], "rmse", "RMSE"),
    ]:
        for label, style in styles.items():
            sub = df[df["estimator"] == label].sort_values("n")
            ax.plot(sub["n"], sub[metric], **style)
        ax.set_xlabel("Sample size n")
        ax.set_ylabel(ylabel)
        ax.set_title(f"VG Estimators: {ylabel} vs n")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        if metric == "rel_bias":
            ax.axhline(0, color="black", lw=0.8, ls=":")

    plt.suptitle("Figure 1 Replication: VG MLE vs Firth Bias Reduction", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def pivot_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide-format summary table matching the paper's style."""
    wide = df.pivot_table(
        index="n",
        columns="estimator",
        values=["rel_bias", "rmse"],
    )
    wide.columns = [f"{v}_{k}" for v, k in wide.columns]
    wide = wide.reset_index()
    return wide


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("EXP 3: VG Monte Carlo Bias-Reduction Study (MLE vs Firth)")
    print("=" * 65)
    print(f"Implementation choice: random seed = {SEED} (not specified by paper)")
    print(f"Parameterisation: S_± ~ Gamma(shape=nTC, scale=1/λ_±^true)")

    # --- Run study ---
    df = run_simulation()
    print("\nFull MC results (long form):")
    print(df.to_string(index=False, float_format="{:.6f}".format))

    # --- Pivot table ---
    wide = pivot_summary_table(df)
    print("\nWide-format summary table:")
    print(wide.to_string(index=False, float_format="{:.6f}".format))

    # --- Analytic comparison ---
    df_analytic = compare_to_analytic(df)
    print("\nAnalytic vs MC comparison:")
    print(df_analytic.to_string(index=False, float_format="{:.6f}".format))

    # --- Save ---
    df.to_csv("results/exp3_mc_long_form.csv", index=False)
    wide.to_csv("results/exp3_mc_wide_table.csv", index=False)
    df_analytic.to_csv("results/exp3_analytic_comparison.csv", index=False)

    # --- Plot ---
    plot_figure1(df, "results/exp3_figure1_replication.png")
    print("\nFigure 1 replication saved to results/exp3_figure1_replication.png")

    # --- Qualitative assertions matching paper's claims ---
    for n in SAMPLE_SIZES:
        for param in ["λ+", "λ-"]:
            mle_row = df[(df["n"] == n) & (df["estimator"] == f"MLE {param}")]
            firth_row = df[(df["n"] == n) & (df["estimator"] == f"Firth {param}")]
            assert not mle_row.empty and not firth_row.empty

            mle_bias = float(mle_row["rel_bias"].iloc[0])
            firth_bias = float(firth_row["rel_bias"].iloc[0])
            mle_rmse = float(mle_row["rmse"].iloc[0])
            firth_rmse = float(firth_row["rmse"].iloc[0])

            # MLE should be positively biased (for nTC > 1 which always holds here)
            assert mle_bias > 0, f"MLE should be positively biased: n={n}, {param}, bias={mle_bias:.4f}"
            # Firth should be near zero
            assert abs(firth_bias) < abs(mle_bias), \
                f"Firth bias should be < MLE bias: n={n}, {param}"
            # Firth RMSE ≤ MLE RMSE
            assert firth_rmse <= mle_rmse + 1e-6, \
                f"Firth RMSE should ≤ MLE RMSE: n={n}, {param}"

    # Analytic check: MC MLE mean ≈ analytic E[MLE] = k/(k-1)*λ
    for _, row in df_analytic.iterrows():
        mc_val = row["E_MLE_mc"]
        analytic_val = row["E_MLE_analytic"]
        rel_err = abs(mc_val - analytic_val) / analytic_val
        assert rel_err < 0.05, f"MC MLE mean deviates from analytic: {rel_err:.4f} at n={row['n']}"

    print("\n✓ All qualitative assertions passed.")
    print("\n[Summary]")
    summary = {
        "MLE_max_rel_bias_lp": float(df[df["estimator"] == "MLE λ+"]["rel_bias"].max()),
        "Firth_max_abs_rel_bias_lp": float(df[df["estimator"] == "Firth λ+"]["rel_bias"].abs().max()),
        "MLE_max_rel_bias_lm": float(df[df["estimator"] == "MLE λ-"]["rel_bias"].max()),
        "Firth_max_abs_rel_bias_lm": float(df[df["estimator"] == "Firth λ-"]["rel_bias"].abs().max()),
        "MLE_RMSE_n50_lp": float(df[(df["n"] == 50) & (df["estimator"] == "MLE λ+")]["rmse"].iloc[0]),
        "Firth_RMSE_n50_lp": float(df[(df["n"] == 50) & (df["estimator"] == "Firth λ+")]["rmse"].iloc[0]),
    }
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")

    return summary, df


if __name__ == "__main__":
    main()
