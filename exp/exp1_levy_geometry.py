"""
Experiment 1: Analytical and Numerical Verification of General Lévy-Process
α-Divergence and Information Geometry.

Verifies:
1. Compact Δ formula matches structure
2. α-duality: D^α(P||Q) = D^{-α}(Q||P)
3. Zero divergence at equal parameters
4. Fisher information matrix symmetry and numerical vs direct agreement
5. α-connection numerical vs direct formula agreement

Uses the GTS family as the concrete test case (covers σ=0 regime).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.gts import GTS, GTSParams
from src.divergence import (
    alpha_divergence, verify_duality, verify_zero_divergence,
    delta_T_alpha,
)
from src.info_geometry import (
    fisher_information_numerical, fisher_information_direct,
    alpha_connection_numerical, alpha_connection_direct,
    check_fisher_symmetry,
)

# ---------------------------------------------------------------------------
# Parameter grid (implementation choice: documented)
# ---------------------------------------------------------------------------
# Base parameters (C±, a±, λ±):  only λ± vary between P and Q.
# Implementation choice: use C=0.3, a=0.5 as the shared structural parameters.

ALPHA_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
T = 1.0

gts = GTS()

# Two parameter pairs for P and Q (only λ± differ)
PARAM_PAIRS = [
    # (params_p, params_q)
    (
        GTSParams(C_plus=0.3, C_minus=0.2, a_plus=0.5, a_minus=0.7, lam_plus=1.0, lam_minus=1.0),
        GTSParams(C_plus=0.3, C_minus=0.2, a_plus=0.5, a_minus=0.7, lam_plus=1.5, lam_minus=1.2),
    ),
    (
        GTSParams(C_plus=0.1, C_minus=0.1, a_plus=1.3, a_minus=1.7, lam_plus=2.0, lam_minus=0.8),
        GTSParams(C_plus=0.1, C_minus=0.1, a_plus=1.3, a_minus=1.7, lam_plus=3.0, lam_minus=1.5),
    ),
]


def run_divergence_checks() -> pd.DataFrame:
    """Check α-duality and zero-divergence across the parameter grid."""
    rows = []
    for pair_idx, (pp, pq) in enumerate(PARAM_PAIRS):
        for alpha in ALPHA_GRID:
            # Zero divergence
            zero_err = verify_zero_divergence(gts, pp, alpha, T)
            # Duality
            duality = verify_duality(gts, pp, pq, alpha, T)
            # Compact vs closed-form
            D_numeric = alpha_divergence(gts, pp, pq, alpha, T)
            D_closed = gts.closed_form_divergence(pp, pq, alpha, T)
            cf_err = abs(D_numeric - D_closed)
            rows.append(
                {
                    "pair": pair_idx,
                    "alpha": alpha,
                    "D_alpha_PQ_numeric": duality["D_alpha_PQ"],
                    "D_neg_alpha_QP_numeric": duality["D_neg_alpha_QP"],
                    "duality_abs_error": duality["abs_error"],
                    "duality_rel_error": duality["rel_error"],
                    "zero_div_error": zero_err,
                    "D_numeric": D_numeric,
                    "D_closed_form": D_closed,
                    "closed_form_abs_error": cf_err,
                }
            )
    return pd.DataFrame(rows)


def run_geometry_checks(params_idx: int = 0) -> dict:
    """Compare numerical and direct Fisher information and α-connection."""
    pp = PARAM_PAIRS[params_idx][0]
    # Work only on the (λ+, λ-) subspace (indices 4 and 5 of the full 6-param vector)
    # For the geometry check we fix C and a and let only λ vary.
    # Use reduced 2-param representation: just (lam_plus, lam_minus)
    # via a wrapper lambda-only family.
    from src.gts import GTS as GTSfull

    class GTSLambdaOnly(GTSfull):
        """GTS with fixed C,a and only (λ+, λ-) as free parameters."""
        def __init__(self, C_plus, C_minus, a_plus, a_minus):
            self.C_plus = C_plus
            self.C_minus = C_minus
            self.a_plus = a_plus
            self.a_minus = a_minus

        def levy_density(self, x, params):
            lp, lm = params
            p = GTSParams(self.C_plus, self.C_minus, self.a_plus, self.a_minus, lp, lm)
            return super().levy_density(x, p)

        def log_levy_density(self, x, params):
            lp, lm = params
            p = GTSParams(self.C_plus, self.C_minus, self.a_plus, self.a_minus, lp, lm)
            return super().log_levy_density(x, p)

        def drift(self, params): return 0.0
        def diffusion(self, params): return 0.0

    fam = GTSLambdaOnly(pp.C_plus, pp.C_minus, pp.a_plus, pp.a_minus)
    xi = np.array([pp.lam_plus, pp.lam_minus])

    g_num = fisher_information_numerical(fam, xi, T, eps=1e-4)
    g_dir = fisher_information_direct(fam, xi, T, fd_eps=1e-5)

    # Closed-form Fisher (analytical)
    g_cf = gts.fisher_matrix(pp, T)
    # Extract only (λ+, λ-) block (indices 4,5) — but fisher_matrix returns 2x2 for λ±
    # g_cf is already 2x2.

    fisher_resid_num_vs_direct = np.abs(g_num - g_dir)
    fisher_resid_num_vs_cf = np.abs(g_num - g_cf)

    # α-connection for α=0.5
    alpha_test = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Gamma_num = alpha_connection_numerical(fam, xi, alpha_test, T, eps=1e-3)
        Gamma_dir = alpha_connection_direct(fam, xi, alpha_test, T)

    conn_cf = gts.alpha_connection(pp, alpha_test, T)
    Gamma_cf = np.zeros((2, 2, 2))
    Gamma_cf[0, 0, 0] = conn_cf["Gamma_lp_lp_lp"]
    Gamma_cf[1, 1, 1] = conn_cf["Gamma_lm_lm_lm"]

    conn_resid_num_vs_dir = np.abs(Gamma_num - Gamma_dir)
    conn_resid_num_vs_cf = np.abs(Gamma_num - Gamma_cf)

    return {
        "g_numerical": g_num,
        "g_direct": g_dir,
        "g_closed_form": g_cf,
        "fisher_resid_num_vs_direct": fisher_resid_num_vs_direct,
        "fisher_resid_num_vs_closed": fisher_resid_num_vs_cf,
        "fisher_symmetric": check_fisher_symmetry(g_num),
        "alpha_test": alpha_test,
        "Gamma_numerical": Gamma_num,
        "Gamma_direct": Gamma_dir,
        "Gamma_closed_form": Gamma_cf,
        "conn_resid_num_vs_direct": conn_resid_num_vs_dir,
        "conn_resid_num_vs_closed": conn_resid_num_vs_cf,
    }


def plot_divergence_vs_alpha(df: pd.DataFrame, save_path: str) -> None:
    """Plot D^α(P||Q) vs α for both parameter pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for pair_idx in [0, 1]:
        sub = df[df["pair"] == pair_idx]
        ax = axes[pair_idx]
        ax.plot(sub["alpha"], sub["D_numeric"], "o-", label="D^α numeric", color="royalblue")
        ax.plot(sub["alpha"], sub["D_closed_form"], "s--", label="D^α closed form", color="tomato")
        ax.set_xlabel("α")
        ax.set_ylabel("D^α(P||Q)")
        ax.set_title(f"Pair {pair_idx + 1}: α-Divergence vs α")
        ax.legend()
        ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_duality_errors(df: pd.DataFrame, save_path: str) -> None:
    """Plot duality absolute error vs α."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair_idx in [0, 1]:
        sub = df[df["pair"] == pair_idx]
        ax.semilogy(sub["alpha"], sub["duality_abs_error"] + 1e-16, "o-",
                    label=f"Pair {pair_idx + 1}")
    ax.set_xlabel("α")
    ax.set_ylabel("|D^α(P||Q) − D^{-α}(Q||P)|")
    ax.set_title("α-Duality Absolute Error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("EXP 1: General Lévy α-Divergence and Information Geometry")
    print("=" * 65)

    # --- Divergence checks ---
    print("\n[1] Running divergence and duality checks ...")
    df_div = run_divergence_checks()

    print("\nDivergence table (all pairs × α values):")
    print(df_div.to_string(index=False, float_format="{:.6e}".format))
    df_div.to_csv("results/exp1_divergence_checks.csv", index=False)

    max_duality_err = df_div["duality_abs_error"].max()
    max_zero_err = df_div["zero_div_error"].max()
    max_cf_err = df_div["closed_form_abs_error"].max()
    print(f"\nMax duality error       : {max_duality_err:.3e}")
    print(f"Max zero-divergence err : {max_zero_err:.3e}")
    print(f"Max closed-form error   : {max_cf_err:.3e}")

    assert max_duality_err < 1e-4, f"Duality failed: max error={max_duality_err:.3e}"
    assert max_zero_err < 1e-6, f"Zero-divergence failed: max error={max_zero_err:.3e}"

    # --- Geometry checks ---
    print("\n[2] Running geometry checks (Fisher info & α-connection) ...")
    geom = run_geometry_checks(params_idx=0)

    print("\nFisher matrix (numerical):")
    print(np.array2string(geom["g_numerical"], precision=4))
    print("\nFisher matrix (direct theorem formula):")
    print(np.array2string(geom["g_direct"], precision=4))
    print("\nFisher matrix (closed form):")
    print(np.array2string(geom["g_closed_form"], precision=4))
    print(f"\nFisher matrix symmetric: {geom['fisher_symmetric']}")
    print("\nMax |numerical - direct| residual:")
    print(np.array2string(geom["fisher_resid_num_vs_direct"], precision=3))
    print("\nMax |numerical - closed form| residual:")
    print(np.array2string(geom["fisher_resid_num_vs_closed"], precision=3))

    print(f"\nα-connection (α={geom['alpha_test']}) — closed form diagonal:")
    print(f"  Γ_lp,lp,lp = {geom['Gamma_closed_form'][0,0,0]:.6f}")
    print(f"  Γ_lm,lm,lm = {geom['Gamma_closed_form'][1,1,1]:.6f}")
    print("α-connection max |num − direct| residual (diag elements):")
    print(f"  [0,0,0]: {geom['conn_resid_num_vs_direct'][0,0,0]:.3e}")
    print(f"  [1,1,1]: {geom['conn_resid_num_vs_direct'][1,1,1]:.3e}")

    # --- Plots ---
    plot_divergence_vs_alpha(df_div, "results/exp1_divergence_vs_alpha.png")
    plot_duality_errors(df_div, "results/exp1_duality_errors.png")
    print("\nPlots saved to results/.")

    # --- Summary ---
    summary = {
        "max_duality_abs_error": float(max_duality_err),
        "max_zero_divergence_error": float(max_zero_err),
        "max_closed_form_vs_numeric_error": float(max_cf_err),
        "fisher_max_num_vs_direct": float(geom["fisher_resid_num_vs_direct"].max()),
        "fisher_max_num_vs_closed": float(geom["fisher_resid_num_vs_closed"].max()),
        "fisher_symmetric": geom["fisher_symmetric"],
        "conn_max_num_vs_direct_diag": float(
            max(geom["conn_resid_num_vs_direct"][0, 0, 0],
                geom["conn_resid_num_vs_direct"][1, 1, 1])
        ),
    }
    print("\n[Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary, df_div, geom


if __name__ == "__main__":
    main()
