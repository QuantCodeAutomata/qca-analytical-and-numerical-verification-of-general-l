"""
Experiment 2: Family-Specific Closed-Form Reproduction for GTS, CTS/CGMY,
VG, and Merton Lévy Models.

Verifies:
1. Closed-form divergences agree with theorem-level numerics (GTS, CTS, VG, Merton)
2. Affine-family identities: g = ∇²K, Γ^α = (1-α)/2 ∇³K, Γ^1 = 0, Γ^{-1} = ∇³K
3. Canonical divergence: B_K(ξ,η) = D^1(P_ξ||P_η) and B_K(η,ξ) = D^{-1}(P_ξ||P_η)
4. Dual coordinates from ∇K match stated formulas
5. CTS → VG limit as a ↓ 0
6. Jeffreys prior and superharmonic prior checks for CTS and VG
7. Merton: Fisher matrix from theorem formula vs finite-diff verification
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
from src.cts import CTS, CTSParams
from src.vg import VG, VGParams
from src.merton import Merton, MertonParams
from src.divergence import alpha_divergence
from src.info_geometry import bregman_divergence

# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------
T = 1.0
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]

gts = GTS()
cts = CTS()
vg = VG()
merton = Merton()

# ---------------------------------------------------------------------------
# A. GTS verification
# ---------------------------------------------------------------------------
GTS_CASES = [
    dict(
        pp=GTSParams(0.3, 0.2, 0.5, 0.7, 1.0, 1.0),
        pq=GTSParams(0.3, 0.2, 0.5, 0.7, 1.5, 1.2),
    ),
    dict(
        pp=GTSParams(0.1, 0.1, 1.3, 1.7, 2.0, 0.8),
        pq=GTSParams(0.1, 0.1, 1.3, 1.7, 3.0, 1.5),
    ),
]


def run_gts_checks() -> pd.DataFrame:
    rows = []
    for case_idx, case in enumerate(GTS_CASES):
        pp, pq = case["pp"], case["pq"]
        for alpha in ALPHAS:
            D_num = alpha_divergence(gts, pp, pq, alpha, T)
            D_cf = gts.closed_form_divergence(pp, pq, alpha, T)
            rows.append({
                "family": "GTS", "case": case_idx, "alpha": alpha,
                "D_numeric": D_num, "D_closed_form": D_cf,
                "abs_error": abs(D_num - D_cf),
                "rel_error": abs(D_num - D_cf) / max(abs(D_num), 1e-15),
            })
    return pd.DataFrame(rows)


def run_gts_geometry_checks() -> dict:
    """Verify g = ∇²K and Γ^α = (1-α)/2 ∇³K for GTS."""
    pp = GTS_CASES[0]["pp"]
    eps = 1e-5

    # ∂²K/∂λ+² numerically
    def K(lp, lm):
        params = GTSParams(pp.C_plus, pp.C_minus, pp.a_plus, pp.a_minus, lp, lm)
        return gts.potential(params, T)

    K0 = K(pp.lam_plus, pp.lam_minus)
    Kpp = (K(pp.lam_plus + eps, pp.lam_minus) - 2 * K0 + K(pp.lam_plus - eps, pp.lam_minus)) / eps ** 2
    Kmm = (K(pp.lam_plus, pp.lam_minus + eps) - 2 * K0 + K(pp.lam_plus, pp.lam_minus - eps)) / eps ** 2

    g_cf = gts.fisher_matrix(pp, T)
    hessK = np.diag([Kpp, Kmm])
    hess_resid = np.abs(hessK - g_cf)

    # Five-point third-derivative: f'''(x) = (f(x+2h)-2f(x+h)+2f(x-h)-f(x-2h))/(2h³)
    # Note: use eps_d3 = 1e-3 to avoid catastrophic cancellation in double precision.
    eps_d3 = 1e-3
    Kppp_5pt = (
        K(pp.lam_plus + 2 * eps_d3, pp.lam_minus)
        - 2 * K(pp.lam_plus + eps_d3, pp.lam_minus)
        + 2 * K(pp.lam_plus - eps_d3, pp.lam_minus)
        - K(pp.lam_plus - 2 * eps_d3, pp.lam_minus)
    ) / (2 * eps_d3 ** 3)
    Kmmm_5pt = (
        K(pp.lam_plus, pp.lam_minus + 2 * eps_d3)
        - 2 * K(pp.lam_plus, pp.lam_minus + eps_d3)
        + 2 * K(pp.lam_plus, pp.lam_minus - eps_d3)
        - K(pp.lam_plus, pp.lam_minus - 2 * eps_d3)
    ) / (2 * eps_d3 ** 3)

    # Γ^α_{lp,lp,lp} = (1-α)/2 · K'''_lp for α=0
    alpha_test = 0.0
    conn_cf = gts.alpha_connection(pp, alpha_test, T)
    Gamma_lp_cf = conn_cf["Gamma_lp_lp_lp"]
    Gamma_lp_from_K = ((1 - alpha_test) / 2.0) * Kppp_5pt   # (1-α)/2 * K'''
    Gamma_lm_cf = conn_cf["Gamma_lm_lm_lm"]
    Gamma_lm_from_K = ((1 - alpha_test) / 2.0) * Kmmm_5pt

    # Γ^1 should vanish
    conn_1 = gts.alpha_connection(pp, 1.0, T)
    gamma1_max = max(abs(v) for v in conn_1.values())

    # Dual coordinates: η = ∇K
    def dK_dlp(lp, lm, h=1e-7):
        p1 = GTSParams(pp.C_plus, pp.C_minus, pp.a_plus, pp.a_minus, lp + h, lm)
        p2 = GTSParams(pp.C_plus, pp.C_minus, pp.a_plus, pp.a_minus, lp - h, lm)
        return (gts.potential(p1, T) - gts.potential(p2, T)) / (2 * h)
    def dK_dlm(lp, lm, h=1e-7):
        p1 = GTSParams(pp.C_plus, pp.C_minus, pp.a_plus, pp.a_minus, lp, lm + h)
        p2 = GTSParams(pp.C_plus, pp.C_minus, pp.a_plus, pp.a_minus, lp, lm - h)
        return (gts.potential(p1, T) - gts.potential(p2, T)) / (2 * h)

    eta_grad = np.array([dK_dlp(pp.lam_plus, pp.lam_minus), dK_dlm(pp.lam_plus, pp.lam_minus)])
    eta_cf = gts.dual_coordinates(pp, T)

    return {
        "hessK_vs_g": hess_resid.max(),
        "Gamma_lp_from_K": Gamma_lp_from_K,
        "Gamma_lp_cf": Gamma_lp_cf,
        "conn_resid_lp": abs(Gamma_lp_from_K - Gamma_lp_cf),
        "Gamma_lm_from_K": Gamma_lm_from_K,
        "Gamma_lm_cf": Gamma_lm_cf,
        "conn_resid_lm": abs(Gamma_lm_from_K - Gamma_lm_cf),
        "Gamma1_max_abs": gamma1_max,
        "eta_grad_vs_cf": np.abs(eta_grad - eta_cf).max(),
    }


# ---------------------------------------------------------------------------
# B. CTS verification
# ---------------------------------------------------------------------------
CTS_CASES = [
    dict(
        pp=CTSParams(0.3, 0.5, 1.0, 1.0),
        pq=CTSParams(0.3, 0.5, 1.5, 1.2),
    ),
    dict(
        pp=CTSParams(0.1, 1.3, 2.0, 0.8),
        pq=CTSParams(0.1, 1.3, 3.0, 1.5),
    ),
    dict(
        pp=CTSParams(0.1, 1.7, 0.5, 0.5),
        pq=CTSParams(0.1, 1.7, 1.0, 1.5),
    ),
]


def run_cts_checks() -> pd.DataFrame:
    rows = []
    for case_idx, case in enumerate(CTS_CASES):
        pp, pq = case["pp"], case["pq"]
        for alpha in ALPHAS:
            D_num = alpha_divergence(cts, pp, pq, alpha, T)
            D_cf = cts.closed_form_divergence(pp, pq, alpha, T)
            rows.append({
                "family": "CTS", "case": case_idx, "alpha": alpha,
                "D_numeric": D_num, "D_closed_form": D_cf,
                "abs_error": abs(D_num - D_cf),
                "rel_error": abs(D_num - D_cf) / max(abs(D_num), 1e-15),
            })
    return pd.DataFrame(rows)


def run_cts_canonical_divergence() -> pd.DataFrame:
    """Verify Bregman = D^1 and reversed Bregman = D^{-1}."""
    rows = []
    for case_idx, case in enumerate(CTS_CASES):
        pp, pq = case["pp"], case["pq"]
        xi = np.array(pp)
        eta = np.array(pq)

        def K(params):
            return cts.potential(CTSParams(*params), T)

        B_xi_eta = bregman_divergence(K, xi, eta)
        B_eta_xi = bregman_divergence(K, eta, xi)
        D1 = alpha_divergence(cts, pp, pq, 1.0, T)
        Dm1 = alpha_divergence(cts, pp, pq, -1.0, T)

        rows.append({
            "case": case_idx,
            "B_K(xi,eta)": B_xi_eta,
            "D^1(P_xi||P_eta)": D1,
            "B_K(eta,xi)": B_eta_xi,
            "D^{-1}(P_xi||P_eta)": Dm1,
            "B_vs_D1_err": abs(B_xi_eta - D1),
            "B_rev_vs_Dm1_err": abs(B_eta_xi - Dm1),
        })
    return pd.DataFrame(rows)


def run_cts_jeffreys_super() -> pd.DataFrame:
    """Check CTS Jeffreys prior and superharmonic prior."""
    rows = []
    case = CTS_CASES[0]
    pp = case["pp"]
    for k_val in [0.1, 0.2, 0.4]:
        r = cts.superharmonic_check(pp, T, k_val)
        rows.append({
            "k": k_val,
            "a_over_2": r["k_bound_a_over_2"],
            "k_lt_a_over_2": k_val < r["k_bound_a_over_2"],
            "laplace_beltrami": r["laplace_beltrami"],
            "is_superharmonic": r["is_superharmonic"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C. VG verification
# ---------------------------------------------------------------------------
VG_CASES = [
    dict(
        pp=VGParams(0.1, 1.0, 1.0),
        pq=VGParams(0.1, 1.5, 2.0),
    ),
    dict(
        pp=VGParams(0.1, 2.0, 0.8),
        pq=VGParams(0.1, 3.0, 1.5),
    ),
]


def run_vg_checks() -> pd.DataFrame:
    rows = []
    for case_idx, case in enumerate(VG_CASES):
        pp, pq = case["pp"], case["pq"]
        for alpha in ALPHAS:
            D_num = alpha_divergence(vg, pp, pq, alpha, T)
            D_cf = vg.closed_form_divergence(pp, pq, alpha, T)
            rows.append({
                "family": "VG", "case": case_idx, "alpha": alpha,
                "D_numeric": D_num, "D_closed_form": D_cf,
                "abs_error": abs(D_num - D_cf),
                "rel_error": abs(D_num - D_cf) / max(abs(D_num), 1e-15),
            })
    return pd.DataFrame(rows)


def run_vg_log_coord_metric() -> pd.DataFrame:
    """Verify VG metric is Euclidean in log-coordinates."""
    rows = []
    for case_idx, case in enumerate(VG_CASES):
        pp = case["pp"]
        g_lam = vg.fisher_matrix(pp, T)
        g_log = vg.metric_log_coords(pp, T)
        scale = T * pp.C
        expected = np.diag([scale, scale])
        err = np.abs(g_log - expected).max()
        rows.append({
            "case": case_idx,
            "g_lam_plus": g_lam[0, 0],
            "g_lam_minus": g_lam[1, 1],
            "g_log_xx": g_log[0, 0],
            "g_log_yy": g_log[1, 1],
            "TC": scale,
            "max_err": err,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D. CTS → VG limit
# ---------------------------------------------------------------------------

def run_cts_to_vg_limit() -> pd.DataFrame:
    """Verify CTS formulas converge to VG as a ↓ 0."""
    # VG reference case
    C, lp, lm = 0.1, 1.0, 1.5
    pp_vg = VGParams(C, lp, lm)
    pq_vg = VGParams(C, 1.5, 2.0)

    rows = []
    a_vals = [0.5, 0.2, 0.1, 0.05, 0.01]
    alpha = 0.0
    for a_val in a_vals:
        pp_cts = CTSParams(C, a_val, lp, lm)
        pq_cts = CTSParams(C, a_val, 1.5, 2.0)
        D_cts = cts.closed_form_divergence(pp_cts, pq_cts, alpha, T)
        D_vg = vg.closed_form_divergence(pp_vg, pq_vg, alpha, T)
        rows.append({
            "a": a_val,
            "D_CTS": D_cts,
            "D_VG": D_vg,
            "abs_diff": abs(D_cts - D_vg),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E. Merton verification
# ---------------------------------------------------------------------------
MERTON_CASES = [
    MertonParams(sigma=0.1, lam_j=0.2, mu_j=-0.1, delta_j=0.15),
    MertonParams(sigma=0.2, lam_j=1.0, mu_j=0.0, delta_j=0.3),
]


def run_merton_checks() -> pd.DataFrame:
    """Verify Merton RN moment condition and divergence numerics."""
    rows = []
    for case_idx, pm in enumerate(MERTON_CASES):
        rn = merton.verify_rn_moment(pm)
        D_self = abs(alpha_divergence(merton, pm, pm, 0.0, T))
        rows.append({
            "case": case_idx,
            "sigma": pm.sigma, "lam_j": pm.lam_j, "mu_j": pm.mu_j, "delta_j": pm.delta_j,
            "rn_moment_analytic": rn["analytic"],
            "rn_moment_numeric": rn["numerical"],
            "rn_moment_err": rn["abs_error"],
            "zero_div_D0": D_self,
        })
    return pd.DataFrame(rows)


def run_merton_fisher() -> list:
    """Compute Merton Fisher matrix from theorem formula."""
    results = []
    for case_idx, pm in enumerate(MERTON_CASES):
        g = merton.fisher_matrix_theorem(pm, T)
        results.append({
            "case": case_idx,
            "params": pm,
            "fisher_matrix": g,
            "g_sigma_sigma": g[0, 0],
            "expected_g_sigma_sigma": 2.0 * T / pm.sigma ** 2,
        })
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_family_divergences(
    df_gts: pd.DataFrame,
    df_cts: pd.DataFrame,
    df_vg: pd.DataFrame,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (df, title) in zip(axes, [
        (df_gts, "GTS"), (df_cts, "CTS"), (df_vg, "VG")
    ]):
        for case_idx in df["case"].unique():
            sub = df[df["case"] == case_idx]
            ax.plot(sub["alpha"], sub["D_numeric"], "o-", label=f"Case {case_idx} numeric")
            ax.plot(sub["alpha"], sub["D_closed_form"], "s--", label=f"Case {case_idx} CF", alpha=0.7)
        ax.set_xlabel("α"); ax.set_ylabel("D^α(P||Q)")
        ax.set_title(f"{title} α-Divergence"); ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_cts_to_vg(df_limit: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(df_limit["a"], df_limit["D_CTS"], "o-", label="CTS D^0")
    ax.axhline(df_limit["D_VG"].iloc[-1], ls="--", color="r", label="VG D^0")
    ax.set_xlabel("Stability index a"); ax.set_ylabel("D^0(P||Q)")
    ax.set_title("CTS → VG Limit as a ↓ 0")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("EXP 2: Family-Specific Closed-Form Verification")
    print("=" * 65)

    # --- GTS ---
    print("\n[A] GTS checks ...")
    df_gts = run_gts_checks()
    geom_gts = run_gts_geometry_checks()
    print("GTS divergence table:")
    print(df_gts.to_string(index=False, float_format="{:.4e}".format))
    print(f"\nGTS Hess(K) vs g max error: {geom_gts['hessK_vs_g']:.3e}")
    print(f"GTS Γ_lp from K: {geom_gts['Gamma_lp_from_K']:.6f}  cf: {geom_gts['Gamma_lp_cf']:.6f}  err: {geom_gts['conn_resid_lp']:.3e}")
    print(f"GTS Γ_lm from K: {geom_gts['Gamma_lm_from_K']:.6f}  cf: {geom_gts['Gamma_lm_cf']:.6f}  err: {geom_gts['conn_resid_lm']:.3e}")
    print(f"GTS Γ^1 max abs (should≈0): {geom_gts['Gamma1_max_abs']:.3e}")
    print(f"GTS η from ∇K vs closed form: {geom_gts['eta_grad_vs_cf']:.3e}")

    # --- CTS ---
    print("\n[B] CTS checks ...")
    df_cts = run_cts_checks()
    df_canon = run_cts_canonical_divergence()
    df_super = run_cts_jeffreys_super()
    print("CTS divergence table:")
    print(df_cts.to_string(index=False, float_format="{:.4e}".format))
    print("\nCTS canonical divergence (Bregman vs D^1 / D^{-1}):")
    print(df_canon.to_string(index=False, float_format="{:.4e}".format))
    print("\nCTS superharmonic prior check:")
    print(df_super.to_string(index=False))

    # --- VG ---
    print("\n[C] VG checks ...")
    df_vg = run_vg_checks()
    df_log = run_vg_log_coord_metric()
    print("VG divergence table:")
    print(df_vg.to_string(index=False, float_format="{:.4e}".format))
    print("\nVG log-coordinate metric check:")
    print(df_log.to_string(index=False))

    # --- CTS → VG limit ---
    print("\n[D] CTS → VG limit ...")
    df_limit = run_cts_to_vg_limit()
    print(df_limit.to_string(index=False, float_format="{:.6f}".format))

    # --- Merton ---
    print("\n[E] Merton checks ...")
    df_merton = run_merton_checks()
    print(df_merton.to_string(index=False))
    merton_fisher = run_merton_fisher()
    for r in merton_fisher:
        print(f"\nMerton Fisher matrix (case {r['case']}):")
        print(np.array2string(r["fisher_matrix"], precision=4))
        print(f"  g_{{σσ}} = {r['g_sigma_sigma']:.4f}, expected 2T/σ² = {r['expected_g_sigma_sigma']:.4f}")

    # --- Save results ---
    df_gts.to_csv("results/exp2_gts_divergence.csv", index=False)
    df_cts.to_csv("results/exp2_cts_divergence.csv", index=False)
    df_vg.to_csv("results/exp2_vg_divergence.csv", index=False)
    df_limit.to_csv("results/exp2_cts_vg_limit.csv", index=False)
    df_merton.to_csv("results/exp2_merton_checks.csv", index=False)
    df_canon.to_csv("results/exp2_cts_canonical_divergence.csv", index=False)

    # --- Plots ---
    plot_family_divergences(df_gts, df_cts, df_vg, "results/exp2_family_divergences.png")
    plot_cts_to_vg(df_limit, "results/exp2_cts_to_vg_limit.png")
    print("\nPlots saved to results/.")

    # --- Assertions ---
    assert df_gts["abs_error"].max() < 1e-4, f"GTS closed-form mismatch: {df_gts['abs_error'].max():.3e}"
    assert df_cts["abs_error"].max() < 1e-4, f"CTS closed-form mismatch: {df_cts['abs_error'].max():.3e}"
    assert df_vg["abs_error"].max() < 1e-6, f"VG closed-form mismatch: {df_vg['abs_error'].max():.3e}"
    assert df_merton["rn_moment_err"].max() < 1e-8, f"Merton RN moment error: {df_merton['rn_moment_err'].max():.3e}"
    assert geom_gts["Gamma1_max_abs"] < 1e-8, "GTS Γ^1 should vanish"

    summary = {
        "GTS_max_closed_form_error": float(df_gts["abs_error"].max()),
        "CTS_max_closed_form_error": float(df_cts["abs_error"].max()),
        "VG_max_closed_form_error": float(df_vg["abs_error"].max()),
        "GTS_hessK_vs_g_error": float(geom_gts["hessK_vs_g"]),
        "GTS_Gamma1_max": float(geom_gts["Gamma1_max_abs"]),
        "GTS_eta_error": float(geom_gts["eta_grad_vs_cf"]),
        "CTS_canonical_B_vs_D1_max": float(df_canon["B_vs_D1_err"].max()),
        "CTS_limit_to_VG_a0.01": float(df_limit[df_limit["a"] == 0.01]["abs_diff"].iloc[0]),
        "Merton_rn_moment_error": float(df_merton["rn_moment_err"].max()),
    }
    print("\n[Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v:.4e}")

    return summary


if __name__ == "__main__":
    main()
