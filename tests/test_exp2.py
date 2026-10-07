"""
Tests for Experiment 2: Family-Specific Closed-Form Verification.

Covers:
- GTS, CTS, VG, Merton closed-form divergences vs theorem numerics
- Affine-family identities (g = ∇²K, Γ^1 = 0, canonical divergence)
- Dual coordinates, Jeffreys prior, CTS superharmonic prior
- CTS → VG limit
- Merton risk-neutral moment and Fisher matrix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.gts import GTS, GTSParams
from src.cts import CTS, CTSParams
from src.vg import VG, VGParams
from src.merton import Merton, MertonParams
from src.divergence import alpha_divergence
from src.info_geometry import bregman_divergence

gts = GTS()
cts = CTS()
vg = VG()
merton = Merton()

T = 1.0
ALPHAS = [-0.5, 0.0, 0.5]


# ---------------------------------------------------------------------------
# GTS tests
# ---------------------------------------------------------------------------

class TestGTS:
    PP = GTSParams(0.3, 0.2, 0.5, 0.7, 1.0, 1.0)
    PQ = GTSParams(0.3, 0.2, 0.5, 0.7, 1.5, 1.2)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_closed_form_vs_numeric(self, alpha):
        D_num = alpha_divergence(gts, self.PP, self.PQ, alpha, T)
        D_cf = gts.closed_form_divergence(self.PP, self.PQ, alpha, T)
        rel_err = abs(D_num - D_cf) / max(abs(D_num), 1e-15)
        assert rel_err < 0.01, f"GTS CF vs numeric: rel_err={rel_err:.3e}, α={alpha}"

    def test_zero_divergence(self):
        D = gts.closed_form_divergence(self.PP, self.PP, 0.0, T)
        assert abs(D) < 1e-6, f"GTS D(P||P) ≠ 0: {D:.3e}"

    def test_fisher_matrix_diagonal(self):
        """GTS Fisher matrix should be diagonal."""
        g = gts.fisher_matrix(self.PP, T)
        assert g[0, 1] == 0.0, "Off-diagonal entry g[0,1] should be 0"
        assert g[1, 0] == 0.0, "Off-diagonal entry g[1,0] should be 0"

    def test_fisher_matrix_formula(self):
        """Verify g = diag(TC+ Γ(2-a+)/λ+^{2-a+}, TC- Γ(2-a-)/λ-^{2-a-})."""
        import mpmath
        p = self.PP
        g = gts.fisher_matrix(p, T)
        expected_plus = T * p.C_plus * float(mpmath.gamma(2 - p.a_plus)) / p.lam_plus ** (2 - p.a_plus)
        expected_minus = T * p.C_minus * float(mpmath.gamma(2 - p.a_minus)) / p.lam_minus ** (2 - p.a_minus)
        assert abs(g[0, 0] - expected_plus) < 1e-10
        assert abs(g[1, 1] - expected_minus) < 1e-10

    def test_alpha_connection_formula(self):
        """Verify Γ^α_{λ+λ+,λ+} = -(1-α)/2 TC+ Γ(3-a+)/λ+^{3-a+}."""
        import mpmath
        p = self.PP
        alpha = 0.0
        conn = gts.alpha_connection(p, alpha, T)
        expected_lp = -(1 - alpha) / 2 * T * p.C_plus * float(mpmath.gamma(3 - p.a_plus)) / p.lam_plus ** (3 - p.a_plus)
        assert abs(conn["Gamma_lp_lp_lp"] - expected_lp) < 1e-10

    def test_gamma1_vanishes(self):
        """Γ^1 should be 0 (1-connected is flat for affine family)."""
        conn = gts.alpha_connection(self.PP, 1.0, T)
        for v in conn.values():
            assert abs(v) < 1e-14, f"Γ^1 ≠ 0: {v}"

    def test_potential_and_dual_coords(self):
        """η_± = ∂K/∂λ_± verified by finite difference."""
        p = self.PP
        eps = 1e-7
        K0 = gts.potential(p, T)
        p_lp = GTSParams(p.C_plus, p.C_minus, p.a_plus, p.a_minus, p.lam_plus + eps, p.lam_minus)
        p_lm = GTSParams(p.C_plus, p.C_minus, p.a_plus, p.a_minus, p.lam_plus, p.lam_minus + eps)
        eta_num_lp = (gts.potential(p_lp, T) - K0) / eps
        eta_num_lm = (gts.potential(p_lm, T) - K0) / eps
        eta_cf = gts.dual_coordinates(p, T)
        assert abs(eta_num_lp - eta_cf[0]) / max(abs(eta_cf[0]), 1e-10) < 1e-4
        assert abs(eta_num_lm - eta_cf[1]) / max(abs(eta_cf[1]), 1e-10) < 1e-4


# ---------------------------------------------------------------------------
# CTS tests
# ---------------------------------------------------------------------------

class TestCTS:
    PP = CTSParams(0.3, 0.5, 1.0, 1.0)
    PQ = CTSParams(0.3, 0.5, 1.5, 1.2)

    @pytest.mark.parametrize("alpha", ALPHAS + [-1.0, 1.0])
    def test_closed_form_vs_numeric(self, alpha):
        D_num = alpha_divergence(cts, self.PP, self.PQ, alpha, T)
        D_cf = cts.closed_form_divergence(self.PP, self.PQ, alpha, T)
        rel_err = abs(D_num - D_cf) / max(abs(D_num), 1e-15)
        assert rel_err < 0.01, f"CTS CF vs numeric: rel_err={rel_err:.3e}, α={alpha}"

    def test_bregman_equals_D1(self):
        """B_K(ξ, η) = D^1(P_ξ||P_η) for CTS."""
        xi = np.array(self.PP)
        eta = np.array(self.PQ)
        def K(p): return cts.potential(CTSParams(*p), T)
        B = bregman_divergence(K, xi, eta)
        D1 = alpha_divergence(cts, self.PP, self.PQ, 1.0, T)
        assert abs(B - D1) / max(D1, 1e-10) < 0.01, f"B_K vs D^1: {abs(B-D1):.3e}"

    def test_reversed_bregman_equals_Dm1(self):
        """B_K(η, ξ) = D^{-1}(P_ξ||P_η) for CTS."""
        xi = np.array(self.PP)
        eta = np.array(self.PQ)
        def K(p): return cts.potential(CTSParams(*p), T)
        B_rev = bregman_divergence(K, eta, xi)
        Dm1 = alpha_divergence(cts, self.PP, self.PQ, -1.0, T)
        assert abs(B_rev - Dm1) / max(Dm1, 1e-10) < 0.01, f"B_K(η,ξ) vs D^{{-1}}: {abs(B_rev-Dm1):.3e}"

    def test_gamma1_vanishes(self):
        conn = cts.alpha_connection(self.PP, 1.0, T)
        for v in conn.values():
            assert abs(v) < 1e-14

    def test_jeffreys_prior_positive(self):
        J = cts.jeffreys_prior(self.PP, T)
        assert J > 0, "Jeffreys prior should be positive"

    @pytest.mark.parametrize("k_val, a, should_be_superharmonic", [
        (0.1, 0.5, True),   # k=0.1 < a/2=0.25: superharmonic
        (0.3, 0.5, False),  # k=0.3 > a/2=0.25: not superharmonic
        (0.2, 0.7, True),   # k=0.2 < a/2=0.35
    ])
    def test_superharmonic_prior(self, k_val, a, should_be_superharmonic):
        p = CTSParams(C=0.3, a=a, lam_plus=1.0, lam_minus=1.0)
        result = cts.superharmonic_check(p, T, k_val)
        assert result["is_superharmonic"] == should_be_superharmonic, \
            f"k={k_val}, a={a}: expected superharmonic={should_be_superharmonic}, got {result['is_superharmonic']}"

    def test_dual_coords_from_gradient(self):
        p = self.PP
        eps = 1e-7
        K0 = cts.potential(p, T)
        p_lp = CTSParams(p.C, p.a, p.lam_plus + eps, p.lam_minus)
        eta_num = (cts.potential(p_lp, T) - K0) / eps
        eta_cf = cts.dual_coordinates(p, T)
        assert abs(eta_num - eta_cf[0]) / max(abs(eta_cf[0]), 1e-10) < 1e-4


# ---------------------------------------------------------------------------
# VG tests
# ---------------------------------------------------------------------------

class TestVG:
    PP = VGParams(0.1, 1.0, 1.0)
    PQ = VGParams(0.1, 1.5, 2.0)

    @pytest.mark.parametrize("alpha", ALPHAS + [-1.0, 1.0])
    def test_closed_form_vs_numeric(self, alpha):
        D_num = alpha_divergence(vg, self.PP, self.PQ, alpha, T)
        D_cf = vg.closed_form_divergence(self.PP, self.PQ, alpha, T)
        assert abs(D_num - D_cf) < 1e-6, f"VG CF vs numeric: err={abs(D_num-D_cf):.3e}, α={alpha}"

    def test_fisher_exact_values(self):
        p = self.PP
        g = vg.fisher_matrix(p, T)
        assert abs(g[0, 0] - T * p.C / p.lam_plus ** 2) < 1e-12
        assert abs(g[1, 1] - T * p.C / p.lam_minus ** 2) < 1e-12
        assert abs(g[0, 1]) < 1e-14

    def test_log_coord_metric_euclidean(self):
        g_log = vg.metric_log_coords(self.PP, T)
        scale = T * self.PP.C
        assert abs(g_log[0, 0] - scale) < 1e-14
        assert abs(g_log[1, 1] - scale) < 1e-14
        assert abs(g_log[0, 1]) < 1e-14

    def test_potential_and_dual_coords(self):
        p = self.PP
        eta = vg.dual_coordinates(p, T)
        expected_plus = -T * p.C / p.lam_plus
        expected_minus = -T * p.C / p.lam_minus
        assert abs(eta[0] - expected_plus) < 1e-12
        assert abs(eta[1] - expected_minus) < 1e-12

    def test_jeffreys_prior_formula(self):
        p = self.PP
        J = vg.jeffreys_prior(p, T)
        expected = T * p.C / (p.lam_plus * p.lam_minus)
        assert abs(J - expected) < 1e-12

    def test_gamma1_vanishes(self):
        conn = vg.alpha_connection(self.PP, 1.0, T)
        for v in conn.values():
            assert abs(v) < 1e-14

    def test_bregman_equals_D1(self):
        xi = np.array(self.PP)
        eta = np.array(self.PQ)
        def K(p): return vg.potential(VGParams(*p), T)
        B = bregman_divergence(K, xi, eta)
        D1 = alpha_divergence(vg, self.PP, self.PQ, 1.0, T)
        assert abs(B - D1) / max(D1, 1e-10) < 0.01


# ---------------------------------------------------------------------------
# CTS → VG limit
# ---------------------------------------------------------------------------

class TestCTSToVGLimit:
    def test_divergence_converges(self):
        C, lp, lm = 0.1, 1.0, 1.5
        pp_vg = VGParams(C, lp, lm)
        pq_vg = VGParams(C, 1.5, 2.0)
        D_vg = vg.closed_form_divergence(pp_vg, pq_vg, 0.0, T)

        a_small = 0.01
        pp_cts = CTSParams(C, a_small, lp, lm)
        pq_cts = CTSParams(C, a_small, 1.5, 2.0)
        D_cts = cts.closed_form_divergence(pp_cts, pq_cts, 0.0, T)

        rel_err = abs(D_cts - D_vg) / max(abs(D_vg), 1e-15)
        assert rel_err < 0.05, f"CTS(a=0.01) did not converge to VG: rel_err={rel_err:.4f}"

    def test_metric_converges(self):
        C, lp, lm = 0.1, 1.0, 1.5
        g_vg = vg.fisher_matrix(VGParams(C, lp, lm), T)
        g_cts = cts.fisher_matrix(CTSParams(C, 0.01, lp, lm), T)
        rel_err = np.abs(g_cts - g_vg) / (np.abs(g_vg) + 1e-15)
        assert rel_err.max() < 0.1, f"CTS metric did not converge to VG: {rel_err.max():.4f}"


# ---------------------------------------------------------------------------
# Merton tests
# ---------------------------------------------------------------------------

class TestMerton:
    PM = MertonParams(sigma=0.1, lam_j=0.5, mu_j=0.0, delta_j=0.2)

    def test_rn_moment_condition(self):
        result = merton.verify_rn_moment(self.PM)
        assert result["abs_error"] < 1e-8, \
            f"Merton RN moment error: {result['abs_error']:.3e}"

    def test_levy_density_positive(self):
        x_vals = np.linspace(-1.0, 1.0, 20)
        for x in x_vals:
            nu = merton.levy_density(x, self.PM)
            assert nu > 0, f"Merton Lévy density ≤ 0 at x={x}: {nu}"

    def test_levy_density_integrates_to_lambda(self):
        """∫ ν(dx) = λ_j (total jump intensity)."""
        from scipy import integrate
        p = self.PM
        result, _ = integrate.quad(
            lambda x: merton.levy_density(x, p),
            p.mu_j - 20 * p.delta_j,
            p.mu_j + 20 * p.delta_j,
        )
        assert abs(result - p.lam_j) / p.lam_j < 1e-6, \
            f"∫ν ≠ λ_j: {result:.6f} vs {p.lam_j}"

    def test_fisher_sigma_entry(self):
        """g_{σσ} = 2T/σ² (Wiener component)."""
        g = merton.fisher_matrix_theorem(self.PM, T)
        expected = 2.0 * T / self.PM.sigma ** 2
        assert abs(g[0, 0] - expected) < 1e-10, \
            f"g_{{σσ}} mismatch: {g[0,0]:.6f} vs {expected:.6f}"

    def test_zero_divergence_merton(self):
        D = abs(alpha_divergence(merton, self.PM, self.PM, 0.0, T))
        assert D < 1e-6, f"Merton D(P||P) ≠ 0: {D:.3e}"


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_gts_invalid_a(self):
        with pytest.raises(AssertionError):
            p = GTSParams(0.3, 0.2, 1.0, 0.7, 1.0, 1.0)  # a_plus=1 invalid
            gts.fisher_matrix(p, T)

    def test_gts_negative_C(self):
        with pytest.raises(AssertionError):
            p = GTSParams(-0.1, 0.2, 0.5, 0.7, 1.0, 1.0)
            gts.fisher_matrix(p, T)

    def test_cts_invalid_params(self):
        with pytest.raises(AssertionError):
            p = CTSParams(C=0, a=0.5, lam_plus=1.0, lam_minus=1.0)  # C=0
            cts.fisher_matrix(p, T)

    def test_vg_negative_lambda(self):
        with pytest.raises(AssertionError):
            p = VGParams(C=0.1, lam_plus=-1.0, lam_minus=1.0)
            vg.fisher_matrix(p, T)

    def test_merton_zero_sigma(self):
        with pytest.raises(AssertionError):
            p = MertonParams(sigma=0.0, lam_j=1.0, mu_j=0.0, delta_j=0.2)
            merton.fisher_matrix_theorem(p, T)
