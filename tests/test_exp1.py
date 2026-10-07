"""
Tests for Experiment 1: General Lévy-process α-divergence and information geometry.

Tests verify:
- α-duality: D^α(P||Q) = D^{-α}(Q||P)
- Zero divergence at equal parameters: D^α(P||P) = 0
- Fisher matrix symmetry
- Divergence positivity
- Endpoint formulas (α = ±1)
- Edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.gts import GTS, GTSParams
from src.cts import CTS, CTSParams
from src.vg import VG, VGParams
from src.divergence import (
    alpha_divergence, verify_duality, verify_zero_divergence,
    delta_T_alpha, compute_psi_Q, compute_drift_correction,
)
from src.info_geometry import fisher_information_numerical, check_fisher_symmetry

gts = GTS()
cts = CTS()
vg = VG()

# Standard test parameters
PP_GTS = GTSParams(C_plus=0.3, C_minus=0.2, a_plus=0.5, a_minus=0.7, lam_plus=1.0, lam_minus=1.0)
PQ_GTS = GTSParams(C_plus=0.3, C_minus=0.2, a_plus=0.5, a_minus=0.7, lam_plus=1.5, lam_minus=1.2)

PP_VG = VGParams(C=0.1, lam_plus=1.0, lam_minus=1.0)
PQ_VG = VGParams(C=0.1, lam_plus=1.5, lam_minus=2.0)

T = 1.0
ALPHA_VALS = [-1.0, -0.5, 0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Tests: zero divergence D^α(P||P) = 0
# ---------------------------------------------------------------------------

class TestZeroDivergence:
    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_gts_zero_divergence(self, alpha):
        err = verify_zero_divergence(gts, PP_GTS, alpha, T)
        assert err < 1e-6, f"GTS D^α(P||P)≠0: err={err:.3e}, α={alpha}"

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_cts_zero_divergence(self, alpha):
        pp = CTSParams(C=0.3, a=0.5, lam_plus=1.0, lam_minus=1.0)
        err = verify_zero_divergence(cts, pp, alpha, T)
        assert err < 1e-6, f"CTS D^α(P||P)≠0: err={err:.3e}"

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_vg_zero_divergence(self, alpha):
        err = verify_zero_divergence(vg, PP_VG, alpha, T)
        assert err < 1e-8, f"VG D^α(P||P)≠0: err={err:.3e}"


# ---------------------------------------------------------------------------
# Tests: α-duality D^α(P||Q) = D^{-α}(Q||P)
# ---------------------------------------------------------------------------

class TestAlphaDuality:
    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_gts_duality(self, alpha):
        result = verify_duality(gts, PP_GTS, PQ_GTS, alpha, T)
        assert result["abs_error"] < 1e-4, \
            f"GTS duality failed: err={result['abs_error']:.3e}, α={alpha}"

    @pytest.mark.parametrize("alpha", [-0.5, 0.0, 0.5])
    def test_vg_duality(self, alpha):
        result = verify_duality(vg, PP_VG, PQ_VG, alpha, T)
        assert result["abs_error"] < 1e-6, \
            f"VG duality failed: err={result['abs_error']:.3e}, α={alpha}"

    def test_duality_at_endpoints_gts(self):
        """α=1 duality: D^1(P||Q) = D^{-1}(Q||P)."""
        D1 = alpha_divergence(gts, PP_GTS, PQ_GTS, 1.0, T)
        Dm1_rev = alpha_divergence(gts, PQ_GTS, PP_GTS, -1.0, T)
        assert abs(D1 - Dm1_rev) < 1e-4, f"|D^1 - D^{{-1}}(Q||P)| = {abs(D1-Dm1_rev):.3e}"


# ---------------------------------------------------------------------------
# Tests: positivity of divergence
# ---------------------------------------------------------------------------

class TestDivergencePositivity:
    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_gts_positivity(self, alpha):
        D = alpha_divergence(gts, PP_GTS, PQ_GTS, alpha, T)
        assert D >= -1e-10, f"GTS D^α < 0: D={D:.4f}, α={alpha}"

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_vg_positivity(self, alpha):
        D = alpha_divergence(vg, PP_VG, PQ_VG, alpha, T)
        assert D >= -1e-10, f"VG D^α < 0: D={D:.4f}, α={alpha}"


# ---------------------------------------------------------------------------
# Tests: psi_Q computation
# ---------------------------------------------------------------------------

class TestPsiQ:
    def test_psi_q_positive_axis(self):
        x = 0.5
        psi = compute_psi_Q(x, gts, PP_GTS, PQ_GTS)
        expected = -(PP_GTS.lam_plus - PQ_GTS.lam_plus) * x
        assert abs(psi - expected) < 1e-12, f"ψ_Q mismatch at x={x}: {psi} vs {expected}"

    def test_psi_q_negative_axis(self):
        x = -0.5
        psi = compute_psi_Q(x, gts, PP_GTS, PQ_GTS)
        expected = -(PP_GTS.lam_minus - PQ_GTS.lam_minus) * abs(x)
        assert abs(psi - expected) < 1e-12, f"ψ_Q mismatch at x={x}: {psi} vs {expected}"

    def test_psi_q_equal_params(self):
        x = 1.0
        psi = compute_psi_Q(x, gts, PP_GTS, PP_GTS)
        assert abs(psi) < 1e-14, f"ψ_Q should be 0 for equal params: {psi}"


# ---------------------------------------------------------------------------
# Tests: drift correction (σ=0 case)
# ---------------------------------------------------------------------------

class TestDriftCorrection:
    def test_zero_for_sigma_zero(self):
        c = compute_drift_correction(gts, PP_GTS, PQ_GTS, T)
        assert c == 0.0, f"c should be 0 for σ=0 family: {c}"


# ---------------------------------------------------------------------------
# Tests: Fisher information matrix
# ---------------------------------------------------------------------------

class TestFisherInformation:
    def test_gts_fisher_symmetry(self):
        class GTSLambdaOnly(GTS):
            def __init__(self, C_plus, C_minus, a_plus, a_minus):
                self.C_plus = C_plus; self.C_minus = C_minus
                self.a_plus = a_plus; self.a_minus = a_minus
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

        fam = GTSLambdaOnly(0.3, 0.2, 0.5, 0.7)
        xi = np.array([1.0, 1.0])
        g = fisher_information_numerical(fam, xi, T, eps=1e-4)
        assert check_fisher_symmetry(g, tol=1e-4), f"Fisher not symmetric: {g}"

    def test_gts_fisher_positive_diagonal(self):
        g = gts.fisher_matrix(PP_GTS, T)
        assert g[0, 0] > 0, "g_{λ+λ+} must be positive"
        assert g[1, 1] > 0, "g_{λ-λ-} must be positive"

    def test_vg_fisher_formula(self):
        g = vg.fisher_matrix(PP_VG, T)
        expected_plus = T * PP_VG.C / PP_VG.lam_plus ** 2
        expected_minus = T * PP_VG.C / PP_VG.lam_minus ** 2
        assert abs(g[0, 0] - expected_plus) < 1e-12
        assert abs(g[1, 1] - expected_minus) < 1e-12


# ---------------------------------------------------------------------------
# Tests: delta_T_alpha properties
# ---------------------------------------------------------------------------

class TestDeltaT:
    def test_delta_non_negative(self):
        """Δ_T^α(P||Q) ≥ 0 (since it enters 1-exp(-Δ) to give D^α ≥ 0)."""
        for alpha in [-0.5, 0.0, 0.5]:
            delta = delta_T_alpha(gts, PP_GTS, PQ_GTS, alpha, T)
            assert delta >= -1e-10, f"Δ < 0: {delta:.4f}, α={alpha}"

    def test_delta_zero_at_equal_params(self):
        for alpha in [-0.5, 0.0, 0.5]:
            delta = delta_T_alpha(gts, PP_GTS, PP_GTS, alpha, T)
            assert abs(delta) < 1e-8, f"Δ(P||P) ≠ 0: {delta:.4e}, α={alpha}"

    def test_delta_scales_with_T(self):
        """Δ_T^α should scale linearly with T."""
        d1 = delta_T_alpha(gts, PP_GTS, PQ_GTS, 0.0, 1.0)
        d2 = delta_T_alpha(gts, PP_GTS, PQ_GTS, 0.0, 2.0)
        assert abs(d2 / d1 - 2.0) < 1e-4, f"Linear scaling violated: {d2/d1:.4f}"


# ---------------------------------------------------------------------------
# Tests: admissibility check
# ---------------------------------------------------------------------------

class TestAdmissibility:
    def test_different_sigma_raises(self):
        from src.merton import Merton, MertonParams
        m = Merton()
        p1 = MertonParams(sigma=0.1, lam_j=1.0, mu_j=0.0, delta_j=0.2)
        p2 = MertonParams(sigma=0.2, lam_j=1.0, mu_j=0.0, delta_j=0.2)
        with pytest.raises(ValueError, match="Admissibility"):
            m.admissibility_check(p1, p2)

    def test_same_sigma_passes(self):
        from src.merton import Merton, MertonParams
        m = Merton()
        p1 = MertonParams(sigma=0.1, lam_j=0.5, mu_j=0.0, delta_j=0.2)
        p2 = MertonParams(sigma=0.1, lam_j=1.0, mu_j=0.1, delta_j=0.3)
        assert m.admissibility_check(p1, p2) is True


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_alpha_near_endpoint_continuity(self):
        """D^α should be continuous near α = ±1."""
        alpha_near_1 = 1.0 - 1e-6
        D_near = alpha_divergence(gts, PP_GTS, PQ_GTS, alpha_near_1, T)
        D_at = alpha_divergence(gts, PP_GTS, PQ_GTS, 1.0, T)
        assert abs(D_near - D_at) / max(D_at, 1e-10) < 0.01, \
            f"D^α discontinuous near α=1: {D_near:.6f} vs {D_at:.6f}"

    def test_large_lambda_difference(self):
        """With large λ difference, D should still be finite."""
        pp = GTSParams(0.3, 0.2, 0.5, 0.7, 1.0, 1.0)
        pq = GTSParams(0.3, 0.2, 0.5, 0.7, 10.0, 8.0)
        D = alpha_divergence(gts, pp, pq, 0.0, T)
        assert np.isfinite(D), f"D not finite for large Δλ: {D}"

    def test_vg_single_param(self):
        """VG with λ+ = λ-."""
        pp = VGParams(C=0.1, lam_plus=2.0, lam_minus=2.0)
        pq = VGParams(C=0.1, lam_plus=3.0, lam_minus=3.0)
        for alpha in [-0.5, 0.0, 0.5]:
            D = alpha_divergence(vg, pp, pq, alpha, T)
            assert D > 0 and np.isfinite(D)
