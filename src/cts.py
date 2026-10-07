"""
Classical Tempered Stable (CTS) / CGMY Lévy family.

Parameters ξ = (C, a, λ_+, λ_-) with C > 0, a ∈ (0,2)\\{1}, λ_± > 0, σ = 0.
This is the symmetric-intensity specialisation of GTS with C_+ = C_- = C and
a_+ = a_- = a.

Closed-form formulas from the paper's Eq. (94) structure.
Custom implementation — Context7 confirmed no library provides CTS
information-geometry formulas.
"""

from __future__ import annotations

from typing import NamedTuple

import mpmath
import numpy as np
from scipy.special import gamma as sp_gamma

from src.levy_base import LevyFamily


class CTSParams(NamedTuple):
    """Parameter container for the CTS/CGMY family."""
    C: float         # intensity (same for positive and negative)
    a: float         # stability index ∈ (0,2)\{1}
    lam_plus: float  # tempering rate for positive jumps
    lam_minus: float # tempering rate for negative jumps


def _validate_cts(p: CTSParams) -> None:
    assert p.C > 0, "C must be positive"
    assert 0 < p.a < 2 and p.a != 1, "a must be in (0,2)\\{1}"
    assert p.lam_plus > 0 and p.lam_minus > 0, "λ_± must be positive"


def _gamma_neg(a: float) -> float:
    """Γ(-a) via mpmath for a ∈ (0,2)\\{1}."""
    return float(mpmath.gamma(-a))


class CTS(LevyFamily):
    """Classical Tempered Stable / CGMY Lévy family (σ = 0).

    Lévy density:
        ν(x; ξ) = C exp(-λ_+ x)/x^{a+1}  for x > 0
                  C exp(-λ_- |x|)/|x|^{a+1} for x < 0
    """

    def levy_density(self, x: float, params) -> float:
        p = CTSParams(*params)
        if x > 0:
            return p.C * np.exp(-p.lam_plus * x) / (x ** (p.a + 1))
        elif x < 0:
            ax = abs(x)
            return p.C * np.exp(-p.lam_minus * ax) / (ax ** (p.a + 1))
        return 0.0

    def log_levy_density(self, x: float, params) -> float:
        p = CTSParams(*params)
        if x > 0:
            return np.log(p.C) - p.lam_plus * x - (p.a + 1) * np.log(x)
        elif x < 0:
            ax = abs(x)
            return np.log(p.C) - p.lam_minus * ax - (p.a + 1) * np.log(ax)
        return -np.inf

    def drift(self, params) -> float:
        return 0.0

    def diffusion(self, params) -> float:
        return 0.0

    def param_names(self) -> list[str]:
        return ["C", "a", "lam_plus", "lam_minus"]

    # ------------------------------------------------------------------
    # Closed-form Δ using gamma identities (same derivation as GTS)
    # ------------------------------------------------------------------

    def closed_form_delta(self, params_p, params_q, alpha: float, T: float) -> float:
        """Δ_T^α for CTS with shared C and a."""
        p_p = CTSParams(*params_p)
        p_q = CTSParams(*params_q)
        _validate_cts(p_q)

        alpha_a = (1.0 - alpha) / 2.0
        alpha_b = (1.0 + alpha) / 2.0
        G_neg_a = _gamma_neg(p_q.a)

        # Positive side: κ = α_b λ_+^Q + α_a λ_+^P
        kappa_p = alpha_b * p_q.lam_plus + alpha_a * p_p.lam_plus
        I_plus = p_q.C * G_neg_a * (
            alpha_a * (p_p.lam_plus ** p_q.a)
            + alpha_b * (p_q.lam_plus ** p_q.a)
            - kappa_p ** p_q.a
        )

        # Negative side
        kappa_m = alpha_b * p_q.lam_minus + alpha_a * p_p.lam_minus
        I_minus = p_q.C * G_neg_a * (
            alpha_a * (p_p.lam_minus ** p_q.a)
            + alpha_b * (p_q.lam_minus ** p_q.a)
            - kappa_m ** p_q.a
        )

        return T * (I_plus + I_minus)

    def closed_form_divergence(self, params_p, params_q, alpha: float, T: float) -> float:
        """CTS α-divergence via closed-form gamma integrals."""
        _AEP = 1e-9
        if abs(alpha - 1.0) < _AEP:
            return self._kl_endpoint(params_p, params_q, T, direction=1)
        if abs(alpha + 1.0) < _AEP:
            return self._kl_endpoint(params_p, params_q, T, direction=-1)

        delta = self.closed_form_delta(params_p, params_q, alpha, T)
        return (4.0 / (1.0 - alpha ** 2)) * (1.0 - np.exp(-delta))

    def _kl_endpoint(self, params_p, params_q, T: float, direction: int) -> float:
        if direction == -1:
            p_p = CTSParams(*params_p)
            p_q = CTSParams(*params_q)
            G_neg_a = _gamma_neg(p_q.a)
            Γ_1ma = float(mpmath.gamma(1 - p_q.a))
            Dp = p_p.lam_plus - p_q.lam_plus
            Dm = p_p.lam_minus - p_q.lam_minus
            I_plus = p_q.C * (
                (-Dp) * Γ_1ma / (p_p.lam_plus ** (1 - p_q.a))
                - G_neg_a * ((p_p.lam_plus ** p_q.a) - (p_q.lam_plus ** p_q.a))
            )
            I_minus = p_q.C * (
                (-Dm) * Γ_1ma / (p_p.lam_minus ** (1 - p_q.a))
                - G_neg_a * ((p_p.lam_minus ** p_q.a) - (p_q.lam_minus ** p_q.a))
            )
            return T * (I_plus + I_minus)
        else:
            return self._kl_endpoint(params_q, params_p, T, direction=-1)

    # ------------------------------------------------------------------
    # Fisher information matrix (diagonal)
    # ------------------------------------------------------------------

    def fisher_matrix(self, params, T: float) -> np.ndarray:
        """g = diag(T C Γ(2-a)/λ_+^{2-a}, T C Γ(2-a)/λ_-^{2-a})."""
        p = CTSParams(*params)
        _validate_cts(p)
        Γ_2ma = float(mpmath.gamma(2 - p.a))
        g_plus = T * p.C * Γ_2ma / (p.lam_plus ** (2 - p.a))
        g_minus = T * p.C * Γ_2ma / (p.lam_minus ** (2 - p.a))
        return np.diag([g_plus, g_minus])

    # ------------------------------------------------------------------
    # α-connection (nonzero diagonal cubic components)
    # ------------------------------------------------------------------

    def alpha_connection(self, params, alpha: float, T: float) -> dict:
        """Γ^α_{λ±λ±,λ±} = -(1-α)/2 · T C Γ(3-a)/λ_±^{3-a}."""
        p = CTSParams(*params)
        _validate_cts(p)
        coeff = -(1.0 - alpha) / 2.0
        Γ_3ma = float(mpmath.gamma(3 - p.a))
        return {
            "Gamma_lp_lp_lp": coeff * T * p.C * Γ_3ma / (p.lam_plus ** (3 - p.a)),
            "Gamma_lm_lm_lm": coeff * T * p.C * Γ_3ma / (p.lam_minus ** (3 - p.a)),
        }

    # ------------------------------------------------------------------
    # Potential, dual coordinates
    # ------------------------------------------------------------------

    def potential(self, params, T: float) -> float:
        """K_CTS = T C Γ(-a)(λ_+^a + λ_-^a)."""
        p = CTSParams(*params)
        _validate_cts(p)
        G_neg_a = _gamma_neg(p.a)
        return float(T * p.C * G_neg_a * (p.lam_plus ** p.a + p.lam_minus ** p.a))

    def dual_coordinates(self, params, T: float) -> np.ndarray:
        """η_± = -T C Γ(1-a) λ_±^{a-1}."""
        p = CTSParams(*params)
        _validate_cts(p)
        Γ_1ma = float(mpmath.gamma(1 - p.a))
        η_plus = -T * p.C * Γ_1ma * (p.lam_plus ** (p.a - 1))
        η_minus = -T * p.C * Γ_1ma * (p.lam_minus ** (p.a - 1))
        return np.array([η_plus, η_minus])

    # ------------------------------------------------------------------
    # Jeffreys and superharmonic priors
    # ------------------------------------------------------------------

    def jeffreys_prior(self, params, T: float) -> float:
        """J(ξ) ∝ T C Γ(2-a) / sqrt((λ_+ λ_-)^{2-a}).

        Returns the unnormalised density value (up to a constant).
        """
        p = CTSParams(*params)
        _validate_cts(p)
        Γ_2ma = float(mpmath.gamma(2 - p.a))
        return T * p.C * Γ_2ma / np.sqrt((p.lam_plus * p.lam_minus) ** (2 - p.a))

    def superharmonic_check(self, params, T: float, k: float) -> dict:
        """Check superharmonicity of ρ(λ_+, λ_-) = λ_+^k λ_-^k under the CTS metric.

        The Laplace–Beltrami operator under diagonal metric g = diag(g_+, g_-) is:
            Δ_g ρ = (1/sqrt|g|) ∂_i (sqrt|g| g^{ij} ∂_j ρ)
        For diagonal g with g_± ∝ λ_±^{-(2-a)}:
            Δ_g ρ = sum_± (1/g_±) [ρ_{λ±λ±} - (∂_{λ±} log g_±)/2 · ρ_{λ±}]
        ρ = λ_+^k λ_-^k is separable; condition is k < a/2.
        """
        p = CTSParams(*params)
        _validate_cts(p)
        g = self.fisher_matrix(params, T)
        g_plus, g_minus = g[0, 0], g[1, 1]

        # Second derivatives of ρ
        rho = p.lam_plus ** k * p.lam_minus ** k
        # ∂²ρ/∂λ+² = k(k-1) λ_+^{k-2} λ_-^k
        d2rho_plus = k * (k - 1) * (p.lam_plus ** (k - 2)) * (p.lam_minus ** k)
        d2rho_minus = k * (k - 1) * (p.lam_minus ** (k - 2)) * (p.lam_plus ** k)
        # ∂ρ/∂λ±
        drho_plus = k * (p.lam_plus ** (k - 1)) * (p.lam_minus ** k)
        drho_minus = k * (p.lam_minus ** (k - 1)) * (p.lam_plus ** k)

        # ∂ log g_± / ∂λ± = -(2-a)/λ±
        d_log_gp = -(2 - p.a) / p.lam_plus
        d_log_gm = -(2 - p.a) / p.lam_minus

        # Laplace–Beltrami: sum ± (1/g_±)(ρ'' - (log g_±)'/2 · ρ')
        lbo_plus = (d2rho_plus - 0.5 * d_log_gp * drho_plus) / g_plus
        lbo_minus = (d2rho_minus - 0.5 * d_log_gm * drho_minus) / g_minus
        lbo_total = lbo_plus + lbo_minus

        return {
            "k": k,
            "rho": rho,
            "laplace_beltrami": lbo_total,
            "is_superharmonic": lbo_total <= 0,
            "k_bound_a_over_2": p.a / 2.0,
        }
