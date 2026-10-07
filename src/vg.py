"""
Variance Gamma (VG) Lévy family.

Parameters ξ = (C, λ_+, λ_-) with C > 0, λ_± > 0, σ = 0.
The VG Lévy density has a 1/|x| singularity; integrals use the Frullani identity
to avoid numerical instability near zero.

Closed-form formulas from the paper's Eq. (114) structure.
Custom implementation — Context7 confirmed no library provides VG
information-geometry formulas.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.levy_base import LevyFamily


class VGParams(NamedTuple):
    """Parameter container for the VG family."""
    C: float         # intensity coefficient
    lam_plus: float  # tempering rate for positive jumps
    lam_minus: float # tempering rate for negative jumps


def _validate_vg(p: VGParams) -> None:
    assert p.C > 0, "C must be positive"
    assert p.lam_plus > 0 and p.lam_minus > 0, "λ_± must be positive"


class VG(LevyFamily):
    """Variance Gamma Lévy family (σ = 0, a → 0 limit of CTS).

    Lévy density:
        ν(x; ξ) = C exp(-λ_+ x) / x   for x > 0
                  C exp(-λ_- |x|) / |x| for x < 0
    """

    def levy_density(self, x: float, params) -> float:
        p = VGParams(*params)
        if x > 0:
            return p.C * np.exp(-p.lam_plus * x) / x
        elif x < 0:
            ax = abs(x)
            return p.C * np.exp(-p.lam_minus * ax) / ax
        return 0.0

    def log_levy_density(self, x: float, params) -> float:
        p = VGParams(*params)
        if x > 0:
            return np.log(p.C) - p.lam_plus * x - np.log(x)
        elif x < 0:
            ax = abs(x)
            return np.log(p.C) - p.lam_minus * ax - np.log(ax)
        return -np.inf

    def drift(self, params) -> float:
        return 0.0

    def diffusion(self, params) -> float:
        return 0.0

    def param_names(self) -> list[str]:
        return ["C", "lam_plus", "lam_minus"]

    # ------------------------------------------------------------------
    # Closed-form Δ using Frullani identity
    # ------------------------------------------------------------------

    def closed_form_delta(self, params_p, params_q, alpha: float, T: float) -> float:
        """Δ_T^α via Frullani identity.

        For the + side (with Q: λ_+^Q, P: λ_+^P):
            Δ_+  =  T C [-(1-α)/2 log λ_+^P - (1+α)/2 log λ_+^Q
                         + log((1-α)/2 · λ_+^P + (1+α)/2 · λ_+^Q)]

        Derivation: ∫_0^∞ [a e^{-ψ} + b - e^{-aψ}] C e^{-λ^Q x}/x dx
        with ψ = -(λ^P - λ^Q)x = (λ^Q - λ^P)x actually ψ_Q = log(ν_P/ν_Q).
        Here ν_P/ν_Q = exp(-(λ^P - λ^Q)x) so ψ_Q = -(λ^P-λ^Q)x.

        Wait – careful re-check:
          ψ_Q(x) = log(dν_P/dν_Q)(x) for x>0 = log(λ^P/λ^Q ... no:
          ν_P = C exp(-λ^P x)/x, ν_Q = C exp(-λ^Q x)/x
          ψ_Q = log(ν_P/ν_Q) = (λ^Q - λ^P)x

        The integrand of Δ is:
          [(1-α)/2 exp(ψ_Q) + (1+α)/2 - exp((1-α)/2 · ψ_Q)] ν_Q(x)
          = [a exp((λ^Q-λ^P)x) + b - exp(a(λ^Q-λ^P)x)] C exp(-λ^Q x)/x
          = C [a exp(-(λ^P)x) + b exp(-λ^Q x) - exp(-(λ^Q + a(λ^P-λ^Q))x)] / x
          = C [a exp(-λ^P x) + b exp(-λ^Q x) - exp(-κ x)] / x

        where κ = λ^Q + a(λ^P - λ^Q) = b λ^Q + a λ^P, a=(1-α)/2, b=(1+α)/2.

        By Frullani: ∫_0^∞ (e^{-ux} - e^{-vx})/x dx = log(v/u), so:
          ∫_0^∞ [a e^{-λ^P x} + b e^{-λ^Q x} - e^{-κ x}] / x dx
        Note a + b = 1, so this is:
          a log(κ/λ^P) + b log(κ/λ^Q) - 0  ... need to decompose:
          = a [log κ - log λ^P] + b [log κ - log λ^Q]
          = log κ - a log λ^P - b log λ^Q.

        So Δ_+^α = T C [log(b λ^Q + a λ^P) - a log λ^P - b log λ^Q].
        """
        p_p = VGParams(*params_p)
        p_q = VGParams(*params_q)
        _validate_vg(p_q)

        alpha_a = (1.0 - alpha) / 2.0
        alpha_b = (1.0 + alpha) / 2.0

        # Positive side
        kappa_p = alpha_b * p_q.lam_plus + alpha_a * p_p.lam_plus
        delta_plus = (
            np.log(kappa_p)
            - alpha_a * np.log(p_p.lam_plus)
            - alpha_b * np.log(p_q.lam_plus)
        )

        # Negative side (same formula with λ_-)
        kappa_m = alpha_b * p_q.lam_minus + alpha_a * p_p.lam_minus
        delta_minus = (
            np.log(kappa_m)
            - alpha_a * np.log(p_p.lam_minus)
            - alpha_b * np.log(p_q.lam_minus)
        )

        return T * p_q.C * (delta_plus + delta_minus)

    def closed_form_divergence(self, params_p, params_q, alpha: float, T: float) -> float:
        """VG α-divergence using Frullani-based closed form."""
        _AEP = 1e-9
        if abs(alpha - 1.0) < _AEP:
            return self._kl_endpoint(params_p, params_q, T, direction=1)
        if abs(alpha + 1.0) < _AEP:
            return self._kl_endpoint(params_p, params_q, T, direction=-1)

        delta = self.closed_form_delta(params_p, params_q, alpha, T)
        return (4.0 / (1.0 - alpha ** 2)) * (1.0 - np.exp(-delta))

    def _kl_endpoint(self, params_p, params_q, T: float, direction: int) -> float:
        """α = ±1 endpoint using KL / reverse-KL formulas."""
        if direction == -1:
            # D^{-1}(P||Q) = T C [ψ_Q exp(ψ_Q) - exp(ψ_Q) + 1] integrated against ν_Q.
            # ψ_Q(x) = (λ_+^Q - λ_+^P)x for x>0.
            # ∫_0^∞ [d x e^{dx} - e^{dx} + 1] C e^{-λ^Q x}/x dx  where d=λ^Q-λ^P.
            # Let ρ = λ^P/λ^Q (ratio), then ψ = (λ^Q-λ^P)x = λ^Q(1 - ρ^{-1})... this gets complex.
            # Use integration-by-parts / closed form:
            # ∫_0^∞ [ψ e^ψ - e^ψ + 1] e^{-λ^Q x}/x dx
            # with ψ = (λ^Q - λ^P)x.
            # Let Δ = λ^P - λ^Q. Then ψ = -Δ x.
            # Integrand: [(-Δx)e^{-Δx} - e^{-Δx} + 1] e^{-λ^Q x}/x
            # = [(-Δx - 1)e^{-Δx} + 1] e^{-λ^Q x}/x
            # = (-Δx - 1)e^{-λ^P x}/x + e^{-λ^Q x}/x
            # = -Δ e^{-λ^P x} - e^{-λ^P x}/x + e^{-λ^Q x}/x
            # ∫_0^∞ -Δ e^{-λ^P x} dx = -Δ/λ^P
            # ∫_0^∞ [e^{-λ^Q x} - e^{-λ^P x}]/x dx = log(λ^P/λ^Q)   [Frullani]
            # Total for + side: -Δ/λ^P + log(λ^P/λ^Q)
            #                 = -(λ^P-λ^Q)/λ^P + log(λ^P/λ^Q)
            #                 = 1 - λ^Q/λ^P + log(λ^Q/λ^P) * (-1) ... sign check.
            # Let r = λ^Q/λ^P:  -Δ/λ^P = -(1 - r) = r - 1; log(λ^P/λ^Q) = -log r.
            # So: (r - 1) - log r.   [This is always ≥ 0 by convexity: f(r)=r-1-log r ≥ 0]
            p_p = VGParams(*params_p)
            p_q = VGParams(*params_q)
            Dp = p_p.lam_plus - p_q.lam_plus
            Dm = p_p.lam_minus - p_q.lam_minus
            r_plus = p_q.lam_plus / p_p.lam_plus
            r_minus = p_q.lam_minus / p_p.lam_minus
            I_plus = p_q.C * (r_plus - 1.0 - np.log(r_plus))
            I_minus = p_q.C * (r_minus - 1.0 - np.log(r_minus))
            return T * (I_plus + I_minus)
        else:
            return self._kl_endpoint(params_q, params_p, T, direction=-1)

    # ------------------------------------------------------------------
    # Fisher information matrix (diagonal)
    # ------------------------------------------------------------------

    def fisher_matrix(self, params, T: float) -> np.ndarray:
        """g = diag(TC/λ_+², TC/λ_-²)."""
        p = VGParams(*params)
        _validate_vg(p)
        g_plus = T * p.C / (p.lam_plus ** 2)
        g_minus = T * p.C / (p.lam_minus ** 2)
        return np.diag([g_plus, g_minus])

    # ------------------------------------------------------------------
    # α-connection
    # ------------------------------------------------------------------

    def alpha_connection(self, params, alpha: float, T: float) -> dict:
        """Γ^α_{λ+λ+,λ+} = -(1-α) TC/λ_+³, Γ^α_{λ-λ-,λ-} = -(1-α) TC/λ_-³."""
        p = VGParams(*params)
        _validate_vg(p)
        coeff = -(1.0 - alpha)
        return {
            "Gamma_lp_lp_lp": coeff * T * p.C / (p.lam_plus ** 3),
            "Gamma_lm_lm_lm": coeff * T * p.C / (p.lam_minus ** 3),
        }

    # ------------------------------------------------------------------
    # Potential, dual coordinates
    # ------------------------------------------------------------------

    def potential(self, params, T: float) -> float:
        """K_VG = -TC(log λ_+ + log λ_-)."""
        p = VGParams(*params)
        _validate_vg(p)
        return -T * p.C * (np.log(p.lam_plus) + np.log(p.lam_minus))

    def dual_coordinates(self, params, T: float) -> np.ndarray:
        """η_± = -TC/λ_±."""
        p = VGParams(*params)
        _validate_vg(p)
        return np.array([-T * p.C / p.lam_plus, -T * p.C / p.lam_minus])

    # ------------------------------------------------------------------
    # Jeffreys prior
    # ------------------------------------------------------------------

    def jeffreys_prior(self, params, T: float) -> float:
        """J_VG(ξ) ∝ TC/(λ_+ λ_-)."""
        p = VGParams(*params)
        _validate_vg(p)
        return T * p.C / (p.lam_plus * p.lam_minus)

    # ------------------------------------------------------------------
    # Log-coordinate Euclidean metric verification
    # ------------------------------------------------------------------

    def metric_log_coords(self, params, T: float) -> np.ndarray:
        """Return Fisher matrix in log-coordinates (x = log λ_+, y = log λ_-).

        Under the coordinate change x = log λ_+, y = log λ_-:
            g_{xx} = g_{λ+λ+} (∂λ+/∂x)² = (TC/λ_+²) λ_+² = TC
            g_{yy} = TC, g_{xy} = 0.
        So ds² = TC(dx² + dy²): Euclidean with scale TC.
        """
        p = VGParams(*params)
        _validate_vg(p)
        scale = T * p.C
        return np.diag([scale, scale])
