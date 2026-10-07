"""
Generalized Tempered Stable (GTS) Lévy family.

Parameters ξ = (C_+, C_-, a_+, a_-, λ_+, λ_-)
Constraints: C_± > 0, a_± ∈ (0,2)\\{1}, λ_± > 0, σ = 0.

Closed-form formulas derived from the paper's Eq. (93) and surrounding results.
Custom implementation — Context7 confirmed no library provides GTS
information-geometry formulas.
"""

from __future__ import annotations

from typing import NamedTuple

import mpmath
import numpy as np
from scipy.special import gamma as sp_gamma, gammaln

from src.levy_base import LevyFamily


class GTSParams(NamedTuple):
    """Parameter container for the GTS family."""
    C_plus: float    # intensity coefficient positive jumps
    C_minus: float   # intensity coefficient negative jumps
    a_plus: float    # stability index positive side  ∈ (0,2)\{1}
    a_minus: float   # stability index negative side  ∈ (0,2)\{1}
    lam_plus: float  # tempering rate positive jumps
    lam_minus: float # tempering rate negative jumps


def _validate_gts(p: GTSParams) -> None:
    assert p.C_plus > 0 and p.C_minus > 0, "C_± must be positive"
    assert 0 < p.a_plus < 2 and p.a_plus != 1, "a_+ must be in (0,2)\\{1}"
    assert 0 < p.a_minus < 2 and p.a_minus != 1, "a_- must be in (0,2)\\{1}"
    assert p.lam_plus > 0 and p.lam_minus > 0, "λ_± must be positive"


def _gamma_neg(a: float) -> float:
    """Γ(-a) using mpmath for reliability near non-integer values."""
    return float(mpmath.gamma(-a))


class GTS(LevyFamily):
    """Generalized Tempered Stable Lévy family (σ = 0).

    Lévy density:
        ν(x; ξ) = C_+ exp(-λ_+ x) / x^(a_++1)  for x > 0
                  C_- exp(-λ_- |x|) / |x|^(a_-+1) for x < 0
    """

    def levy_density(self, x: float, params) -> float:
        """GTS Lévy measure density ν(x; ξ)."""
        p = GTSParams(*params)
        if x > 0:
            return p.C_plus * np.exp(-p.lam_plus * x) / (x ** (p.a_plus + 1))
        elif x < 0:
            ax = abs(x)
            return p.C_minus * np.exp(-p.lam_minus * ax) / (ax ** (p.a_minus + 1))
        return 0.0

    def log_levy_density(self, x: float, params) -> float:
        """log ν(x; ξ) for the GTS family."""
        p = GTSParams(*params)
        if x > 0:
            return np.log(p.C_plus) - p.lam_plus * x - (p.a_plus + 1) * np.log(x)
        elif x < 0:
            ax = abs(x)
            return np.log(p.C_minus) - p.lam_minus * ax - (p.a_minus + 1) * np.log(ax)
        return -np.inf

    def drift(self, params) -> float:
        """σ = 0 family; drift set to 0."""
        return 0.0

    def diffusion(self, params) -> float:
        """Pure-jump process: σ = 0."""
        return 0.0

    def param_names(self) -> list[str]:
        return ["C_plus", "C_minus", "a_plus", "a_minus", "lam_plus", "lam_minus"]

    # ------------------------------------------------------------------
    # Closed-form divergence (paper Eq. 93 structure)
    # ------------------------------------------------------------------

    def closed_form_delta(self, params_p, params_q, alpha: float, T: float) -> float:
        """Compute Δ_T^α using the GTS closed-form gamma-function integrals.

        Only λ_± may differ between P and Q; C_± and a_± are held fixed.

        ψ_Q(x; P) = -(λ_+^P - λ_+^Q) x  for x > 0
                   -(λ_-^P - λ_-^Q)|x|  for x < 0

        Each integral over the positive axis is:
          ∫_0^∞ [b + a exp(ψ) - exp(a ψ)] C_+ exp(-λ_+^Q x)/x^(a_++1) dx
        where a=(1-α)/2, b=(1+α)/2, ψ = -Δλ x.

        Using ∫_0^∞ x^{β-1} exp(-κ x) dx = Γ(β)/κ^β:
          = C_+ { b Γ(-a_+)/(λ_+^Q)^{-a_+} → wait this needs β = -a_+ which fails.

        Rewrite carefully: ∫_0^∞ exp(-κx)/x^{a+1} dx = κ^a Γ(-a) (for 0<a<1)
        or use the identity ∫_0^∞ x^{β-1} e^{-κx} dx = Γ(β)/κ^β for β>0.

        Let β = -a_+ so β = -a < 0. Use standard extension via Euler's formula.
        Better: write directly as differences of Laplace transforms.

        For a_+ ∈ (0,1):  ∫_0^∞ e^{-κx}/x^{a_++1} dx = κ^{a_+} Γ(-a_+) with Γ(-a_+) < 0.
        So C_+ ∫_0^∞ e^{-κx}/x^{a_++1} dx = C_+ Γ(-a_+) κ^{a_+}.

        The three terms in the integrand with κ values:
          - b * C_+ Γ(-a_+) (λ_+^Q)^{a_+}
          - a * C_+ Γ(-a_+) (λ_+^P)^{a_+}    [since exp(ψ)·e^{-λ^Q x} = e^{-(λ^P)x}]
          + C_+ Γ(-a_+) ((1-a)/2 · λ_+^P + (1+a)/2 · λ_+^Q … wait this is for the exp(aψ) term.

        Wait: exp(ψ) = exp(-(λ_+^P - λ_+^Q)x) so ψ_Q = (λ_+^Q - λ_+^P)x.
        Actually ψ_Q = log(dν_P/dν_Q) = log(ν_P/ν_Q).

        For positive x:
          ν_P(x) = C_+ exp(-λ_+^P x)/x^{a_++1}
          ν_Q(x) = C_+ exp(-λ_+^Q x)/x^{a_++1}
          ψ_Q(x) = log(ν_P/ν_Q) = -(λ_+^P - λ_+^Q)x = (λ_+^Q - λ_+^P)x

        Let Δ_+ = λ_+^P - λ_+^Q, so ψ_Q(x) = -Δ_+ x for x > 0.

        Integrand on positive side (integrate against ν_Q):
          [a exp(ψ_Q) + b - exp(a ψ_Q)] ν_Q
          = [a exp(-Δ_+ x) + b - exp(-a Δ_+ x)] C_+ exp(-λ_+^Q x)/x^{a_++1}
          = C_+ { a exp(-(λ_+^Q + Δ_+)x)/x^{a_++1} + b exp(-λ_+^Q x)/x^{a_++1} - exp(-(λ_+^Q + aΔ_+)x)/x^{a_++1} }
          = C_+ { a exp(-λ_+^P x)/x^{a_++1} + b exp(-λ_+^Q x)/x^{a_++1} - exp(-(λ_+^Q + aΔ_+)x)/x^{a_++1} }

        Integrating each term using ∫_0^∞ exp(-κx)/x^{a+1} dx = Γ(-a) κ^a (valid for a ∈ (0,1)).
        For a ∈ (1,2) we need a_+ ∈ (1,2): Γ(-a) is defined and negative.

        So:
          I_+ = C_+ Γ(-a_+) { a (λ_+^P)^{a_+} + b (λ_+^Q)^{a_+} - (λ_+^Q + a Δ_+)^{a_+} }

        Note: λ_+^Q + a Δ_+ = λ_+^Q + ((1-α)/2)(λ_+^P - λ_+^Q) = (1+α)/2 λ_+^Q + (1-α)/2 λ_+^P.

        Similarly for negative side:
          I_- = C_- Γ(-a_-) { a (λ_-^P)^{a_-} + b (λ_-^Q)^{a_-} - (λ_-^Q + a Δ_-)^{a_-} }
        """
        p_p = GTSParams(*params_p)
        p_q = GTSParams(*params_q)
        _validate_gts(p_q)

        alpha_a = (1.0 - alpha) / 2.0  # a coefficient
        alpha_b = (1.0 + alpha) / 2.0  # b coefficient

        # -- Positive side --
        G_neg_ap = _gamma_neg(p_q.a_plus)  # Γ(-a_+), negative for a_+ ∈ (0,1) or (1,2)
        kappa_pos = alpha_b * p_q.lam_plus + alpha_a * p_p.lam_plus
        I_plus = p_q.C_plus * G_neg_ap * (
            alpha_a * (p_p.lam_plus ** p_q.a_plus)
            + alpha_b * (p_q.lam_plus ** p_q.a_plus)
            - kappa_pos ** p_q.a_plus
        )

        # -- Negative side --
        G_neg_am = _gamma_neg(p_q.a_minus)
        kappa_neg = alpha_b * p_q.lam_minus + alpha_a * p_p.lam_minus
        I_minus = p_q.C_minus * G_neg_am * (
            alpha_a * (p_p.lam_minus ** p_q.a_minus)
            + alpha_b * (p_q.lam_minus ** p_q.a_minus)
            - kappa_neg ** p_q.a_minus
        )

        return T * (I_plus + I_minus)

    def closed_form_divergence(
        self, params_p, params_q, alpha: float, T: float
    ) -> float:
        """GTS α-divergence using closed-form gamma integrals."""
        _AEP = 1e-9
        if abs(alpha - 1.0) < _AEP:
            return self._kl_endpoint(params_p, params_q, T, direction=1)
        if abs(alpha + 1.0) < _AEP:
            return self._kl_endpoint(params_p, params_q, T, direction=-1)

        delta = self.closed_form_delta(params_p, params_q, alpha, T)
        return (4.0 / (1.0 - alpha ** 2)) * (1.0 - np.exp(-delta))

    def _kl_endpoint(self, params_p, params_q, T: float, direction: int) -> float:
        """KL-type endpoint formula: direction=1 gives D^1, direction=-1 gives D^{-1}."""
        if direction == -1:
            # D^{-1}(P||Q) = T ∫ [ψ_Q exp(ψ_Q) - exp(ψ_Q) + 1] ν_Q dx
            p_p = GTSParams(*params_p)
            p_q = GTSParams(*params_q)
            G_neg_ap = _gamma_neg(p_q.a_plus)
            G_neg_am = _gamma_neg(p_q.a_minus)
            Dp = p_p.lam_plus - p_q.lam_plus
            Dm = p_p.lam_minus - p_q.lam_minus
            # ∫ [ψ exp(ψ) - exp(ψ) + 1] ν_Q dx using series or symbolic form.
            # ψ = -Δ x for x>0; ∫_0^∞ [(-Δx)e^{-Δx} - e^{-Δx} + 1] C exp(-λ^Q x)/x^{a+1} dx
            # = C ∫_0^∞ [-Δx e^{-(λ^P)x} - e^{-(λ^P)x} + e^{-(λ^Q)x}] / x^{a+1} dx
            # First two terms: -(Δ/(a-1)) + ... need careful treatment.
            # Use: ∫_0^∞ x^k exp(-κx)/x^{a+1} dx = Γ(k-a)/κ^{k-a}
            # Term1: -Dp * ∫ x e^{-λ^P x}/x^{a+1} dx = -Dp * Γ(1-a_+)/(λ^P)^{1-a_+}... wait

            # More carefully:
            # ψ exp(ψ) - exp(ψ) + 1 = ψ_Q exp(ψ_Q) - (exp(ψ_Q) - 1)
            # For positive side with ψ = -Δ_+ x:
            # = (-Δ_+ x)e^{-Δ_+ x} - e^{-Δ_+ x} + 1
            # Integrate against C_+ e^{-λ_+^Q x}/x^{a_++1}:
            # = C_+ [(-Δ_+) ∫ x e^{-(λ_+^P)x}/x^{a+1} dx - ∫ e^{-(λ_+^P)x}/x^{a+1} dx + ∫ e^{-(λ_+^Q)x}/x^{a+1} dx]
            # = C_+ [(-Δ_+) Γ(-a_+)/(λ_+^P)^{-a_+} · (−a_+)... wait
            # ∫_0^∞ x^{1-a_+-1} e^{-λ^P x} dx = ∫_0^∞ x^{-a_+} e^{-λ^P x} dx = Γ(1-a_+)/(λ^P)^{1-a_+}

            Γ_neg_ap = G_neg_ap
            Γ_1mAp = float(mpmath.gamma(1 - p_q.a_plus))
            Γ_neg_am = G_neg_am
            Γ_1mAm = float(mpmath.gamma(1 - p_q.a_minus))

            I_plus_kl = p_q.C_plus * (
                (-Dp) * Γ_1mAp / (p_p.lam_plus ** (1 - p_q.a_plus))
                - Γ_neg_ap * ((p_p.lam_plus ** p_q.a_plus) - (p_q.lam_plus ** p_q.a_plus))
            )
            I_minus_kl = p_q.C_minus * (
                (-Dm) * Γ_1mAm / (p_p.lam_minus ** (1 - p_q.a_minus))
                - Γ_neg_am * ((p_p.lam_minus ** p_q.a_minus) - (p_q.lam_minus ** p_q.a_minus))
            )
            return T * (I_plus_kl + I_minus_kl)
        else:
            # D^1(P||Q) = D^{-1}(Q||P)
            return self._kl_endpoint(params_q, params_p, T, direction=-1)

    # ------------------------------------------------------------------
    # Fisher information matrix (closed form, diagonal)
    # ------------------------------------------------------------------

    def fisher_matrix(self, params, T: float) -> np.ndarray:
        """Diagonal Fisher matrix g = diag(g_{λ+}, g_{λ-}).

        g_{λ+} = T C_+ Γ(2-a_+) / λ_+^{2-a_+}
        g_{λ-} = T C_- Γ(2-a_-) / λ_-^{2-a_-}

        Returns 2×2 matrix indexed by (λ_+, λ_-).
        """
        p = GTSParams(*params)
        _validate_gts(p)
        Γ_2ma_plus = float(mpmath.gamma(2 - p.a_plus))
        Γ_2ma_minus = float(mpmath.gamma(2 - p.a_minus))
        g_plus = T * p.C_plus * Γ_2ma_plus / (p.lam_plus ** (2 - p.a_plus))
        g_minus = T * p.C_minus * Γ_2ma_minus / (p.lam_minus ** (2 - p.a_minus))
        return np.diag([g_plus, g_minus])

    # ------------------------------------------------------------------
    # α-connection (closed form, only diagonal cubic components nonzero)
    # ------------------------------------------------------------------

    def alpha_connection(self, params, alpha: float, T: float) -> dict:
        """Nonzero α-connection components.

        Γ^α_{λ+λ+,λ+} = -(1-α)/2 · T C_+ Γ(3-a_+)/λ_+^{3-a_+}
        Γ^α_{λ-λ-,λ-} = -(1-α)/2 · T C_- Γ(3-a_-)/λ_-^{3-a_-}
        """
        p = GTSParams(*params)
        _validate_gts(p)
        coeff = -(1.0 - alpha) / 2.0
        Γ_3ma_plus = float(mpmath.gamma(3 - p.a_plus))
        Γ_3ma_minus = float(mpmath.gamma(3 - p.a_minus))
        return {
            "Gamma_lp_lp_lp": coeff * T * p.C_plus * Γ_3ma_plus / (p.lam_plus ** (3 - p.a_plus)),
            "Gamma_lm_lm_lm": coeff * T * p.C_minus * Γ_3ma_minus / (p.lam_minus ** (3 - p.a_minus)),
        }

    # ------------------------------------------------------------------
    # Potential K and dual coordinates η
    # ------------------------------------------------------------------

    def potential(self, params, T: float) -> float:
        """GTS potential K = T C_+ Γ(-a_+) λ_+^{a_+} + T C_- Γ(-a_-) λ_-^{a_-}."""
        p = GTSParams(*params)
        _validate_gts(p)
        K = (
            T * p.C_plus * _gamma_neg(p.a_plus) * (p.lam_plus ** p.a_plus)
            + T * p.C_minus * _gamma_neg(p.a_minus) * (p.lam_minus ** p.a_minus)
        )
        return float(K)

    def dual_coordinates(self, params, T: float) -> np.ndarray:
        """Dual (η) coordinates: η_± = ∂K/∂λ_± = -T C_± Γ(1-a_±) λ_±^{a_±-1}."""
        p = GTSParams(*params)
        _validate_gts(p)
        η_plus = -T * p.C_plus * float(mpmath.gamma(1 - p.a_plus)) * (p.lam_plus ** (p.a_plus - 1))
        η_minus = -T * p.C_minus * float(mpmath.gamma(1 - p.a_minus)) * (p.lam_minus ** (p.a_minus - 1))
        return np.array([η_plus, η_minus])
