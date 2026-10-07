"""
Merton jump-diffusion Lévy family.

Parameters ξ = (σ, λ_j, μ_j, δ_j) where:
    σ > 0   – common diffusion coefficient
    λ_j > 0 – jump intensity (Poisson rate)
    μ_j ∈ ℝ – mean jump size (log-normal)
    δ_j > 0 – jump size standard deviation

The Lévy measure is absolutely continuous w.r.t. Lebesgue measure with density:
    ν(x; ξ) = λ_j / (δ_j √(2π)) · exp(-(x - μ_j)² / (2 δ_j²))

Custom implementation — Context7 confirmed no library provides Merton
information-geometry formulas at this level of generality.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import integrate

from src.levy_base import LevyFamily

_QUAD_LIMIT = 400
_TRUNC = 30.0   # σ-multiples for Gaussian tail truncation


class MertonParams(NamedTuple):
    """Parameter container for the Merton family."""
    sigma: float    # diffusion coefficient
    lam_j: float    # jump intensity
    mu_j: float     # mean jump size
    delta_j: float  # jump size std deviation


def _validate_merton(p: MertonParams) -> None:
    assert p.sigma > 0, "σ must be positive"
    assert p.lam_j > 0, "λ_j must be positive"
    assert p.delta_j > 0, "δ_j must be positive"


def _safe_quad(func, a: float, b: float) -> float:
    val, _ = integrate.quad(func, a, b, limit=_QUAD_LIMIT, epsabs=1e-12, epsrel=1e-10)
    return val


class Merton(LevyFamily):
    """Merton jump-diffusion Lévy family.

    Lévy density (jump part):
        ν(x; ξ) = λ_j / (δ_j √(2π)) · exp(-(x-μ_j)² / (2δ_j²))

    Diffusion part: σ²/2 times the second derivative of the characteristic exponent.
    """

    def levy_density(self, x: float, params) -> float:
        """Merton Lévy measure density (jump component)."""
        p = MertonParams(*params)
        _validate_merton(p)
        return (p.lam_j / (p.delta_j * np.sqrt(2.0 * np.pi))) * np.exp(
            -((x - p.mu_j) ** 2) / (2.0 * p.delta_j ** 2)
        )

    def log_levy_density(self, x: float, params) -> float:
        """log ν(x; ξ)."""
        p = MertonParams(*params)
        _validate_merton(p)
        return (
            np.log(p.lam_j)
            - np.log(p.delta_j)
            - 0.5 * np.log(2.0 * np.pi)
            - ((x - p.mu_j) ** 2) / (2.0 * p.delta_j ** 2)
        )

    def drift(self, params) -> float:
        """Drift parameter (set to 0; absorbed into model convention)."""
        return 0.0

    def diffusion(self, params) -> float:
        """Diffusion coefficient σ."""
        return MertonParams(*params).sigma

    def param_names(self) -> list[str]:
        return ["sigma", "lam_j", "mu_j", "delta_j"]

    # ------------------------------------------------------------------
    # Risk-neutral moment check
    # ------------------------------------------------------------------

    def verify_rn_moment(self, params) -> dict:
        """Verify ∫(e^x - 1) ν(dx) = λ_j (exp(μ_j + δ_j²/2) - 1).

        This is the risk-neutral condition for Merton's model.
        """
        p = MertonParams(*params)
        _validate_merton(p)
        analytic = p.lam_j * (np.exp(p.mu_j + 0.5 * p.delta_j ** 2) - 1.0)

        x_lo = p.mu_j - _TRUNC * p.delta_j
        x_hi = p.mu_j + _TRUNC * p.delta_j
        numerical = _safe_quad(
            lambda x: (np.exp(x) - 1.0) * self.levy_density(x, params), x_lo, x_hi
        )
        return {
            "analytic": analytic,
            "numerical": numerical,
            "abs_error": abs(analytic - numerical),
        }

    # ------------------------------------------------------------------
    # Fisher information matrix (theorem-level integral formula)
    # ------------------------------------------------------------------

    def fisher_matrix_theorem(self, params, T: float) -> np.ndarray:
        """Fisher information matrix via direct theorem integral.

        Parameters (σ, λ_j, μ_j, δ_j) → 4×4 matrix.

        Jump component: g_ij^{jump} = T ∫ (∂_i log ν)(∂_j log ν) ν(dx)
        Diffusion component (Brownian part): contributes 2T/σ² to g_{σσ}.

        Score functions for the jump Lévy density:
            ∂_{σ}   log ν = 0  (ν does not depend on σ)
            ∂_{λ_j} log ν = 1/λ_j
            ∂_{μ_j} log ν = (x - μ_j)/δ_j²
            ∂_{δ_j} log ν = -1/δ_j + (x-μ_j)²/δ_j³

        Wiener contribution to g_{σσ}: for a Brownian motion with diffusion σ,
        the Fisher information for σ from the Brownian component over [0,T] is
        2T/σ². (Standard result: score = (σW_T² - T)/σ³, Fisher = 2T/σ².)
        """
        p = MertonParams(*params)
        _validate_merton(p)
        n = 4
        g = np.zeros((n, n))

        x_lo = p.mu_j - _TRUNC * p.delta_j
        x_hi = p.mu_j + _TRUNC * p.delta_j

        def score_jump(x: float) -> np.ndarray:
            """Score vector (∂_{λ_j}, ∂_{μ_j}, ∂_{δ_j}) log ν(x)."""
            return np.array([
                1.0 / p.lam_j,                                         # ∂_{λ_j}
                (x - p.mu_j) / (p.delta_j ** 2),                      # ∂_{μ_j}
                -1.0 / p.delta_j + (x - p.mu_j) ** 2 / p.delta_j ** 3,  # ∂_{δ_j}
            ])

        # Jump part: indices 1,2,3 in the 4-parameter vector
        for i in range(3):
            for j in range(i, 3):
                def integrand(x: float, ii=i, jj=j) -> float:
                    s = score_jump(x)
                    return s[ii] * s[jj] * self.levy_density(x, params)

                val = T * _safe_quad(integrand, x_lo, x_hi)
                g[i + 1, j + 1] = val
                g[j + 1, i + 1] = val

        # Wiener contribution: g_{σσ} = 2T/σ²
        g[0, 0] = 2.0 * T / (p.sigma ** 2)

        return g

    # ------------------------------------------------------------------
    # α-connection via finite differences on the theorem divergence
    # (see info_geometry.py for the generic implementation)
    # ------------------------------------------------------------------

    def admissibility_check(self, params_p, params_q) -> bool:
        """For Merton, require same σ."""
        p_p = MertonParams(*params_p)
        p_q = MertonParams(*params_q)
        if not np.isclose(p_p.sigma, p_q.sigma, rtol=1e-10):
            raise ValueError(
                f"Admissibility violated: σ_P={p_p.sigma} ≠ σ_Q={p_q.sigma}."
            )
        return True
