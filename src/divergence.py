"""
General Lévy-process α-divergence.

Implements the compact theorem-level formula for the α-divergence between two
Lévy laws P_ξ and P_η sharing the same diffusion coefficient σ, together with
verification utilities.

Custom implementation — Context7 confirmed no library provides this exact
Lévy information-geometry divergence formula.
"""

from typing import Any

import numpy as np
from scipy import integrate

from src.levy_base import LevyFamily


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ALPHA_EPS = 1e-9   # proximity threshold for α = ±1 endpoint detection
_QUAD_LIMIT = 200   # max subdivisions for scipy.integrate.quad
_TRUNC = 100.0      # tail truncation for semi-infinite integrals


def _safe_quad(func, a: float, b: float, limit: int = _QUAD_LIMIT) -> float:
    """Wrapper around scipy.integrate.quad returning the integral value only."""
    result, _ = integrate.quad(func, a, b, limit=limit, epsabs=1e-10, epsrel=1e-8)
    return result


# ---------------------------------------------------------------------------
# Log Radon–Nikodym derivative ψ_Q
# ---------------------------------------------------------------------------

def compute_psi_Q(x: float, family: LevyFamily, params_p: Any, params_q: Any) -> float:
    """Log Radon–Nikodym derivative ψ_Q(x; P) = log(dν_P/dν_Q)(x).

    Parameters
    ----------
    x : float
        Jump size.
    family : LevyFamily
        Lévy family instance.
    params_p : array-like
        Parameters of P.
    params_q : array-like
        Parameters of Q.

    Returns
    -------
    float
        log(dν_P/dν_Q)(x).
    """
    log_nu_p = family.log_levy_density(x, params_p)
    log_nu_q = family.log_levy_density(x, params_q)
    return log_nu_p - log_nu_q


# ---------------------------------------------------------------------------
# Drift correction c_Q(P)
# ---------------------------------------------------------------------------

def compute_drift_correction(
    family: LevyFamily,
    params_p: Any,
    params_q: Any,
    T: float,          # noqa: ARG001 – kept for interface consistency
) -> float:
    """Compute the continuous/drift correction term c_Q(P).

    If σ ≠ 0:
        c_Q(P) = (1/σ)(γ_P - γ_Q - ∫_{-1}^{1} x (ν_P - ν_Q)(dx))

    If σ = 0 the function returns 0 (the drift restriction is assumed to be
    enforced by the family construction and checked in
    :func:`verify_drift_restriction`).

    Parameters
    ----------
    family : LevyFamily
    params_p : array-like
    params_q : array-like
    T : float
        Time horizon (unused here; included for interface symmetry).

    Returns
    -------
    float
        c_Q(P).
    """
    sigma = family.diffusion(params_q)
    if abs(sigma) < 1e-14:
        return 0.0

    gamma_p = family.drift(params_p)
    gamma_q = family.drift(params_q)

    # ∫_{-1}^{1} x (ν_P - ν_Q)(dx)
    def integrand_correction(x: float) -> float:
        nu_p = family.levy_density(x, params_p)
        nu_q = family.levy_density(x, params_q)
        return x * (nu_p - nu_q)

    integral_neg = _safe_quad(integrand_correction, -1.0, -1e-8)
    integral_pos = _safe_quad(integrand_correction, 1e-8, 1.0)
    integral_correction = integral_neg + integral_pos

    return (gamma_p - gamma_q - integral_correction) / sigma


def verify_drift_restriction(
    family: LevyFamily,
    params_p: Any,
    params_q: Any,
    tol: float = 1e-6,
) -> float:
    """Verify the drift restriction when σ = 0.

    The condition is:  γ_P - γ_Q = ∫_{-1}^{1} x (ν_P - ν_Q)(dx).

    Parameters
    ----------
    family : LevyFamily
    params_p : array-like
    params_q : array-like
    tol : float
        Acceptable absolute deviation.

    Returns
    -------
    float
        Residual |LHS - RHS|.  Warns if residual exceeds tol.
    """
    gamma_p = family.drift(params_p)
    gamma_q = family.drift(params_q)
    lhs = gamma_p - gamma_q

    def integrand(x: float) -> float:
        return x * (family.levy_density(x, params_p) - family.levy_density(x, params_q))

    rhs = _safe_quad(integrand, -1.0, -1e-8) + _safe_quad(integrand, 1e-8, 1.0)
    residual = abs(lhs - rhs)
    if residual > tol:
        import warnings
        warnings.warn(
            f"Drift restriction residual {residual:.3e} exceeds tolerance {tol:.3e}",
            stacklevel=2,
        )
    return residual


# ---------------------------------------------------------------------------
# Compact theorem expression Δ_T^α
# ---------------------------------------------------------------------------

def delta_T_alpha(
    family: LevyFamily,
    params_p: Any,
    params_q: Any,
    alpha: float,
    T: float,
) -> float:
    """Compact theorem expression for Δ_T^(α)(P||Q).

    Formula:
        Δ_T^α(P||Q) = ((1-α²)/8) T c_Q(P)²
                      + T ∫ [(1-α)/2 exp(ψ_Q) + (1+α)/2 - exp(((1-α)/2)ψ_Q)] ν_Q(dx)

    where ψ_Q(x) = log(dν_P/dν_Q)(x) and c_Q(P) is the drift correction.

    Parameters
    ----------
    family : LevyFamily
    params_p : array-like
    params_q : array-like
    alpha : float
        Divergence index.
    T : float
        Time horizon.

    Returns
    -------
    float
        Δ_T^(α)(P||Q).
    """
    c = compute_drift_correction(family, params_p, params_q, T)
    drift_term = ((1.0 - alpha ** 2) / 8.0) * T * c ** 2

    a = (1.0 - alpha) / 2.0
    b = (1.0 + alpha) / 2.0

    def integrand(x: float) -> float:
        # Numerically stable formulation via log-space:
        #   (b + a·exp(ψ) - exp(a·ψ)) · ν_Q = a·ν_P + b·ν_Q - ν_P^a · ν_Q^b
        # This avoids overflow of exp(ψ) when ν_P >> ν_Q.
        log_nu_p = family.log_levy_density(x, params_p)
        log_nu_q = family.log_levy_density(x, params_q)
        if not (np.isfinite(log_nu_p) and np.isfinite(log_nu_q)):
            return 0.0
        nu_p = np.exp(log_nu_p)
        nu_q = np.exp(log_nu_q)
        term_mixed = np.exp(a * log_nu_p + b * log_nu_q)   # ν_P^a · ν_Q^b
        return a * nu_p + b * nu_q - term_mixed

    # Split integration over negative and positive semi-axes to handle singularities
    jump_neg = _safe_quad(integrand, -_TRUNC, -1e-8)
    jump_pos = _safe_quad(integrand, 1e-8, _TRUNC)
    jump_term = T * (jump_neg + jump_pos)

    return drift_term + jump_term


# ---------------------------------------------------------------------------
# α-divergence D^α(P||Q)
# ---------------------------------------------------------------------------

def alpha_divergence(
    family: LevyFamily,
    params_p: Any,
    params_q: Any,
    alpha: float,
    T: float,
) -> float:
    """Lévy-process α-divergence D^(α)(P||Q).

    Endpoint cases α = ±1 use the direct Kullback–Leibler-like formulas;
    interior cases use the compact Δ-based formula.

    Parameters
    ----------
    family : LevyFamily
    params_p : array-like
    params_q : array-like
    alpha : float
        Divergence index α ∈ ℝ.
    T : float
        Time horizon T > 0.

    Returns
    -------
    float
        D^(α)(P||Q) ≥ 0.
    """
    family.admissibility_check(params_p, params_q)

    # α = 1 endpoint: D^1(P||Q) = D^{-1}(Q||P)
    if abs(alpha - 1.0) < _ALPHA_EPS:
        return _kl_minus1(family, params_q, params_p, T)

    # α = -1 endpoint
    if abs(alpha + 1.0) < _ALPHA_EPS:
        return _kl_minus1(family, params_p, params_q, T)

    # Interior: D^α = 4/(1-α²) * (1 - exp(-Δ_T^α))
    delta = delta_T_alpha(family, params_p, params_q, alpha, T)
    return (4.0 / (1.0 - alpha ** 2)) * (1.0 - np.exp(-delta))


def _kl_minus1(
    family: LevyFamily,
    params_p: Any,
    params_q: Any,
    T: float,
) -> float:
    """D^{-1}(P||Q): KL-like divergence for α = -1.

    Formula:
        D^{-1}(P||Q) = (T/2) c_Q(P)² + T ∫ [ψ_Q exp(ψ_Q) - exp(ψ_Q) + 1] ν_Q(dx)

    where ψ_Q(x) = log(dν_P/dν_Q)(x).
    """
    c = compute_drift_correction(family, params_p, params_q, T)
    drift_term = (T / 2.0) * c ** 2

    def integrand(x: float) -> float:
        psi = compute_psi_Q(x, family, params_p, params_q)
        nu_q = family.levy_density(x, params_q)
        return (psi * np.exp(psi) - np.exp(psi) + 1.0) * nu_q

    jump_term = T * (
        _safe_quad(integrand, -_TRUNC, -1e-8)
        + _safe_quad(integrand, 1e-8, _TRUNC)
    )
    return drift_term + jump_term


# ---------------------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------------------

def verify_duality(
    family: LevyFamily,
    params_p: Any,
    params_q: Any,
    alpha: float,
    T: float,
) -> dict:
    """Verify α-duality: D^α(P||Q) = D^{-α}(Q||P).

    Parameters
    ----------
    family : LevyFamily
    params_p : array-like
    params_q : array-like
    alpha : float
    T : float

    Returns
    -------
    dict
        Keys: 'D_alpha_PQ', 'D_neg_alpha_QP', 'abs_error', 'rel_error'.
    """
    d_pq = alpha_divergence(family, params_p, params_q, alpha, T)
    d_qp = alpha_divergence(family, params_q, params_p, -alpha, T)
    abs_err = abs(d_pq - d_qp)
    rel_err = abs_err / (max(abs(d_pq), 1e-15))
    return {
        "D_alpha_PQ": d_pq,
        "D_neg_alpha_QP": d_qp,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }


def verify_zero_divergence(
    family: LevyFamily,
    params: Any,
    alpha: float,
    T: float,
) -> float:
    """Verify D^α(P||P) = 0 (zero divergence at equal parameters).

    Returns the absolute value of D^α(P||P).
    """
    return abs(alpha_divergence(family, params, params, alpha, T))
