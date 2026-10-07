"""
Variance Gamma Monte Carlo bias-reduction study (exp_3).

Reproduces the paper's simulation comparing MLE and Firth estimators for
the VG parameters λ_+ and λ_- using the sufficient-statistic formulation.

The sufficient statistics S_± ~ Gamma(shape = nTC, rate = λ_±^true) are
simulated directly (no path simulation needed).

Custom implementation — Context7 found no library providing this exact
Firth-bias-corrected VG estimator; scipy.stats.gamma used for simulation.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate_vg_bias_study(
    n: int,
    R: int,
    C: float,
    T: float,
    lambda_plus: float,
    lambda_minus: float,
    rng: np.random.Generator,
) -> dict:
    """Run one Monte Carlo replication batch for a given sample size n.

    The VG log-likelihood (Eq. 114 region) factorises as
        l(λ_+) = nTC log λ_+ - λ_+ S_+  +  const.
        l(λ_-) = nTC log λ_- - λ_- S_-  +  const.
    with sufficient statistics S_± ~ Gamma(shape=nTC, rate=λ_±^true).

    MLE:   λ̂_±^MLE   = nTC / S_±
    Firth: λ̂_±^Firth = (nTC - 1) / S_±

    Parameters
    ----------
    n : int
        Sample size (number of VG observations).
    R : int
        Number of Monte Carlo replications.
    C : float
        VG intensity parameter (same for + and -).
    T : float
        Time horizon.
    lambda_plus : float
        True λ_+ (positive-jump tempering rate).
    lambda_minus : float
        True λ_- (negative-jump tempering rate).
    rng : np.random.Generator
        Reproducible random number generator.

    Returns
    -------
    dict
        Keys: 'n', 'shape', 'estimates_plus_mle', 'estimates_plus_firth',
        'estimates_minus_mle', 'estimates_minus_firth', plus summary metrics.
    """
    shape = n * T * C
    if shape <= 1.0:
        warnings.warn(
            f"shape = nTC = {shape} ≤ 1; Firth estimator E[1/S] may be infinite.",
            stacklevel=2,
        )

    # S_± ~ Gamma(shape=shape, scale=1/rate) — NumPy parameterises by (shape, scale)
    # scale = 1 / λ_true (shape-rate ↔ shape-scale conversion)
    S_plus = rng.gamma(shape=shape, scale=1.0 / lambda_plus, size=R)
    S_minus = rng.gamma(shape=shape, scale=1.0 / lambda_minus, size=R)

    # MLE
    lam_plus_mle = shape / S_plus
    lam_minus_mle = shape / S_minus

    # Firth bias-corrected estimator
    lam_plus_firth = (shape - 1.0) / S_plus
    lam_minus_firth = (shape - 1.0) / S_minus

    return {
        "n": n,
        "shape": shape,
        "lam_plus_mle": lam_plus_mle,
        "lam_plus_firth": lam_plus_firth,
        "lam_minus_mle": lam_minus_mle,
        "lam_minus_firth": lam_minus_firth,
    }


def summarise_results(
    results: dict,
    lambda_plus: float,
    lambda_minus: float,
) -> pd.DataFrame:
    """Compute relative bias and RMSE from simulation results.

    Parameters
    ----------
    results : dict
        Output from :func:`simulate_vg_bias_study`.
    lambda_plus : float
        True λ_+.
    lambda_minus : float
        True λ_-.

    Returns
    -------
    pd.DataFrame
        Row index: estimator × parameter, columns: mean, rel_bias, rmse, std.
    """
    rows = []
    for (estimates, true_val, label) in [
        (results["lam_plus_mle"], lambda_plus, "MLE λ+"),
        (results["lam_plus_firth"], lambda_plus, "Firth λ+"),
        (results["lam_minus_mle"], lambda_minus, "MLE λ-"),
        (results["lam_minus_firth"], lambda_minus, "Firth λ-"),
    ]:
        mean_est = float(np.mean(estimates))
        rel_bias = (mean_est - true_val) / true_val
        rmse = float(np.sqrt(np.mean((estimates - true_val) ** 2)))
        std = float(np.std(estimates, ddof=1))
        rows.append(
            {
                "n": results["n"],
                "estimator": label,
                "mean": mean_est,
                "rel_bias": rel_bias,
                "rmse": rmse,
                "std": std,
            }
        )
    return pd.DataFrame(rows)


def run_full_study(
    C: float = 0.1,
    T: float = 1.0,
    lambda_plus: float = 1.0,
    lambda_minus: float = 1.5,
    sample_sizes: Optional[list[int]] = None,
    R: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full VG bias-reduction Monte Carlo study.

    Parameters match the paper's settings exactly:
        C = 0.1, T = 1, λ_+ = 1, λ_- = 1.5, n ∈ {50,100,200,500}, R = 5000.
    Random seed is set to 42 (implementation choice; paper does not specify one).

    Parameters
    ----------
    C, T, lambda_plus, lambda_minus : float
        VG model parameters.
    sample_sizes : list of int, optional
        Sample sizes to evaluate.  Defaults to [50, 100, 200, 500].
    R : int
        Number of Monte Carlo replications.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        n, estimator, mean, rel_bias, rmse, std.
    """
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 500]

    rng = np.random.default_rng(seed)
    all_rows = []
    for n in sample_sizes:
        res = simulate_vg_bias_study(n, R, C, T, lambda_plus, lambda_minus, rng)
        df_n = summarise_results(res, lambda_plus, lambda_minus)
        all_rows.append(df_n)
    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Analytical cross-check
# ---------------------------------------------------------------------------

def analytical_expectations(
    C: float,
    T: float,
    lambda_plus: float,
    lambda_minus: float,
    sample_sizes: list[int],
) -> pd.DataFrame:
    """Compute exact expected values of MLE and Firth estimators.

    For S ~ Gamma(k, rate=λ), E[1/S] = λ/(k-1) for k > 1, so:
        E[λ̂_MLE]   = k/(k-1) · λ
        E[λ̂_Firth] = λ        (exactly unbiased)

    Parameters
    ----------
    C, T, lambda_plus, lambda_minus : float
        VG model parameters.
    sample_sizes : list of int

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for n in sample_sizes:
        k = n * T * C
        for (lam_true, label_suffix) in [
            (lambda_plus, "λ+"),
            (lambda_minus, "λ-"),
        ]:
            if k > 1:
                e_mle = k / (k - 1) * lam_true
                e_firth = lam_true  # exactly unbiased
            else:
                e_mle = np.inf
                e_firth = np.nan
            rows.append(
                {
                    "n": n,
                    "shape_k": k,
                    "param": label_suffix,
                    "E[MLE]": e_mle,
                    "rel_bias_MLE": (e_mle - lam_true) / lam_true if k > 1 else np.inf,
                    "E[Firth]": e_firth,
                    "rel_bias_Firth": 0.0 if k > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)
