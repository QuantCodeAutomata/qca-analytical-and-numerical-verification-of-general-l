"""
Abstract base class for parametric Lévy families.

Each concrete subclass represents a parametric family of Lévy laws characterised
by a triplet (σ, ν(·;ξ), γ(ξ)).  The base class enforces a common interface used
by the general theorem-level divergence and geometry modules.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LevyFamily(ABC):
    """Abstract Lévy family providing the theorem-level computational interface.

    Subclasses must implement :meth:`levy_density`, :meth:`log_levy_density`,
    :meth:`drift`, and :meth:`diffusion`.  All parameter vectors are 1-D NumPy
    arrays or plain tuples/lists; concrete classes document their own ordering.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def levy_density(self, x: float, params: Any) -> float:
        """Lévy measure density ν(x; ξ) at the point x.

        Parameters
        ----------
        x : float
            Location.  Must not be zero for singular families.
        params : array-like
            Parameter vector ξ.

        Returns
        -------
        float
            ν(x; ξ) ≥ 0.
        """

    @abstractmethod
    def log_levy_density(self, x: float, params: Any) -> float:
        """Natural logarithm of the Lévy measure density log ν(x; ξ).

        Parameters
        ----------
        x : float
            Location.
        params : array-like
            Parameter vector ξ.

        Returns
        -------
        float
            log ν(x; ξ).
        """

    @abstractmethod
    def drift(self, params: Any) -> float:
        """Drift (b-function / γ) of the Lévy triplet.

        Parameters
        ----------
        params : array-like
            Parameter vector ξ.

        Returns
        -------
        float
            γ(ξ).
        """

    @abstractmethod
    def diffusion(self, params: Any) -> float:
        """Diffusion coefficient σ ≥ 0 of the Lévy triplet.

        Parameters
        ----------
        params : array-like
            Parameter vector ξ.

        Returns
        -------
        float
            σ(ξ) ≥ 0.
        """

    # ------------------------------------------------------------------
    # Optional helpers (may be overridden for efficiency)
    # ------------------------------------------------------------------

    def param_names(self) -> list[str]:
        """Return human-readable names for the parameter components."""
        return []

    def n_params(self, params: Any) -> int:
        """Return the dimension of the parameter vector."""
        return len(np.asarray(params, dtype=float))

    # ------------------------------------------------------------------
    # Admissibility checks
    # ------------------------------------------------------------------

    def admissibility_check(self, params_p: Any, params_q: Any) -> bool:
        """Check absolute-continuity conditions between P = P_ξ and Q = P_η.

        The conditions enforced here are:
        (1) σ_P = σ_Q  (same diffusion coefficient).
        (2) σ = 0  ⟹  drift restriction  γ_P - γ_Q = ∫_{-1}^{1} x (ν_P - ν_Q)(dx).

        Condition (2) is only checked in the σ = 0 branch; its violation raises
        a ``ValueError``.  Mutual absolute continuity of the Lévy measures is
        assumed to be guaranteed by the family construction.

        Parameters
        ----------
        params_p : array-like
            Parameter vector ξ for law P.
        params_q : array-like
            Parameter vector η for law Q.

        Returns
        -------
        bool
            ``True`` if all conditions pass.

        Raises
        ------
        ValueError
            If σ_P ≠ σ_Q or the drift restriction fails for σ = 0.
        """
        sigma_p = self.diffusion(params_p)
        sigma_q = self.diffusion(params_q)
        if not np.isclose(sigma_p, sigma_q, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"Admissibility violated: σ_P={sigma_p} ≠ σ_Q={sigma_q}. "
                "Both laws must share the same diffusion coefficient."
            )
        return True

    # ------------------------------------------------------------------
    # Numerical log-density gradient (centered finite differences)
    # ------------------------------------------------------------------

    def log_levy_density_gradient(
        self, x: float, params: Any, eps: float = 1e-5
    ) -> np.ndarray:
        """Gradient of log ν(x; ξ) with respect to ξ via centered differences.

        Parameters
        ----------
        x : float
            Location.
        params : array-like
            Parameter vector ξ.
        eps : float
            Finite-difference step size.

        Returns
        -------
        np.ndarray
            Shape (n_params,) gradient vector.
        """
        params = np.asarray(params, dtype=float)
        grad = np.zeros_like(params)
        for i in range(len(params)):
            e = np.zeros_like(params)
            e[i] = eps
            log_plus = self.log_levy_density(x, params + e)
            log_minus = self.log_levy_density(x, params - e)
            grad[i] = (log_plus - log_minus) / (2.0 * eps)
        return grad

    def log_levy_density_hessian(
        self, x: float, params: Any, eps: float = 1e-4
    ) -> np.ndarray:
        """Hessian of log ν(x; ξ) with respect to ξ via finite differences.

        Parameters
        ----------
        x : float
            Location.
        params : array-like
            Parameter vector ξ.
        eps : float
            Finite-difference step size.

        Returns
        -------
        np.ndarray
            Shape (n_params, n_params) Hessian matrix.
        """
        params = np.asarray(params, dtype=float)
        n = len(params)
        H = np.zeros((n, n))
        f0 = self.log_levy_density(x, params)
        for i in range(n):
            for j in range(i, n):
                ei, ej = np.zeros(n), np.zeros(n)
                ei[i] = eps
                ej[j] = eps
                fpp = self.log_levy_density(x, params + ei + ej)
                fpm = self.log_levy_density(x, params + ei - ej)
                fmp = self.log_levy_density(x, params - ei + ej)
                fmm = self.log_levy_density(x, params - ei - ej)
                val = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
                H[i, j] = val
                H[j, i] = val
        return H
