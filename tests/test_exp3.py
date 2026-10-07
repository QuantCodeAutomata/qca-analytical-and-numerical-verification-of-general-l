"""
Tests for Experiment 3: VG Monte Carlo Bias-Reduction Study.

Verifies:
- MLE formula: λ̂_MLE = nTC/S
- Firth formula: λ̂_Firth = (nTC-1)/S
- Gamma parameterisation is correct (shape-rate, not shape-scale confusion)
- E[MLE] = k/(k-1)*λ for k > 1
- E[Firth] = λ (exactly unbiased)
- MLE positive bias, Firth smaller bias
- Firth RMSE ≤ MLE RMSE
- nTC > 1 condition holds for all paper parameters
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.vg_simulation import (
    simulate_vg_bias_study,
    summarise_results,
    run_full_study,
    analytical_expectations,
)

# Paper constants
C = 0.1
T = 1.0
LAMBDA_PLUS = 1.0
LAMBDA_MINUS = 1.5
SAMPLE_SIZES = [50, 100, 200, 500]
R_SMALL = 2000   # smaller R for tests to run fast; paper uses 5000
SEED = 42


# ---------------------------------------------------------------------------
# Tests: gamma parameterisation correctness
# ---------------------------------------------------------------------------

class TestGammaParameterisation:
    def test_mean_of_S_plus(self):
        """E[S+] = shape/rate = nTC/λ_+."""
        n = 100
        shape = n * T * C
        rng = np.random.default_rng(SEED)
        res = simulate_vg_bias_study(n, 10000, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)
        mean_S = np.mean(n * T * C / res["lam_plus_mle"])  # S+ = nTC / λ̂_MLE
        expected = shape / LAMBDA_PLUS
        assert abs(mean_S - expected) / expected < 0.05, \
            f"E[S+] = {mean_S:.4f}, expected {expected:.4f}"

    def test_mean_of_S_minus(self):
        """E[S-] = nTC/λ_-."""
        n = 100
        shape = n * T * C
        rng = np.random.default_rng(SEED)
        res = simulate_vg_bias_study(n, 10000, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)
        mean_S = np.mean(n * T * C / res["lam_minus_mle"])
        expected = shape / LAMBDA_MINUS
        assert abs(mean_S - expected) / expected < 0.05, \
            f"E[S-] = {mean_S:.4f}, expected {expected:.4f}"


# ---------------------------------------------------------------------------
# Tests: MLE and Firth estimator formulas
# ---------------------------------------------------------------------------

class TestEstimatorFormulas:
    def test_mle_formula(self):
        """λ̂_MLE = nTC / S."""
        n = 100
        shape = n * T * C
        rng = np.random.default_rng(SEED)
        res = simulate_vg_bias_study(n, 500, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)
        # MLE = shape / S, Firth = (shape-1) / S → ratio = shape/(shape-1)
        ratio = res["lam_plus_mle"] / res["lam_plus_firth"]
        expected_ratio = shape / (shape - 1)
        assert abs(np.mean(ratio) - expected_ratio) < 1e-10, \
            f"MLE/Firth ratio wrong: {np.mean(ratio):.6f} vs {expected_ratio:.6f}"

    def test_firth_formula(self):
        """λ̂_Firth = (nTC-1) / S."""
        n = 200
        shape = n * T * C
        rng = np.random.default_rng(SEED)
        res = simulate_vg_bias_study(n, 500, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)
        # Check Firth is exactly (shape-1)/shape * MLE
        ratio = res["lam_minus_firth"] / res["lam_minus_mle"]
        expected = (shape - 1) / shape
        assert np.allclose(ratio, expected, atol=1e-12), \
            f"Firth/MLE ratio mismatch: {ratio[:5]} vs {expected}"


# ---------------------------------------------------------------------------
# Tests: nTC > 1 for all paper sample sizes
# ---------------------------------------------------------------------------

class TestSampleSizeCondition:
    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_ntc_gt_1(self, n):
        shape = n * T * C
        assert shape > 1, f"nTC={shape} ≤ 1 for n={n}: Firth estimator may be invalid"


# ---------------------------------------------------------------------------
# Tests: analytical expectations
# ---------------------------------------------------------------------------

class TestAnalyticalExpectations:
    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_mle_expectation_formula(self, n):
        """E[λ̂_MLE] = k/(k-1) * λ_true for k = nTC > 1."""
        k = n * T * C
        expected_plus = (k / (k - 1)) * LAMBDA_PLUS
        expected_minus = (k / (k - 1)) * LAMBDA_MINUS
        rng = np.random.default_rng(SEED)
        res = simulate_vg_bias_study(n, 50000, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)
        mc_plus = np.mean(res["lam_plus_mle"])
        mc_minus = np.mean(res["lam_minus_mle"])
        assert abs(mc_plus - expected_plus) / expected_plus < 0.03, \
            f"n={n}: E[MLE_+] MC={mc_plus:.4f} vs analytic={expected_plus:.4f}"
        assert abs(mc_minus - expected_minus) / expected_minus < 0.03, \
            f"n={n}: E[MLE_-] MC={mc_minus:.4f} vs analytic={expected_minus:.4f}"

    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_firth_expectation_formula(self, n):
        """E[λ̂_Firth] = λ_true (exactly unbiased for k > 1)."""
        rng = np.random.default_rng(SEED)
        res = simulate_vg_bias_study(n, 50000, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)
        mc_plus = np.mean(res["lam_plus_firth"])
        mc_minus = np.mean(res["lam_minus_firth"])
        assert abs(mc_plus - LAMBDA_PLUS) / LAMBDA_PLUS < 0.03, \
            f"n={n}: E[Firth_+] MC={mc_plus:.4f} vs true={LAMBDA_PLUS}"
        assert abs(mc_minus - LAMBDA_MINUS) / LAMBDA_MINUS < 0.03, \
            f"n={n}: E[Firth_-] MC={mc_minus:.4f} vs true={LAMBDA_MINUS}"


# ---------------------------------------------------------------------------
# Tests: summary statistics structure
# ---------------------------------------------------------------------------

class TestSummaryStatistics:
    def setup_method(self):
        rng = np.random.default_rng(SEED)
        self.res_100 = simulate_vg_bias_study(100, R_SMALL, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng)

    def test_summary_df_columns(self):
        df = summarise_results(self.res_100, LAMBDA_PLUS, LAMBDA_MINUS)
        required = {"n", "estimator", "mean", "rel_bias", "rmse", "std"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    def test_summary_df_rows(self):
        df = summarise_results(self.res_100, LAMBDA_PLUS, LAMBDA_MINUS)
        assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    def test_rmse_positive(self):
        df = summarise_results(self.res_100, LAMBDA_PLUS, LAMBDA_MINUS)
        assert (df["rmse"] > 0).all(), "All RMSE values should be positive"

    def test_std_positive(self):
        df = summarise_results(self.res_100, LAMBDA_PLUS, LAMBDA_MINUS)
        assert (df["std"] > 0).all(), "All std values should be positive"


# ---------------------------------------------------------------------------
# Tests: qualitative properties matching paper's claims
# ---------------------------------------------------------------------------

class TestQualitativeProperties:
    @classmethod
    def setup_class(cls):
        cls.df = run_full_study(
            C=C, T=T,
            lambda_plus=LAMBDA_PLUS, lambda_minus=LAMBDA_MINUS,
            sample_sizes=SAMPLE_SIZES, R=R_SMALL, seed=SEED,
        )

    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_mle_positive_bias_lplus(self, n):
        """MLE should have positive relative bias for λ+."""
        row = self.df[(self.df["n"] == n) & (self.df["estimator"] == "MLE λ+")]
        assert float(row["rel_bias"].iloc[0]) > 0, \
            f"MLE λ+ bias should be positive at n={n}"

    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_mle_positive_bias_lminus(self, n):
        """MLE should have positive relative bias for λ-."""
        row = self.df[(self.df["n"] == n) & (self.df["estimator"] == "MLE λ-")]
        assert float(row["rel_bias"].iloc[0]) > 0, \
            f"MLE λ- bias should be positive at n={n}"

    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_firth_bias_smaller_than_mle(self, n):
        """Firth bias should be smaller than MLE bias."""
        for param in ["λ+", "λ-"]:
            mle_bias = abs(float(self.df[(self.df["n"] == n) & (self.df["estimator"] == f"MLE {param}")]["rel_bias"].iloc[0]))
            firth_bias = abs(float(self.df[(self.df["n"] == n) & (self.df["estimator"] == f"Firth {param}")]["rel_bias"].iloc[0]))
            assert firth_bias < mle_bias, \
                f"|Firth bias| should < |MLE bias| at n={n}, {param}: {firth_bias:.4f} vs {mle_bias:.4f}"

    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    def test_firth_rmse_leq_mle_rmse(self, n):
        """Firth RMSE ≤ MLE RMSE for all n."""
        for param in ["λ+", "λ-"]:
            mle_rmse = float(self.df[(self.df["n"] == n) & (self.df["estimator"] == f"MLE {param}")]["rmse"].iloc[0])
            firth_rmse = float(self.df[(self.df["n"] == n) & (self.df["estimator"] == f"Firth {param}")]["rmse"].iloc[0])
            assert firth_rmse <= mle_rmse + 1e-6, \
                f"Firth RMSE should ≤ MLE RMSE at n={n}, {param}: {firth_rmse:.4f} vs {mle_rmse:.4f}"

    def test_bias_shrinks_with_n(self):
        """For MLE, bias should shrink as n increases."""
        for param in ["λ+", "λ-"]:
            biases = [
                abs(float(self.df[(self.df["n"] == n) & (self.df["estimator"] == f"MLE {param}")]["rel_bias"].iloc[0]))
                for n in SAMPLE_SIZES
            ]
            # Should be non-increasing on average
            assert biases[0] >= biases[-1], \
                f"MLE bias for {param} did not shrink with n: {biases}"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_deterministic_with_same_seed(self):
        """Two runs with same seed produce identical results."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        r1 = simulate_vg_bias_study(100, 50, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng1)
        r2 = simulate_vg_bias_study(100, 50, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng2)
        assert np.allclose(r1["lam_plus_mle"], r2["lam_plus_mle"])

    def test_different_seeds_differ(self):
        """Two runs with different seeds produce different results (with overwhelming probability)."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        r1 = simulate_vg_bias_study(100, 50, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng1)
        r2 = simulate_vg_bias_study(100, 50, C, T, LAMBDA_PLUS, LAMBDA_MINUS, rng2)
        assert not np.allclose(r1["lam_plus_mle"], r2["lam_plus_mle"])

    def test_analytical_df_shape(self):
        df = analytical_expectations(C, T, LAMBDA_PLUS, LAMBDA_MINUS, SAMPLE_SIZES)
        assert len(df) == len(SAMPLE_SIZES) * 2, "Should have 2 rows per sample size (λ+ and λ-)"

    def test_mle_rel_bias_formula(self):
        """Relative bias of MLE = 1/(k-1) exactly."""
        for n in SAMPLE_SIZES:
            k = n * T * C
            analytic_bias = 1.0 / (k - 1)
            df = analytical_expectations(C, T, LAMBDA_PLUS, LAMBDA_MINUS, [n])
            row = df[df["param"] == "λ+"]
            assert abs(float(row["rel_bias_MLE"].iloc[0]) - analytic_bias) < 1e-12
