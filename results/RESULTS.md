# Experiment Results Summary

Lévy-Process α-Divergence and Information Geometry — Computational Reproduction

All experiments are purely synthetic (no empirical market data). Results are
fully reproducible by running `python exp/exp{1,2,3}_*.py` from the repository root.

---

## Experiment 1 — General Lévy α-Divergence and Information Geometry

**Objective**: Verify the theorem-level Lévy-process α-divergence formula,
Fisher information matrix, and α-connections numerically against direct integral
representations; confirm α-duality and zero-divergence identities.

**Parameter grid** (implementation choice):
- Two GTS parameter pairs over α ∈ {-1, -0.5, 0, 0.5, 1} at T = 1.
- Pair 0: GTS(C₊=0.3, C₋=0.2, a₊=0.5, a₋=0.7, λ₊=1.0, λ₋=1.0) vs (λ₊=1.5, λ₋=2.0).
- Pair 1: GTS(C₊=0.1, C₋=0.3, a₊=0.3, a₋=1.7, λ₊=0.5, λ₋=0.5) vs (λ₊=1.5, λ₋=1.0).

### Key Metrics

| Metric | Value |
|--------|-------|
| Max α-duality absolute error | **0.00e+00** |
| Max zero-divergence error (D^α(P\|\|P)) | **0.00e+00** |
| Max compact vs numeric closed-form error | **6.60e-07** |
| Fisher matrix — max \|numerical − direct\| | **8.20e-06** |
| Fisher matrix — max \|numerical − closed form\| | **8.20e-06** |
| Fisher matrix symmetry | **✓ confirmed** |
| α-connection — max \|numerical − direct\| (diagonal) | **2.85e-05** (Γ_{lp,lp,lp}), **2.02e-03** (Γ_{lm,lm,lm}) |

### Findings

- α-duality D^α(P‖Q) = D^{-α}(Q‖P) holds to machine precision for all tested pairs
  and all five α values including endpoints ±1.
- Zero divergence D^α(P‖P) = 0 confirmed exactly for all tested families.
- The compact theorem expression for Δ_T^α agrees with the Frullani-based (VG) and
  Gamma-function-based (GTS/CTS) closed forms to ≤ 6.6 × 10⁻⁷.
- Fisher information recovered by finite-differences of D^0 matches direct theorem
  integrals to ≤ 8.2 × 10⁻⁶; matrix is symmetric.
- α-connections from third-order finite differences match direct theorem integrals;
  residuals ≤ 2 × 10⁻³ (limited by step-size trade-off in triple finite differences).

**Output files**: `exp1_divergence_checks.csv`, `exp1_divergence_vs_alpha.png`,
`exp1_duality_errors.png`.

---

## Experiment 2 — Family-Specific Closed Forms (GTS, CTS, VG, Merton)

**Objective**: Reproduce model-specific divergences, Fisher matrices, α-connections,
potentials, dual coordinates, priors, and canonical-divergence identities.
Cross-check all closed forms against general theorem integrals.

### GTS Family

| Metric | Value |
|--------|-------|
| Max closed-form vs numeric error | **6.60e-07** |
| Hess(K_GTS) vs g — max entry error | **1.47e-06** |
| Γ^(α=0)_{λ+,λ+,λ+} from K vs closed form | **5.48e-07** |
| Γ^(α=0)_{λ-,λ-,λ-} from K vs closed form | **2.04e-07** |
| Γ^(1) max absolute value (should = 0) | **0.00e+00** |
| η dual coordinates: \|∇K − closed form\| | **9.68e-10** |

### CTS / CGMY Family

| Metric | Value |
|--------|-------|
| Max closed-form vs numeric error | **1.21e-06** |
| Bregman B_K(ξ,η) vs D^1(P_ξ‖P_η) max error | **3.30e-09** |
| Bregman B_K(η,ξ) vs D^{-1}(P_ξ‖P_η) max error | **1.08e-08** |
| Superharmonic prior: k=0.1 < a/2=0.25, Δ_LB < 0 | **✓ confirmed** |
| Superharmonic prior: k=0.2 < a/2=0.25, Δ_LB < 0 | **✓ confirmed** |
| Superharmonic prior: k=0.4 > a/2=0.25, fails | **✓ correctly excluded** |
| CTS → VG limit (a=0.01), abs diff in divergence | **1.19e-05** |

### VG Family

| Metric | Value |
|--------|-------|
| Max closed-form vs numeric error | **5.93e-16** (near machine epsilon) |
| Log-coordinate metric — max deviation from TC·I | **0.00e+00** |
| Bregman B_K vs D^1 | **✓ matches exactly** |
| Jeffreys prior formula | **J(ξ) ∝ TC/(λ₊λ₋) confirmed** |

### Merton Jump-Diffusion

| Metric | Value |
|--------|-------|
| Risk-neutral moment error: \|∫(eˣ-1)ν(dx) − analytic\| | **2.08e-17** |
| Zero divergence D^0(P‖P) | **0.00e+00** |
| g_{σσ} vs 2T/σ² (case 0: σ=0.1) | **200.00 = 200.00 ✓** |
| g_{σσ} vs 2T/σ² (case 1: σ=0.2) | **50.00 = 50.00 ✓** |
| Fisher matrix off-diagonal max | **< 4e-14** (essentially zero) |

### Findings

- All four families' closed-form divergences agree with general theorem integrals
  to numerical tolerance (≤ 1.2 × 10⁻⁶ for GTS/CTS, ≤ 6 × 10⁻¹⁶ for VG).
- For affine families (GTS, CTS, VG): Hessian of K recovers the Fisher matrix;
  third derivative of K scaled by (1−α)/2 recovers the α-connection; Γ^(1) = 0
  exactly; canonical Bregman divergences match D^(±1).
- CTS divergence converges to VG divergence as a → 0 (|Δ| = 1.2 × 10⁻⁵ at a=0.01).
- Merton Fisher matrix diagonal structure confirmed; g_{σσ} = 2T/σ² exactly.
- CTS superharmonic prior candidates ρ = λ^k for k < a/2 are confirmed superharmonic;
  k ≥ a/2 correctly fails the Laplace–Beltrami non-positivity condition.

**Output files**: `exp2_gts_divergence.csv`, `exp2_cts_divergence.csv`,
`exp2_vg_divergence.csv`, `exp2_merton_checks.csv`,
`exp2_cts_canonical_divergence.csv`, `exp2_cts_vg_limit.csv`,
`exp2_family_divergences.png`, `exp2_cts_to_vg_limit.png`.

---

## Experiment 3 — VG Monte Carlo Bias-Reduction Study (MLE vs Firth)

**Objective**: Reproduce the paper's VG simulation study comparing MLE and
Firth estimators for λ₊ and λ₋ via sufficient statistics.

**Setup**: C=0.1, T=1, λ₊^true=1.0, λ₋^true=1.5, n∈{50,100,200,500},
R=5000 replications, seed=42 (implementation choice; not paper-specified).

### Monte Carlo Summary Table

| n | Rel. Bias MLE λ₊ | Rel. Bias Firth λ₊ | Rel. Bias MLE λ₋ | Rel. Bias Firth λ₋ | RMSE MLE λ₊ | RMSE Firth λ₊ | RMSE MLE λ₋ | RMSE Firth λ₋ |
|---|---|---|---|---|---|---|---|---|
| 50  | +0.244 | −0.005 | +0.245 | −0.004 | 0.749 | 0.566 | 1.114 | 0.841 |
| 100 | +0.110 | −0.001 | +0.113 | +0.002 | 0.407 | 0.353 | 0.626 | 0.542 |
| 200 | +0.057 | +0.004 | +0.057 | +0.004 | 0.255 | 0.236 | 0.386 | 0.357 |
| 500 | +0.019 | −0.002 | +0.021 | +0.001 | 0.148 | 0.143 | 0.226 | 0.219 |

### Analytical Cross-Check (exact expectations via Gamma(k, rate=λ))

For S ~ Gamma(k=nTC, rate=λ):
- E[λ̂_MLE] = k/(k−1) × λ (positively biased for k > 1)
- E[λ̂_Firth] = λ (exactly unbiased)

| n | k=nTC | E[MLE λ₊] analytic | MC mean | E[Firth λ₊] analytic | MC mean |
|---|---|---|---|---|---|
| 50  | 5  | 1.2500 | 1.2437 | 1.0000 | 0.9950 |
| 100 | 10 | 1.1111 | 1.1095 | 1.0000 | 0.9986 |
| 200 | 20 | 1.0526 | 1.0572 | 1.0000 | 1.0043 |
| 500 | 50 | 1.0204 | 1.0188 | 1.0000 | 0.9984 |

### Findings

- **MLE positive bias**: confirmed for all n and both parameters; bias shrinks as n
  grows (relative bias 24% at n=50, 2% at n=500).
- **Firth near-unbiased**: relative bias < 0.5% for all n; analytically exactly
  unbiased under the sufficient-statistic Gamma model.
- **Firth lower RMSE**: Firth RMSE ≤ MLE RMSE for all n and both parameters;
  advantage largest at small n (24% RMSE reduction at n=50 for λ₊).
- **Convergence**: both estimators converge as n increases; gap effectively closes
  by n=500.
- **Qualitative agreement with paper's Figure 1**: ✓ all claims reproduced.

**Output files**: `exp3_mc_long_form.csv`, `exp3_mc_wide_table.csv`,
`exp3_analytic_comparison.csv`, `exp3_figure1_replication.png`.

---

## Test Suite

135 tests run via `pytest tests/ -v`:

| Module | Tests | Passed | Failed |
|--------|-------|--------|--------|
| `tests/test_exp1.py` | 47 | 47 | 0 |
| `tests/test_exp2.py` | 49 | 49 | 0 |
| `tests/test_exp3.py` | 39 | 39 | 0 |
| **Total** | **135** | **135** | **0** |

All edge-case tests pass including: zero divergence, α-duality, positivity,
admissibility guards, parameter validation, large-λ-difference stability
(fixed via log-domain integrand formulation), Firth estimator bias properties,
and CTS→VG limit convergence.
