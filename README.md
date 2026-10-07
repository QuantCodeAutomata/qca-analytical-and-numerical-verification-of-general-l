# Lévy-Process α-Divergence and Information Geometry

Computational reproduction of analytical and numerical results for parametric
Lévy-process information geometry, including model-specific closed forms and a
Monte Carlo bias-reduction study.

## Repository Structure

```
.
├── data/               # Placeholder (no market data required)
├── src/                # Core implementations
│   ├── levy_base.py    # Abstract Lévy family base class
│   ├── divergence.py   # General α-divergence theorem formulas
│   ├── info_geometry.py # Fisher information and α-connections
│   ├── gts.py          # Generalized Tempered Stable family
│   ├── cts.py          # Classical Tempered Stable / CGMY family
│   ├── vg.py           # Variance Gamma family
│   ├── merton.py       # Merton jump-diffusion family
│   └── vg_simulation.py # VG Monte Carlo simulation utilities
├── exp/                # Experiment scripts
│   ├── exp1_levy_geometry.py
│   ├── exp2_family_specific.py
│   └── exp3_vg_simulation.py
├── tests/              # Pytest test suite
│   ├── test_exp1.py
│   ├── test_exp2.py
│   └── test_exp3.py
└── results/            # Generated outputs (tables, plots, RESULTS.md)
```

## Experiments

### Experiment 1 – General Lévy-Process α-Divergence
Implements and numerically verifies the theorem-level α-divergence, Fisher
information matrix, and α-connections for parametric Lévy processes.

### Experiment 2 – Family-Specific Closed Forms
Reproduces closed-form divergences and information geometry for GTS, CTS/CGMY,
VG, and Merton families, cross-checking against general theorem integrals.

### Experiment 3 – VG Monte Carlo Bias Study
Monte Carlo comparison of MLE vs Firth bias-reduced estimators for VG
parameters, reproducing the paper's Figure 1.

## Usage

```bash
pip install -r requirements.txt
python exp/exp1_levy_geometry.py
python exp/exp2_family_specific.py
python exp/exp3_vg_simulation.py
pytest tests/ -v
```

## Configuration

No API keys are required. All experiments use synthetic parameters.
