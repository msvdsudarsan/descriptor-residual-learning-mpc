# Structure-preserving residual learning and predictive control of nonlinear descriptor power-network models

Code, data and results for the manuscript submitted to *Communications in Nonlinear Science and Numerical Simulation* (release 1.0.0, October 2026).

## Contents
| Folder | Content |
|---|---|
| `code/` | `common.py` (thresholded regression, split-conformal quantile, bootstrap), `scalar_benchmark.py` and `scalar_experiments.py` (scalar DAE benchmark), `network_benchmark.py` and `network_experiments.py` (MATPOWER case9 and case39 realizations), `network_gradient.py` (exact gradient of the projected-predictor cost) and `network_analytic_gradient.py` (gradient check and closed-loop timing study), `network_trust.py` (plant with a dispatch step, trust gate, dispatch reconciliation), `network_validity_gate.py` (validity atlas, gate calibration and validation) and `network_reconcile.py`, `network_recovery_closed_loop.py`, `network_nuisance.py`, `network_heldout.py` and `make_reconcile_figures.py` (atlas with the gate boundary, reconciliation, recovery, mild-step and recovery closed loop), `network_tube_test.py` (constraint-active test), `network_extra_tests.py` (out-of-library mismatch and library-complexity sweep), `network_transfer_test.py` (transfer across loadings), `make_pipeline_figure.py` (conceptual Fig. 1 and graphical abstract, with the three-layer structure and the C0-A3 ladder) |
| `data/` | `case9.m`, `case39.m` from the official MATPOWER repository (https://github.com/MATPOWER/matpower/tree/master/data) |
| `results/` | every table quoted in the paper (CSV) and the run logs |
| `figures/` | every figure of the paper (PDF and PNG) |
| `tests/` | consistency tests tying the code to the claims |
| `tools/` | `check_paper_values.py` verifies that the numbers printed in the manuscript match `results/`; `regularity_chain_check.py` checks Proposition 3 (manifold sensitivity and Lipschitz bound) numerically |
| `paper/` | `main.tex` (manuscript source) and `values_used.json` (the values it quotes) |

## Reproduce
```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/descriptor-residual-learning-mpc.git && cd descriptor-residual-learning-mpc
pip install -r requirements.txt
bash run_all.sh        # run time depends on the hardware; the scalar closed-loop stage and case39 are the longest
```
All random numbers use fixed seeds and nothing is downloaded. Timing columns (`solve_ms`) depend on the machine; every other column reproduces exactly. `python tools/check_paper_values.py` compares the regenerated results with the numbers in the paper.

## Model summary
* Scalar benchmark: one angle/frequency pair and one algebraic voltage-like variable; the physics model has wrong gains and omits the cubic term.
* Networks: classical swing generators, all bus angles algebraic (lossless network, unit voltages). Inertia, damping and transient reactance are **declared assumptions** (they are not in MATPOWER); see Appendix A of the paper.
* Variants: A0 physics-only (tangent update), A1 + learned residual, A2 + Newton projection, A3 + penalty tightening by the prediction-error envelope; C0 static feedback.

## Limits (see the paper)
Reduced models, no electromagnetic-transient validation; the residual transfers across loadings when loading and dispatch are known but not under an unannounced dispatch shift; the residual models are thresholded, not sparse, on the networks; the envelope is an empirical margin and not a guarantee; the penalty tightening is inactive in the network experiments; with finite-difference gradients the case39 controller is slower than the sampling period; the exact gradient (A2, A3) brings it below on the machine used, and A0 and A1 still use finite differences; the trust gate and the reconciliation are heuristics without a guarantee.

## Citing
See `CITATION.cff` and `DEPOSIT_TO_ZENODO.md`. MIT License for the code (`LICENSE`); the MATPOWER files keep their own BSD-3-Clause license.

## Scientific scope

This repository reproduces the numerical evidence for structure-preserving residual learning, descriptor-manifold projection, the validity map, trust gating, dispatch reconciliation and exact-gradient predictive control reported in the manuscript. It makes no claim beyond the reduced models and the tested scenarios.
