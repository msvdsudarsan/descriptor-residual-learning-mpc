# extensions/contingency/

Contingency-conditioned prediction margin (manuscript Section 7, Supplement S-contingency). Reduced lossless MATPOWER case9/case39 models only; no AC, no real grid.

| file | purpose |
|---|---|
| `run_contingency.py` | builds the nominal margin, eps+K_static, windowwise K_T, predictor-twin margin and first-order margin from 20 NOMINAL calibration trajectories and tests them on 20 held-out plant trajectories per candidate (`case9|case39 partial|full [start stride]`; env `SEEDOFF` 0/10000/20000, `MISMATCH` 0.05, `CONT_OUT`) |
| `diag_X.py`, `diag_D_margin.py` | diagnosis of Assumption (X) and of the identity e = e_model + D_T; lines 8, 10, 11 |
| `cost_benchmark.py` | wall-clock cost of twin runs (results/cost_benchmark.json) |
| `make_tables.py`, `make_supp_tables.py`, `make_figure_contingency.py` | regenerate the CSV, LaTeX tables, `paper/values_contingency.json` and `figures/Figure_contingency.pdf` from the raw JSON |
| `PROTOCOL_DIRECT_CALIBRATION.md`, `direct_percontingency.py`, `analyze_direct.py`, `check_direct_values.py` | (added November 2026) direct per-outage post-contingency conformal calibration versus the predictor-twin margin: protocol written before the run, raw scores (`results/direct_cal/`), summary (`results/direct_cal_summary.json`, `direct_cal_per_line.csv`, `direct_cal_table.tex`), value checks; own implementation of a comparable rule, not the code of Alcantara et al. Reproduce: `PYTHONPATH=code python3 extensions/contingency/direct_percontingency.py 0 1` then `python3 extensions/contingency/analyze_direct.py` |
| `check_contingency_values.py` | 5 groups of value checks (aggregation, logs vs JSON, manuscript values, seed/initial-condition independence, missed-failure lists) |
| `compare_rerun.py` | compares a fresh rerun with the stored raw JSON |
| `results/`, `logs/` | raw JSON per run, summary CSV (`summary_per_contingency.csv`, `summary_by_set.csv`), logs of every run |

Entry point (repository root): `./run_checks.sh` (tests + tables + checks), `./run_checks.sh --quick` (adds a fresh rerun of the case9 full outages and an exact comparison), `./run_checks.sh --full` (reruns every set and regenerates the direct-calibration scores; in the author's final run it finished with exit 0 in 616 s, see `results/final_checks/verification_final/run_checks_full.log`).

Known failures are kept: case39 lines 8 and 10 (twin coverage 0.55-0.75 in two independent runs), line 11 (original), lines 14 and 17 (5% mismatch). The cause is unresolved. The coverage statement of the manuscript is conditional on Assumption (X), which nominal data cannot establish.

Scope note: the replication (`c2rep_*`) and mismatch (`c2repmis_*`) runs use new held-out test trajectories but the SAME 20 nominal calibration trajectories (seed 5001), so their margins equal those of the original run. Variability over calibration draws was not tested. The mismatch is a uniform random perturbation of up to +-5% of each plant line susceptance. The 35 case39 candidates are the non-islanding lines.
