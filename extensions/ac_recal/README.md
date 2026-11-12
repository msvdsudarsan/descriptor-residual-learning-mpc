# extensions/ac_recal/

Recalibrated-physics comparison in the voltage-magnitude (AC algebraic) variant of the reduced swing model: physics-only, recalibrated physics and library residual, 24 operating points (case9 and case39, constant-power and constant-impedance loads, loading factors 0.7 to 1.2). The protocol was written before the runs (`PROTOCOL.md`). This is an AC algebraic-model extension; it is not validation against an independent AC dynamic simulator or measured data. No outages, margins or closed loop.

| file | purpose |
|---|---|
| `PROTOCOL.md` | questions, data, methods, metrics, decision rule, failure handling (fixed before the runs) |
| `ac_recal_comparison.py` | one operating point: `PYTHONPATH=code:extensions/ac_baselines python3 extensions/ac_recal/ac_recal_comparison.py case9\|case39 pq\|z LAMBDA [NTEST]` |
| `run_ac_recal_all.sh` | the 24 points of the protocol (about 15 min on two cores); logs in `logs/` |
| `summarize_ac_recal.py`, `make_supp_table.py` | tables (`tables/ac_recal_all_points.csv`), figure (`tables/Figure_ac_recal.pdf`) and the LaTeX table, from `results/*.json` only |
| `results/` | raw JSON, one file per operating point, with every run and every method |
| `checks/` | the lambda = 1 points of case39 with 8 test runs, as in the earlier AC table, used to confirm that the pipeline reproduces the stored A2/A0 ratios |

The lambda = 1.0 rows reproduce the stored one-second A2/A0 ratios of `extensions/ac_baselines/results` (0.2315, 0.1659, 0.3482, 0.2057). Run once by the author's session; not an independent reproduction.

Definitions. The paired ratios (P1/A0, P2/P1, P3/P1) are per-run RMSE ratios over all ten prediction horizons, averaged over the 12 test runs, with a bootstrap 95% interval over the runs (the same definition as the paired table of `extensions/ac_baselines`); the columns marked `1s` use the one-second error only. `diff P1-P3 (ratio to A0)` is the per-run difference of the ratios to A0 (secondary reading, added after the results were seen). The intervals do not include variation between training draws: each operating point was trained once.

A rerun of one point (case9, constant power, loading 1.0, 12 test runs) in a fresh copy gave the same JSON as the stored one in every field except run times.
