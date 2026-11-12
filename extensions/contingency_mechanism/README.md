# extensions/contingency_mechanism/

Exploratory diagnostics of the case39 line outages whose held-out coverage of the predictor-twin margin is below the nominal level (lines 8, 10, 11 and 37; pooled coverage below 0.90 in every calibration draw).

| file | purpose |
|---|---|
| `diag_mechanism.py` | recomputes the twin margin and the held-out coverage of each of the 35 case39 outages from calibration seed 5001 and the original test seeds with the same code base; all 35 margins and coverages equal the stored values (difference below 1e-9). Writes `results/mechanism_case39.json` and `results/mechanism_features_case39.csv` |
| `screen_static.py` | plant-free static statistic: the topology shift evaluated at the post-outage equilibrium against the nominal equilibrium (K_post and K_nom), computed from the reduced model only; writes `results/screen_static_case39.csv` |

Interpretation, stated as found:

* The relative margin cushion (the cushion divided by the margin) does **not** distinguish the four low-coverage lines from the others. Ranked from the smallest relative cushion, line 8 is 9th, line 11 is 16th, line 37 is 25th and line 10 is 28th of 35. Seventeen other lines have a relative cushion below 0.05 and reach coverage between 0.90 and 1.00.
* The four lines are not assumed to share a mechanism. The cause is unresolved for lines 8 and 10, and no mechanism is assigned to lines 11 and 37.
* No screening criterion, failure criterion or theorem is claimed. The features are descriptive output and have not been tested as predictors on independent data.
* These diagnostics are not part of `run_all.sh` or `run_checks.sh` and no manuscript value depends on them.

Run from the repository root, for example `PYTHONPATH=code python3 extensions/contingency_mechanism/diag_mechanism.py case39` and `PYTHONPATH=code python3 extensions/contingency_mechanism/screen_static.py`. Run by the author's session; not an independent reproduction.
