# Protocol: direct per-line post-contingency calibration vs nominal-calibration predictor-twin margin (written before any run)

Purpose. Quantify what the predictor-twin margin (built from nominal data only) gains or loses against a margin calibrated directly on
post-outage trajectories of the plant, in the spirit of per-contingency conformal screening (Alcantara et al., arXiv 2602.07995). This is our own
implementation of a comparable procedure, not their code and not an exact reproduction.

Data. case39 reduced lossless model, the 35 non-islanding single-line outages of the existing study, same plant simulator, same nominal-topology
predictor, horizon and window rule as `run_contingency.py` (J = 4, N = 10, stride 4, load amplitude up to 0.08), no parameter mismatch.

Score (both methods). s = trajectory-level maximum, over windows and over all line angles and rotor-to-bus angles, of the absolute error of the
nominal-topology predictor against the plant under the outage (the manuscript's score).

Splits (disjoint by construction, separate seed blocks, checked):
- Direct calibration pool: 40 new plant trajectories per outage, seed 40001 + line index.
- Test: the existing 20 held-out trajectories per outage (seed 9101+line+1000) and the 20 replication trajectories (SEEDOFF 10000), 40 per outage.
- Twin, nominal and static-shift margins: the stored margins (20 nominal trajectories, seed 5001); they use no post-outage data.

Margins.
- Direct (primary): split-conformal quantile, alpha = 0.10 (coverage target 0.90), n_cal = 20: the k-th smallest calibration score, k = ceil((n+1)(1-alpha)) = 19.
- Direct, other pre-specified settings: n_cal = 10 (k = 10) and n_cal = 40 (k = 37); and the "max" rule (largest calibration score) used by the twin margin, for n_cal = 10, 20, 40.
- Calibration-draw variability: for each n_cal, 200 random subsets of the 40-trajectory pool (subsets overlap, so this is an approximation of independent draws; it is reported as such).
- Twin, nominal, static-shift: as stored.

Metrics. Per outage: empirical coverage on the 40 test trajectories (Wilson 95% interval), margin, margin width ratio to the median and to the maximum test score,
median and maximum prediction score; failure = coverage < 0.80 (and < 0.90 listed). Aggregate: pooled coverage over the 35 x 40 test cases with a bootstrap-over-outages
95% interval; number of outages below 0.90 and 0.80; geometric-mean margin ratio twin/direct with a bootstrap-over-outages interval; Wilcoxon signed-rank test of the
per-outage log margin ratio (twin vs direct, n_cal = 20) and of the coverage difference. Two-sided, no multiplicity correction; descriptive.

No tuning. alpha, n_cal values, rules, seeds and metrics are fixed here. Nothing is changed after seeing test results. If the test coverage of the direct margin or
the paired comparison is not statistically informative, that is reported as such.

Reproduction: `PYTHONPATH=code python3 extensions/contingency/direct_percontingency.py <start> <stride>` (raw scores to
`results/direct_cal/`), then `python3 extensions/contingency/analyze_direct.py` (tables, summary JSON).
