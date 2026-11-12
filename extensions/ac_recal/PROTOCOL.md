# AC recalibrated-physics comparison: protocol (written before any run)

Question. Does the library residual add predictive value beyond recalibrated physical parameters when the algebraic network includes voltage magnitudes, line resistance, charging, taps and constant-power or constant-impedance loads?

Scope. An AC *algebraic-model extension* of the reduced swing model. Plant and predictor share the same algebraic equations, the plant mismatch is the same designed mismatch as in the main model (saliency multiplied by V, saturating friction, load fluctuation, damping 0.8 D in the model). It is NOT independent AC dynamic validation and NOT real-grid validation. No outages, no margins, no closed loop.

Operating points. case9 and case39; load models constant power (npw=nqw=0) and constant impedance (npw=nqw=2); loading factor lambda in {0.7, 0.8, 0.9, 1.0, 1.1, 1.2} (build_ac load_scale: scales P and Q demand and the dispatch). 24 points. The whole pipeline (nominal point, data, fits, tests) is rerun at each point.

Data (identical for all methods at a point). Training: 30 trajectories, seed 1001, noise 1e-3; validation: 8 trajectories, seed 1002; test: 12 held-out trajectories, seed 2001 (load fluctuation up to 0.08), 10-step (1 s) windows with stride 4. These are the seeds and settings of the existing AC table; the existing table used 8 test runs on case39, so the first 8 runs coincide.

Methods.
A0 physics only, tangent algebraic update (the existing reference).
P0 physics only, Newton projection at every stage.
P1 recalibrated physics: per generator (M, D, Pm) by bounded least squares (+-50 % of nominal) on the weak-form target, ridge toward nominal chosen on the validation set from {1e-8, 1e-6, 1e-4, 1e-2} (as in code/network_recal_baseline.py), Newton projection, no residual.
P2 library residual (coefficients fitted on the original-physics defect, existing fit_net) plus Newton projection (the existing A2).
P3 recalibrated physics plus library residual fitted on the remaining defect of P1 (fit_net unchanged), Newton projection.
A1 (residual, tangent update) is kept for continuity with the existing table.
The centre-of-inertia weighting always uses the original M.

Metrics (per method and point). RMSE of the state error (COI-referenced rotor angles and speeds) at 1 s and pooled over the ten horizons; rms of the line-angle and rotor-to-bus-angle error and of the voltage-magnitude error at 1 s; algebraic residual max|g| at the predicted state (mean and maximum over windows); Newton convergence (a window fails when a non-finite value appears or, for the projected methods, max|g| > 1e-6); wall-clock time per prediction window and per fit.

Primary comparison. Paired per-run ratios on the 12 test runs: P2/P1 and P3/P1 (pooled horizons and 1 s), with a bootstrap 95 % interval over runs (common.bootstrap_ci, 4000 resamples). Decision rule fixed now: "residual adds value" if the upper limit of P3/P1 is below 0.95; "equivalent" if the interval lies within [0.95, 1.05]; "recalibration better" if the lower limit is above 1.05; otherwise "inconclusive". The same rule for P2/P1.

Failures. Nominal-point failure (power flow residual above 1e-8), failure of a plant simulation, non-finite predictions, or Newton non-convergence are recorded at the point and in the report; a run enters a ratio only if all methods are finite for that run, and the number of excluded runs is reported. No point is dropped silently.

Rules. No tuning after seeing test results. Run once; if a bug is found, fix it and rerun all 24 points, keeping the superseded output. No existing file is edited. Raw output and logs are kept in extensions/ac_recal/results and logs.
