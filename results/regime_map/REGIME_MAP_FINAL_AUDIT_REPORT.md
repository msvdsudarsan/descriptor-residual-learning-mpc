# REGIME MAP FINAL AUDIT REPORT (Protocol)

Generated from the stored CSV/JSON files only (tools/make_regime_report.py). No value typed by hand. `_4` package untouched. Seeds, margins, levels and thresholds were not changed after any result.

## 1. Execution record

| Stage | Case | Rows | NaN | Failures | Not evaluable |
|---|---|---|---|---|---|
| PRED | case9 | 240 | 0 | 0 | 0 |
| CL | case9 | 144 | 0 | 0 | 0 |
| PRED | case39 | 150 | 0 | 0 | 0 |
| CL | case39 | 96 | 0 | 0 | 0 |

## 2. Prediction: paired differences of per-run RMSE ratio to A0 (margin 0.02, Student-t 95% CI, P1* = P1c in every cell)

| Case | Regime | Setting | Difference | n | Mean | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| case9 | M0 | a (identify-in-regime) | P2-P1c | 8 | -0.0117 | [-0.0152, -0.0083] | equivalent |
| case9 | M0 | a (identify-in-regime) | P3-P2 | 8 | 0.0001 | [0.0, 0.0001] | equivalent |
| case9 | M1 | a (identify-in-regime) | P2-P1c | 8 | -0.0095 | [-0.0129, -0.0061] | equivalent |
| case9 | M1 | a (identify-in-regime) | P3-P2 | 8 | 0.0001 | [0.0, 0.0001] | equivalent |
| case9 | M1 | b (deployment-shift) | P2-P1c | 8 | -0.0531 | [-0.0533, -0.0529] | first better |
| case9 | M1 | b (deployment-shift) | P3-P2 | 8 | 0.0005 | [0.0005, 0.0005] | equivalent |
| case9 | M2 | a (identify-in-regime) | P2-P1c | 8 | -0.0042 | [-0.0054, -0.003] | equivalent |
| case9 | M2 | a (identify-in-regime) | P3-P2 | 8 | 0.0004 | [0.0003, 0.0006] | equivalent |
| case9 | M2 | b (deployment-shift) | P2-P1c | 8 | -0.0094 | [-0.0121, -0.0067] | equivalent |
| case9 | M2 | b (deployment-shift) | P3-P2 | 8 | 0.0 | [0.0, 0.0001] | equivalent |
| case39 | M0 | a (identify-in-regime) | P2-P1c | 5 | -0.0118 | [-0.0164, -0.0071] | equivalent |
| case39 | M0 | a (identify-in-regime) | P3-P2 | 5 | 0.0004 | [0.0002, 0.0006] | equivalent |
| case39 | M1 | a (identify-in-regime) | P2-P1c | 5 | -0.0065 | [-0.0091, -0.0039] | equivalent |
| case39 | M1 | a (identify-in-regime) | P3-P2 | 5 | 0.0 | [0.0, 0.0] | equivalent |
| case39 | M1 | b (deployment-shift) | P2-P1c | 5 | -0.0142 | [-0.0144, -0.014] | equivalent |
| case39 | M1 | b (deployment-shift) | P3-P2 | 5 | 0.0002 | [0.0002, 0.0003] | equivalent |
| case39 | M2 | a (identify-in-regime) | P2-P1c | 5 | -0.0049 | [-0.0073, -0.0025] | equivalent |
| case39 | M2 | a (identify-in-regime) | P3-P2 | 5 | 0.005 | [0.0026, 0.0074] | equivalent |
| case39 | M2 | b (deployment-shift) | P2-P1c | 5 | -0.0089 | [-0.013, -0.0049] | equivalent |
| case39 | M2 | b (deployment-shift) | P3-P2 | 5 | 0.0004 | [0.0002, 0.0006] | equivalent |

## 3. Closed loop: paired differences of w_rms (margin = 2% of mean P1* w_rms), energy ratio, saturation

| Case | Regime | Authority | Difference | Mean | 95% CI | Margin | Verdict | Energy ratio (first/second) |
|---|---|---|---|---|---|---|---|---|
| case9 | M0 | x1 | P2-P1* | -0.00112 | [-0.00131, -0.00094] | 0.01754 | equivalent | 1.0081 |
| case9 | M0 | x1 | P2-C0eq | -0.01534 | [-0.01864, -0.01203] | 0.01754 | INCONCLUSIVE | 1.0022 |
| case9 | M0 | x1 | P2-C0 | -0.15054 | [-0.1689, -0.13219] | 0.01754 | first better | 1.5607 |
| case9 | M0 | x1 | P3-P2 | -0.00026 | [-0.00032, -0.00021] | 0.01754 | equivalent | 1.0015 |
| case9 | M0 | x1 | P2-A0 | -0.15118 | [-0.16656, -0.13581] | 0.01754 | first better | 1.3211 |
| case9 | M0 | x3 | P2-P1* | -0.03104 | [-0.03512, -0.02695] | 0.00193 | first better | 1.0468 |
| case9 | M0 | x3 | P2-C0eq | -0.34966 | [-0.36357, -0.33576] | 0.00193 | first better | 1.0965 |
| case9 | M0 | x3 | P2-C0 | -0.90007 | [-0.93897, -0.86117] | 0.00193 | first better | 4.7052 |
| case9 | M0 | x3 | P3-P2 | -0.00131 | [-0.00153, -0.0011] | 0.00193 | equivalent | 1.0049 |
| case9 | M0 | x3 | P2-A0 | -0.57768 | [-0.5884, -0.56695] | 0.00193 | first better | 0.8093 |
| case9 | M1 | x1 | P2-P1* | 0.0 | [0.0, 0.0] | 0.06873 | equivalent | 1.0 |
| case9 | M1 | x1 | P2-C0eq | -2e-05 | [-4e-05, 1e-05] | 0.06873 | equivalent | 1.0026 |
| case9 | M1 | x1 | P2-C0 | -2e-05 | [-2e-05, -1e-05] | 0.06873 | equivalent | 1.02 |
| case9 | M1 | x1 | P3-P2 | 0.0 | [0.0, 0.0] | 0.06873 | equivalent | 1.0 |
| case9 | M1 | x1 | P2-A0 | -3e-05 | [-3e-05, -2e-05] | 0.06873 | equivalent | 1.0197 |
| case9 | M1 | x3 | P2-P1* | -0.02168 | [-0.02489, -0.01847] | 0.02609 | equivalent | 1.0377 |
| case9 | M1 | x3 | P2-C0eq | -0.06645 | [-0.13624, 0.00335] | 0.02609 | INCONCLUSIVE | 1.0082 |
| case9 | M1 | x3 | P2-C0 | -1.03862 | [-1.09085, -0.98638] | 0.02609 | first better | 2.3429 |
| case9 | M1 | x3 | P3-P2 | 0.00175 | [0.00131, 0.00219] | 0.02609 | equivalent | 0.9954 |
| case9 | M1 | x3 | P2-A0 | -0.58445 | [-0.63376, -0.53514] | 0.02609 | first better | 1.3701 |
| case39 | M0 | x1 | P2-P1* | -0.00024 | [-0.00058, 0.00011] | 0.03757 | equivalent | 1.0013 |
| case39 | M0 | x1 | P2-C0eq | -0.00864 | [-0.01807, 0.00079] | 0.03757 | equivalent | 0.9404 |
| case39 | M0 | x1 | P2-C0 | -0.08169 | [-0.09337, -0.07001] | 0.03757 | first better | 1.3272 |
| case39 | M0 | x1 | P3-P2 | 3e-05 | [-2e-05, 7e-05] | 0.03757 | equivalent | 0.9997 |
| case39 | M0 | x1 | P2-A0 | -0.18856 | [-0.20593, -0.17119] | 0.03757 | first better | 1.0513 |
| case39 | M0 | x3 | P2-P1* | -0.00288 | [-0.0033, -0.00247] | 0.0102 | equivalent | 1.0127 |
| case39 | M0 | x3 | P2-C0eq | -0.18799 | [-0.24804, -0.12794] | 0.0102 | first better | 0.9614 |
| case39 | M0 | x3 | P2-C0 | -0.99951 | [-1.06365, -0.93537] | 0.0102 | first better | 3.221 |
| case39 | M0 | x3 | P3-P2 | -0.00076 | [-0.00189, 0.00037] | 0.0102 | equivalent | 0.9962 |
| case39 | M0 | x3 | P2-A0 | -0.51518 | [-0.68305, -0.34732] | 0.0102 | first better | 1.1351 |
| case39 | M1 | x1 | P2-P1* | 0.0 | [0.0, 0.0] | 0.11602 | equivalent | 1.0 |
| case39 | M1 | x1 | P2-C0eq | -4e-05 | [-0.0001, 2e-05] | 0.11602 | equivalent | 1.0014 |
| case39 | M1 | x1 | P2-C0 | -6e-05 | [-7e-05, -5e-05] | 0.11602 | equivalent | 1.0231 |
| case39 | M1 | x1 | P3-P2 | 0.0 | [0.0, 0.0] | 0.11602 | equivalent | 1.0 |
| case39 | M1 | x1 | P2-A0 | -0.0001 | [-0.0002, -1e-05] | 0.11602 | equivalent | 1.0235 |
| case39 | M1 | x3 | P2-P1* | -0.0 | [-0.0, 0.0] | 0.07098 | equivalent | 1.0004 |
| case39 | M1 | x3 | P2-C0eq | -0.0001 | [-0.00026, 6e-05] | 0.07098 | equivalent | 1.0064 |
| case39 | M1 | x3 | P2-C0 | -0.29874 | [-0.31572, -0.28177] | 0.07098 | first better | 1.2973 |
| case39 | M1 | x3 | P3-P2 | -0.0 | [-0.0, 0.0] | 0.07098 | equivalent | 1.0 |
| case39 | M1 | x3 | P2-A0 | -0.1787 | [-0.25396, -0.10345] | 0.07098 | first better | 1.1634 |

## 4. Closed-loop means per controller (w_rms, peak angle deg, input energy, saturation fraction, violation integral)

| Case | Regime | Authority | Controller | w_rms | Peak | Energy | Saturation | Violation |
|---|---|---|---|---|---|---|---|---|
| case9 | M0 | x1 | C0 | 1.02633 | 23.02086 | 0.75574 | 0.35 | 0.0 |
| case9 | M0 | x1 | A0 | 1.02697 | 23.37199 | 0.89281 | 0.59722 | 0.0 |
| case9 | M0 | x1 | P2 | 0.87579 | 23.08267 | 1.17949 | 0.76944 | 0.0 |
| case9 | M0 | x1 | P1* | 0.87691 | 23.06838 | 1.16997 | 0.75463 | 0.0 |
| case9 | M0 | x1 | P3 | 0.87552 | 23.0789 | 1.18131 | 0.76759 | 0.0 |
| case9 | M0 | x1 | C0eq | 0.89112 | 23.07346 | 1.17691 | 0.8287 | 0.0 |
| case9 | M0 | x3 | C0 | 0.96577 | 22.99739 | 0.87842 | 0.0 | 0.0 |
| case9 | M0 | x3 | A0 | 0.64337 | 22.59413 | 5.107 | 0.18241 | 0.0 |
| case9 | M0 | x3 | P2 | 0.0657 | 23.39814 | 4.13313 | 0.01111 | 0.0 |
| case9 | M0 | x3 | P1* | 0.09673 | 23.39775 | 3.94816 | 0.00648 | 0.0 |
| case9 | M0 | x3 | P3 | 0.06438 | 23.4357 | 4.15358 | 0.01019 | 0.0 |
| case9 | M0 | x3 | C0eq | 0.41536 | 22.92441 | 3.76954 | 0.06944 | 0.0 |
| case9 | M1 | x1 | C0 | 3.43663 | 22.4884 | 1.32593 | 0.975 | 0.0 |
| case9 | M1 | x1 | A0 | 3.43664 | 22.52481 | 1.32633 | 0.97778 | 0.0 |
| case9 | M1 | x1 | P2 | 3.43661 | 22.49104 | 1.3524 | 1.0 | 0.0 |
| case9 | M1 | x1 | P1* | 3.43661 | 22.49104 | 1.3524 | 1.0 | 0.0 |
| case9 | M1 | x1 | P3 | 3.43661 | 22.49104 | 1.3524 | 1.0 | 0.0 |
| case9 | M1 | x1 | C0eq | 3.43663 | 22.59806 | 1.34894 | 0.9963 | 0.0 |
| case9 | M1 | x3 | C0 | 2.3215 | 22.47635 | 5.0787 | 0.0 | 0.0 |
| case9 | M1 | x3 | A0 | 1.86733 | 22.82596 | 8.68438 | 0.56944 | 0.0 |
| case9 | M1 | x3 | P2 | 1.28288 | 22.50441 | 11.89885 | 0.88796 | 0.0 |
| case9 | M1 | x3 | P1* | 1.30456 | 22.51173 | 11.4663 | 0.76204 | 0.0 |
| case9 | M1 | x3 | P3 | 1.28463 | 22.5129 | 11.84464 | 0.83981 | 0.0 |
| case9 | M1 | x3 | C0eq | 1.34933 | 22.95519 | 11.80209 | 0.9537 | 0.0 |
| case39 | M0 | x1 | C0 | 1.96006 | 23.647 | 19.59113 | 0.554 | 0.0 |
| case39 | M0 | x1 | A0 | 2.06692 | 23.68251 | 24.73196 | 0.7655 | 0.0 |
| case39 | M0 | x1 | P2 | 1.87837 | 23.66193 | 26.00092 | 0.8475 | 0.0 |
| case39 | M0 | x1 | P1* | 1.8786 | 23.65644 | 25.96593 | 0.849 | 0.0 |
| case39 | M0 | x1 | P3 | 1.87839 | 23.66083 | 25.99242 | 0.847 | 0.0 |
| case39 | M0 | x1 | C0eq | 1.88701 | 23.68092 | 27.64795 | 0.93 | 0.0 |
| case39 | M0 | x3 | C0 | 1.50681 | 23.60501 | 46.77014 | 0.0 | 0.0 |
| case39 | M0 | x3 | A0 | 1.02248 | 23.59882 | 132.7137 | 0.293 | 0.0 |
| case39 | M0 | x3 | P2 | 0.5073 | 23.66154 | 150.64618 | 0.4445 | 0.0 |
| case39 | M0 | x3 | P1* | 0.51018 | 23.63449 | 148.75257 | 0.4425 | 0.0 |
| case39 | M0 | x3 | P3 | 0.50654 | 23.65073 | 150.06667 | 0.4455 | 0.0 |
| case39 | M0 | x3 | C0eq | 0.69529 | 23.6855 | 156.70239 | 0.449 | 0.0 |
| case39 | M1 | x1 | C0 | 5.80105 | 23.4196 | 28.30933 | 0.9775 | 0.0 |
| case39 | M1 | x1 | A0 | 5.80109 | 23.42136 | 28.29801 | 0.9715 | 0.0 |
| case39 | M1 | x1 | P2 | 5.80099 | 23.41968 | 28.962 | 1.0 | 0.0 |
| case39 | M1 | x1 | P1* | 5.80099 | 23.41968 | 28.962 | 1.0 | 0.0 |
| case39 | M1 | x1 | P3 | 5.80099 | 23.41968 | 28.962 | 1.0 | 0.0 |
| case39 | M1 | x1 | C0eq | 5.80102 | 23.41966 | 28.92289 | 0.997 | 0.0 |
| case39 | M1 | x3 | C0 | 3.8478 | 23.43953 | 200.89542 | 0.485 | 0.0 |
| case39 | M1 | x3 | A0 | 3.72776 | 23.61388 | 224.01632 | 0.721 | 0.0 |
| case39 | M1 | x3 | P2 | 3.54906 | 23.41962 | 260.62488 | 0.9995 | 0.0 |
| case39 | M1 | x3 | P1* | 3.54906 | 23.41962 | 260.52326 | 0.9995 | 0.0 |
| case39 | M1 | x3 | P3 | 3.54906 | 23.41962 | 260.6355 | 0.9995 | 0.0 |
| case39 | M1 | x3 | C0eq | 3.54916 | 23.41998 | 258.97533 | 0.9915 | 0.0 |

## 5. C0eq pilot calibration (seed 80000)

| Case | Cell | Gain (x D) | Pilot energy P2 | Pilot energy C0eq | Bracket reached |
|---|---|---|---|---|---|
| case9 | M0_x1 | 7.1750 | 1.1771 | 1.1771 | True |
| case9 | M0_x3 | 4.9609 | 4.0331 | 4.0327 | True |
| case9 | M1_x1 | 59.9968 | 1.3524 | 1.3497 | False |
| case9 | M1_x3 | 50.3298 | 11.8855 | 11.8855 | True |
| case39 | M0_x1 | 11.6282 | 27.2802 | 27.2802 | True |
| case39 | M0_x3 | 5.6675 | 153.7047 | 153.6987 | True |
| case39 | M1_x1 | 59.9968 | 28.9620 | 28.9185 | False |
| case39 | M1_x3 | 59.9968 | 260.6580 | 258.6965 | False |

## 6. Protocol deviations and disclosures

1. V-stage H1 check drew matched random numbers from seed 9400 (no test trajectory simulated, no test metric computed). This departs from the literal wording "test seed not touched".
2. Re-fitted P2 vs stored `coef_nominal`: case9 abs 6.5e-12 (rel 4.4e-13); case39 abs 2.06e-9 (rel 1.1e-10). The 1e-9 tolerance is a supplementary diagnostic, not a protocol criterion. Case39 exceeds it in absolute terms. Reported as a numerical discrepancy, not a failure. Stored coefficients unchanged.
3. P1* = P1c in every cell, so P2-P1* and P2-P1c are one comparison, shown once.
4. Authority x3 changes the controller bound only; training data keep the nominal u_max, so library input features extrapolate at x3.
5. C0eq gain reached the upper bracket (60 x D) in case9 M1 x1, case39 M1 x1 and case39 M1 x3 (bracket_reached = False). C0eq comparisons in those cells are weak evidence.
6. M1 x1 (both cases) and M1 x3 (case39): saturation near 100%, controllers indistinguishable; these cells do not support any controller-superiority claim.
7. C0eq is pilot-calibrated, not exact equal energy (test energy ratios are listed in section 3).
8. The applied-input equals controller-output assertion is executed inside every closed-loop run; a failure would be written to the `failures` list of the metadata JSON (empty in all cases). The log does not print the assertion separately.
9. Intervals are over test realizations with ONE fitted model per cell; they do not include training-seed or fitting variability. n = 8 (case9 PRED), 5 (case39 PRED), 6 (case9 CL), 4 (case39 CL).
10. Margins, seeds, levels and cells were fixed before any run. No cell was removed or replaced.

## 7. Combined reading (statements the evidence supports)

- Prediction: P2 better than P1c only in case9 M1 deployment-shift; equivalent in the other 9 prediction cells. No "both networks" claim.
- Control, P2 vs P1*: equivalent in 7 of 8 cells; P2 better only in case9 M0 x3.
- Control, P2 vs C0eq: P2 better in M0 x3 in both networks (case9 energy ratio 1.10, case39 0.96), pilot-calibrated comparison.
- P3 vs P2: equivalent in all 9 applicable cells.
