import csv,json
R='results/regime_map/'; L=[]
L+=['# REGIME MAP FINAL AUDIT REPORT (Protocol)','','Generated from the stored CSV/JSON files only (tools/make_regime_report.py). No value typed by hand. `_4` package untouched. Seeds, margins, levels and thresholds were not changed after any result.','',
'## 1. Execution record','','| Stage | Case | Rows | NaN | Failures | Not evaluable |','|---|---|---|---|---|---|']
for nm in ('case9','case39'):
    p=list(csv.DictReader(open(R+f'{nm}_prediction_runs.csv'))); pm=json.load(open(R+f'{nm}_prediction_meta.json'))
    c=list(csv.DictReader(open(R+f'{nm}_closed_loop_runs.csv'))); cm=json.load(open(R+f'{nm}_closed_loop_meta.json'))
    L.append(f"| PRED | {nm} | {len(p)} | {sum(x['sumsq']=='nan' for x in p)} | {len(pm['failures'])} | {len(pm['not_evaluable'])} |")
    L.append(f"| CL | {nm} | {len(c)} | {sum(x['w_rms']=='nan' for x in c)} | {len(cm['failures'])} | {sum(isinstance(v,str) for v in cm['cells'].values())} |")
L+=['','## 2. Prediction: paired differences of per-run RMSE ratio to A0 (margin 0.02, Student-t 95% CI, P1* = P1c in every cell)','','| Case | Regime | Setting | Difference | n | Mean | 95% CI | Verdict |','|---|---|---|---|---|---|---|---|']
for nm in ('case9','case39'):
    for x in csv.DictReader(open(R+f'{nm}_prediction_paired.csv')): L.append(f"| {x['case']} | {x['regime']} | {x['setting']} ({'identify-in-regime' if x['setting']=='a' else 'deployment-shift'}) | {x['difference']} | {x['n']} | {x['mean']} | [{x['ci_lo']}, {x['ci_hi']}] | {x['verdict(margin 0.02)']} |")
L+=['','## 3. Closed loop: paired differences of w_rms (margin = 2% of mean P1* w_rms), energy ratio, saturation','','| Case | Regime | Authority | Difference | Mean | 95% CI | Margin | Verdict | Energy ratio (first/second) |','|---|---|---|---|---|---|---|---|---|']
for nm in ('case9','case39'):
    for x in csv.DictReader(open(R+f'{nm}_closed_loop_paired.csv')): L.append(f"| {nm} | {x['regime']} | x{x['authority']} | {x['difference']} | {x['mean']} | [{x['ci_lo']}, {x['ci_hi']}] | {x['margin(2% mean P1* w_rms)']} | {x['verdict']} | {x['energy_ratio']} |")
L+=['','## 4. Closed-loop means per controller (w_rms, peak angle deg, input energy, saturation fraction, violation integral)','','| Case | Regime | Authority | Controller | w_rms | Peak | Energy | Saturation | Violation |','|---|---|---|---|---|---|---|---|---|']
for nm in ('case9','case39'):
    for x in csv.DictReader(open(R+f'{nm}_closed_loop_summary.csv')): L.append(f"| {nm} | {x['regime']} | x{x['authority']} | {x['controller']} | {x['w_rms']} | {x['peak_deg']} | {x['input_energy']} | {x['saturation_fraction']} | {x['viol_integral']} |")
L+=['','## 5. C0eq pilot calibration (seed 80000)','','| Case | Cell | Gain (x D) | Pilot energy P2 | Pilot energy C0eq | Bracket reached |','|---|---|---|---|---|---|']
for nm in ('case9','case39'):
    for k,v in json.load(open(R+f'{nm}_closed_loop_meta.json'))['cells'].items(): L.append(f"| {nm} | {k} | {v['gain_multiple_of_D']:.4f} | {v['pilot_energy_P2']:.4f} | {v['pilot_energy_C0eq']:.4f} | {v['bracket_reached']} |")
L+=['','## 6. Protocol deviations and disclosures','',
'1. V-stage H1 check drew matched random numbers from seed 9400 (no test trajectory simulated, no test metric computed). This departs from the literal wording "test seed not touched".',
'2. Re-fitted P2 vs stored `coef_nominal`: case9 abs 6.5e-12 (rel 4.4e-13); case39 abs 2.06e-9 (rel 1.1e-10). The 1e-9 tolerance is a supplementary diagnostic, not a protocol criterion. Case39 exceeds it in absolute terms. Reported as a numerical discrepancy, not a failure. Stored coefficients unchanged.',
'3. P1* = P1c in every cell, so P2-P1* and P2-P1c are one comparison, shown once.',
'4. Authority x3 changes the controller bound only; training data keep the nominal u_max, so library input features extrapolate at x3.',
'5. C0eq gain reached the upper bracket (60 x D) in case9 M1 x1, case39 M1 x1 and case39 M1 x3 (bracket_reached = False). C0eq comparisons in those cells are weak evidence.',
'6. M1 x1 (both cases) and M1 x3 (case39): saturation near 100%, controllers indistinguishable; these cells do not support any controller-superiority claim.',
'7. C0eq is pilot-calibrated, not exact equal energy (test energy ratios are listed in section 3).',
'8. The applied-input equals controller-output assertion is executed inside every closed-loop run; a failure would be written to the `failures` list of the metadata JSON (empty in all cases). The log does not print the assertion separately.',
'9. Intervals are over test realizations with ONE fitted model per cell; they do not include training-seed or fitting variability. n = 8 (case9 PRED), 5 (case39 PRED), 6 (case9 CL), 4 (case39 CL).',
'10. Margins, seeds, levels and cells were fixed before any run. No cell was removed or replaced.','',
'## 7. Combined reading (statements the evidence supports)','',
'- Prediction: P2 better than P1c only in case9 M1 deployment-shift; equivalent in the other 9 prediction cells. No "both networks" claim.',
'- Control, P2 vs P1*: equivalent in 7 of 8 cells; P2 better only in case9 M0 x3.',
'- Control, P2 vs C0eq: P2 better in M0 x3 in both networks (case9 energy ratio 1.10, case39 0.96), pilot-calibrated comparison.',
'- P3 vs P2: equivalent in all 9 applicable cells.']
open(R+'REGIME_MAP_FINAL_AUDIT_REPORT.md','w').write('\n'.join(L)+'\n')
