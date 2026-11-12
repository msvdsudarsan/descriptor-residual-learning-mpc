# Recalibrated-physics baseline: results (final protocol; working copy only; _4 untouched; manuscript NOT changed)
Margins (locked): prediction 0.02 absolute (ratio to A0); closed loop 2% of the mean P1 w_rms. Student-t 95% CI on paired run-level differences, no resampling.
Models: P2 library residual (paper); P1a/P1b damping recalibration; P1c bounded recalibration of (M_i,D_i,Pm_i) (all fits converged, no bound active, parameter changes up to 29% (D, case39 stress), typically 10-19%); P3 = P1c + library residual; C0eq = static law with gain matched to P2 input energy on pilot seed 80000 (gain 7.17 D case9, 11.63 D case39).
P1* chosen by validation weak-form RMSE = P1c in both cases (0.222/0.203 vs P1a 0.305/0.390, P1b 0.300/0.389; A0 2.39/5.49).

## Disclosure (deviation caught and fixed)
First run (superseded, kept in results/baseline_recal_superseded_P1c_unprojected) evaluated P1c in the unprojected 'A0' predictor, contrary to protocol "same Newton-projected predictor". Found by me on reading the load-lag result (P1c twice as accurate as everything else = confound). Fixed: P1c now uses the same A2 projected predictor/controller with zero residual. All other models are bit-identical between the two runs. Margins, seeds, model family unchanged. 
Checks: pooled ratios of P2/P1a/P1b reproduce the stored CSV (max diff 4.6e-9); closed-loop w_rms re-simulated for all stored controllers reproduces the stored CSV (max diff 4.3e-8). No failed or non-finite runs (failure files empty).

## Prediction (RMSE ratio to A0; P2 / P1a / P1b / P1c / P3)
case9: in-family .193/.201/.198/.207/.193; saliency x1.5 .339/.359/.356/.368/.339; load-lag .110/.114/.111/.118/.110
case39: in-family .148/.150/.149/.155/.148; saliency .359/.368/.368/.383/.360; load-lag .197/.195/.193/.206/.197
Verdict P2 vs P1c: in-family and load-lag: practically EQUIVALENT (both cases; P2 slightly lower, e.g. -0.013, CI [-0.017,-0.009] case9, inside the margin). Saliency x1.5: P2 BETTER than P1c (case9 -0.030 CI [-0.031,-0.028]; case39 -0.024 CI [-0.026,-0.023]; beyond the 0.02 margin). P2 vs P1a: equivalent except case9 saliency (-0.0206, CI [-0.0212,-0.0200], just beyond margin). P3 vs P2: equivalent everywhere (|d|<=0.0011).
Reading: for in-family and load-lag mismatch, recalibrating physical/damping parameters recovers the same accuracy within 0.02 of A0 error; the library gives a measurable extra gain only for the structural (saliency) mismatch that (M,D,Pm) cannot represent.

## Closed loop (w_rms; n=6 case9, n=4 case39 seeds)
case9: C0 1.026, A0 1.027, P2 .876, P1a .875, P1b .874, P1c .877, P3 .876, C0eq .891
case39: C0 1.960, A0 2.067, P2 1.878, P1a 1.878, P1b 1.878, P1c 1.879, P3 1.878, C0eq 1.887
P2 vs P1c, P3 vs P1c, P3 vs P2: EQUIVALENT in both cases (|d|<=0.0016 vs margins 0.0175/0.0376; half-widths <=0.0003). The closed-loop gain of A2 over A0 is reproduced by a recalibrated PHYSICS-ONLY model in the same controller: the library adds no measurable control benefit.
P2 vs energy-matched static law C0eq: case9 d -0.0153, CI [-0.0186,-0.0120], margin 0.0178: INCONCLUSIVE (energy ratio 1.002); case39 d -0.0086, CI [-0.0181,+0.0008], margin 0.0377: EQUIVALENT (energy ratio 0.94, i.e. test energy is NOT equal; pilot-calibrated comparison only). Much of the manuscript's control gain over the fixed-gain static law (14.7% / 4.2%) is therefore recovered by raising the static gain to the same input energy (13.2% / 3.7% below C0 at equal pilot energy).
P2 vs original C0: first model better (as in the manuscript), but C0 uses 36% / 25% less input energy.

## What this means for the paper (author decision needed; nothing changed)
1. The "no recalibrated-physics baseline" limitation can be replaced by these results.
2. The claim that the learned library improves control must be re-worded: the gain comes from correcting the physics (damping/inertia/offset), not from the library terms. Equal-energy static control matches most of it.
3. The library does help prediction under structural mismatch (saliency). This is the defensible residual-learning contribution, together with the projection/regularity/certificate parts, which this experiment does not touch.
4. Limits: 4-6 closed-loop seeds (CIs from t with n-1 df, normality unchecked); one recalibration family; case39 old P1a/P1b/P2 stored (re-evaluated at run level, reproduced); no real-grid data.
