"""Nuisance-shift test of the trust gate: ordinary parameter errors that are not a dispatch mismatch.
The plant has scaled line reactances, inertia or damping (whole run), the predictor and the gate keep the nominal parameters;
measurement noise is 1e-3 on every state.  Reported: share of samples with tau < 0.5, mean trust, largest deviation of the dispatch
scale rho from one for the reconciling gate, and the A2/A0 error ratio for context."""
import sys
import numpy as np
from network_reconcile import *

SCEN = (('line_scale', 0.85), ('line_scale', 1.15), ('inertia_scale', 0.8), ('inertia_scale', 1.2), ('damping_scale', 0.8), ('damping_scale', 1.2), ('joint', None))


def run(name):
    c0, th_eq, de_eq, coef = nominal_fit(name); lam_ref = regularity(c0, th_eq, de_eq)[0]
    q0, _ = calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, name); rows = []; rng = np.random.default_rng(9701)
    for key, val in SCEN:
        kw = {key: val} if key != 'joint' else dict(line_scale=1.12, inertia_scale=0.85, damping_scale=1.15)
        cp = build_case(name, **kw); thp, dep, rp = equilibrium(cp)
        runs = make_test_runs(cp, thp, dep, 4 if name == 'case39' else 6, 9710 + int(100 * (val or 1)) + len(rows), 0.06, T=8.0)
        ff, mt, dr, r0, r2 = [], [], [], [], []
        for i, (load, lg) in enumerate(runs):
            lg = with_pd(lg, cp, load)
            tg, _, _ = reconcile_series(c0, coef, lg, q0, A_GATE, False, np.random.default_rng(20 + i))
            tr, _, rho = reconcile_series(c0, coef, lg, q0, A_GATE, True, np.random.default_rng(20 + i))
            ff.append(np.mean(tg[1:] < 0.5)); mt.append(np.mean(tg[1:])); dr.append(np.max(np.abs(rho - 1)))
            e0 = predict_windows(c0, [(load, lg)], 'A0', coef, stride=6)[0]; e2 = predict_windows(c0, [(load, lg)], 'A2', coef, stride=6)[0]
            r0.append(np.mean(np.vstack(e0) ** 2)); r2.append(np.mean(np.vstack(e2) ** 2))
        lab = f'{key} {val}' if key != 'joint' else 'joint (lines 1.12; inertia 0.85; damping 1.15)'
        rows.append((lab, float(np.mean(ff)), float(np.mean(mt)), float(np.max(dr)), float(np.sqrt(np.mean(r2) / np.mean(r0)))))
        print(name, rows[-1], flush=True)
    csv(f'{name}_nuisance.csv', ['scenario', 'false_fallback_share', 'mean_trust', 'max_abs_rho_minus_1', 'ratio_A2_over_A0'], rows)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['case9', 'case39']): run(nm)
