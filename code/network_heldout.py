"""Held-out shift family for the trust gate and the reconciliation: a step of the load by a common factor with a dispatch that does NOT scale
uniformly.  After the step each generator's dispatch is the scaled dispatch times a random factor in [0.8, 1.2], renormalized so that the
power balance of the plant holds.  The gate constant, the calibration and the common-scale reconciliation are those of the main tests, so
this family is outside the one on which they were chosen.  The estimator can only find a common scale, which is the point of the test."""
import sys
import numpy as np
from network_reconcile import *


def run(name):
    c0, th_eq, de_eq, coef = nominal_fit(name); lam_ref = regularity(c0, th_eq, de_eq)[0]
    q0, _ = calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, name); nrun = 6 if name == 'case9' else 4; rows = []
    for mu1 in (1.10, 1.25):
        rng = np.random.default_rng(9801 + int(100 * mu1)); out = []
        base = build_case(name, load_scale=mu1)
        for i in range(nrun):
            c1 = dict(base); f = rng.uniform(0.8, 1.2, c0['ng']); pm = base['Pm'] * f; c1['Pm'] = pm * base['Pm'].sum() / pm.sum()
            de0 = de_eq + rng.normal(0, 0.10, c0['ng']); w0 = rng.normal(0, 0.3, c0['ng'])
            th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
            load = (rng.uniform(0.0, 0.06), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
            lg = simulate_shift(c0, c1, T_SHIFT, th0, de0, w0, static_ctrl(c0), 8.0, load)
            out.append((load, lg, c1))
        res = evaluate_runs(c0, coef, out, None, q0, 9820 + int(100 * mu1))
        rm = seg_rmse(res, T_SHIFT + 0.6, 7.0); dr, dd, mt, rt = event_metrics(res, T_SHIFT, T_SHIFT + 2.0)
        rho = float(np.mean([np.mean(r['rho'][(r['t'] >= T_SHIFT + 0.6)]) for r in res]))
        rows.append((mu1, *[float(x) for x in rm], rho, dr, dd, mt)); print(name, rows[-1], flush=True)
    csv(f'{name}_heldout_dispatch.csv', ['mu_after', 'rmse_A0', 'rmse_A2', 'rmse_oracle', 'rmse_gate', 'rmse_reconcile', 'mean_rho_estimate', 'detection_rate', 'detection_delay_s', 'min_trust'], rows)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['case9', 'case39']): run(nm)
