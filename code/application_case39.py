"""Application study on the 39-bus network: an unannounced scheduling update (load and dispatch step together), with the
predictor still holding the old dispatch.  Two parts.

  scan       : prediction after unannounced steps of 2, 4, 6, 8 and 15 percent, with the same runs, seeds and segments as the other
               persistent steps of network_reconcile.py (3, 5, 10, 25, 50 percent), so that the two files form one scan; A0, A2, A2 with
               the true dispatch (oracle), the gate, the gate with reconciliation.
  authority  : closed loop after a 5 percent step for three values of the actuator authority (the input bound is scaled), controllers
               C0, A0, A2, A2 with reconciliation, A2 with the true dispatch.

Usage: python3 code/application_case39.py scan | authority [scale]   (one scale writes ..._authority_x<scale>.csv; merge_authority.py joins them)
Writes results/case39_application_scan.csv and results/case39_application_authority.csv.
"""
import sys
import numpy as np
from network_reconcile import *

NAME = 'case39'


def setup():
    c0, th_eq, de_eq, coef = nominal_fit(NAME); lam_ref = regularity(c0, th_eq, de_eq)[0]
    q0, _ = calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, NAME)
    return c0, th_eq, de_eq, coef, q0


def scan():
    c0, th_eq, de_eq, coef, q0 = setup(); rows = []
    for mu1 in (1.02, 1.04, 1.06, 1.08, 1.15):
        seed = 9510 + int(100 * mu1)                      # same rule as the persistent-step scenarios of network_reconcile.py
        runs = make_runs(c0, th_eq, de_eq, mu1, None, 4, seed, 0.06, 8.0, NAME)
        res = evaluate_runs(c0, coef, runs, None, q0, seed + 5)
        rm = seg_rmse(res, T_SHIFT + 0.6, 7.0)
        dr, dd, mt, rt = event_metrics(res, T_SHIFT, T_SHIFT + 2.0)
        rho = float(np.mean([np.mean(r['rho'][(r['t'] >= T_SHIFT + 0.6) & (r['t'] <= 7.5)]) for r in res]))
        rows.append((mu1, *[float(x) for x in rm], rho, dr, dd, mt))
        print(NAME, mu1, [round(float(x), 3) for x in rm], 'rho', round(rho, 4), 'det', dr, dd, flush=True)
    csv(f'{NAME}_application_scan.csv', ['mu_after', 'rmse_A0', 'rmse_A2', 'rmse_oracle', 'rmse_gate', 'rmse_reconcile', 'mean_rho_estimate', 'detection_rate', 'detection_delay_s', 'min_trust'], rows)


def authority(nreal=3, mu1=1.05, Tcl=5.5, scales=(1.0, 3.0, 6.0), tag=''):
    c0, th_eq, de_eq, coef, q0 = setup(); rows = []
    ref0 = de_eq - np.sum(c0['M'] * de_eq) / np.sum(c0['M'])
    for sc in scales:
        c0s = dict(c0); c0s['umax'] = c0['umax'] * sc
        c1 = build_case(NAME, load_scale=mu1); c1['umax'] = c1['umax'] * sc
        th1, de1, _ = equilibrium(c1); ref1 = de1 - np.sum(c1['M'] * de1) / np.sum(c1['M'])
        ref_fn = lambda t: ref0 if t < T_SHIFT - 1e-9 else ref1
        pm_nom = lambda t: c0['Pm']; pm_true = lambda t: c1['Pm'] if t >= T_SHIFT - 1e-9 else c0['Pm']
        rng = np.random.default_rng(6201); cols = {}
        for i in range(nreal):
            load = (rng.uniform(0.04, 0.08), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
            de0 = de_eq + rng.normal(0, 0.03, c0['ng']); w0 = rng.normal(0, 0.1, c0['ng'])
            th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
            for nm in ('C0', 'A0', 'A2', 'rec', 'oracle'):
                if nm == 'C0': ctrl = static_ctrl(c0s)
                elif nm == 'A0':
                    b0 = make_net_mpc(c0s, 'A0', coef, ref0, np.zeros(6)); b1 = make_net_mpc(c0s, 'A0', coef, ref1, np.zeros(6))
                    ctrl = (lambda b0, b1: (lambda t, de, w, th, up, Pd: (b0 if t < T_SHIFT - 1e-9 else b1)(t, de, w, th, up, Pd)))(b0, b1)
                elif nm == 'oracle': ctrl = make_rec_mpc(c0s, coef, pm_true, ref_fn, q0, A_GATE, 'A2', np.random.default_rng(77 + i))
                else: ctrl = make_rec_mpc(c0s, coef, pm_nom, ref_fn, q0, A_GATE, nm, np.random.default_rng(77 + i))
                lg = simulate_shift(c0s, c1, T_SHIFT, th0, de0, w0, ctrl, Tcl, load)
                post = lg['t'] >= T_SHIFT + 0.5
                cols.setdefault(nm, []).append((float(np.sqrt(np.mean(lg['w'][post] ** 2))), float(np.mean(np.abs(lg['u'][post]) >= 0.999 * c0s['umax'])), lg['solve'] * 1e3))
            print(NAME, 'authority', sc, 'run', i, {k: round(v[-1][0], 4) for k, v in cols.items()}, flush=True)
        for nm, v in cols.items():
            rows.append((sc, nm, float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])), float(np.mean([x[2] for x in v]))))
    csv(f'{NAME}_application_authority{tag}.csv', ['authority_scale', 'controller', 'w_rms', 'input_at_bound_share', 'solve_ms'], rows)


if __name__ == '__main__':
    if sys.argv[1] == 'scan': scan()
    elif len(sys.argv) > 2: authority(scales=(float(sys.argv[2]),), tag='_x' + sys.argv[2])   # one authority level, merged afterwards
    else: authority()
