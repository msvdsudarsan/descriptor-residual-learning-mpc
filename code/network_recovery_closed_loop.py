"""Closed loop on case9 with the physics-plus-projection fallback added to the mild-step test, and a recovery test in which the
dispatch step is undone.  case39 is not repeated: its input sits at the bound in all predictive controllers (Section 7.14)."""
import sys
import numpy as np
from network_reconcile import *

T_END = 9.5


def run(name='case9', nreal=6):
    c0, th_eq, de_eq, coef = nominal_fit(name); lam_ref = regularity(c0, th_eq, de_eq)[0]
    q0, _ = calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, name)
    ref0 = de_eq - np.sum(c0['M'] * de_eq) / np.sum(c0['M']); rows = []
    th0w, _ = solve_theta(th_eq, de_eq, c0['Pd'], *args_of(c0), 1e-12, 30)
    for scen, mu1, t_rest in (('mild 1.05', 1.05, None), ('mild 1.10', 1.10, None), ('recovery 1.10', 1.10, T_REST)):
        c1 = build_case(name, load_scale=mu1); th1, de1, _ = equilibrium(c1); ref1 = de1 - np.sum(c1['M'] * de1) / np.sum(c1['M'])
        on = lambda t: t >= T_SHIFT - 1e-9 and (t_rest is None or t < t_rest - 1e-9)
        ref_fn = lambda t: ref1 if on(t) else ref0
        pm_nom = lambda t: c0['Pm']; pm_true = lambda t: c1['Pm'] if on(t) else c0['Pm']
        for kd in ('A2', 'rec', 'pp'): make_rec_mpc(c0, coef, pm_nom, ref_fn, q0, A_GATE, kd, np.random.default_rng(0))(0.0, de_eq, np.zeros(c0['ng']), th0w, np.zeros(c0['ng']), c0['Pd'])
        rng = np.random.default_rng(6201 if t_rest is None else 6301); cols = {}
        Tcl = 6.0 if t_rest is None else T_END
        for i in range(nreal):
            load = (rng.uniform(0.04, 0.08), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
            de0 = de_eq + rng.normal(0, 0.03, c0['ng']); w0 = rng.normal(0, 0.1, c0['ng'])
            th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
            for nm in ('C0', 'A0', 'A2', 'pp', 'gate', 'rec', 'oracle'):
                if nm == 'C0': ctrl = static_ctrl(c0)
                elif nm == 'A0':
                    b0 = make_net_mpc(c0, 'A0', coef, ref0, np.zeros(6)); b1 = make_net_mpc(c0, 'A0', coef, ref1, np.zeros(6))
                    ctrl = (lambda b0, b1: (lambda t, de, w, th, up, Pd: (b1 if on(t) else b0)(t, de, w, th, up, Pd)))(b0, b1)
                elif nm == 'oracle': ctrl = make_rec_mpc(c0, coef, pm_true, ref_fn, q0, A_GATE, 'A2', np.random.default_rng(77 + i))
                else: ctrl = make_rec_mpc(c0, coef, pm_nom, ref_fn, q0, A_GATE, nm, np.random.default_rng(77 + i))
                lg = simulate_shift(c0, c1, T_SHIFT, th0, de0, w0, ctrl, Tcl, load, t_restore=t_rest); t = lg['t']
                during = (t >= T_SHIFT + 0.5) & ((t < t_rest) if t_rest else True); after = (t >= (t_rest + 0.5)) if t_rest else np.zeros_like(t, bool)
                m = dict(w_during=float(np.sqrt(np.mean(lg['w'][during] ** 2))), w_after=float(np.sqrt(np.mean(lg['w'][after] ** 2))) if after.any() else np.nan)
                if nm in ('gate', 'rec'):
                    tt = np.array(ctrl.state['tau']); tl = t[:len(tt)]
                    m['tau_min'] = float(tt[tl >= T_SHIFT].min()); m['tau_after'] = float(np.mean(tt[tl >= t_rest + 0.5])) if t_rest else np.nan
                cols.setdefault(nm, []).append(m)
            print(name, scen, 'run', i, {n: round(v[-1]['w_during'], 4) for n, v in cols.items()}, flush=True)
        for nm, ms in cols.items():
            rows.append((scen, nm, float(np.mean([m['w_during'] for m in ms])), float(np.mean([m['w_after'] for m in ms])), float(np.mean([m.get('tau_min', np.nan) for m in ms])), float(np.mean([m.get('tau_after', np.nan) for m in ms]))))
    csv(f'{name}_recovery_closed_loop.csv', ['scenario', 'controller', 'w_rms_during', 'w_rms_after_restore', 'tau_min', 'tau_mean_after_restore'], rows)


if __name__ == '__main__':
    run()
