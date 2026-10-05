"""N6: constraint-active test of the residual-envelope tightening (case9, tight line-angle limit)."""
import numpy as np, json
from network_experiments import *

def main(limit_deg=23.0, nreal=6, T=6.0):
    name = 'case9'; mu = STRESS[name]
    c = build_case(name, load_scale=mu); th_eq, de_eq, _ = equilibrium(c)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    coef, _, _ = fit_net(c, train, val, th_eq)
    rel_ref = de_eq - np.sum(c['M'] * de_eq) / np.sum(c['M'])
    gam = np.radians(limit_deg); rng = np.random.default_rng(6001); rows = []
    cal = make_test_runs(c, th_eq, de_eq, 4, 7100, 0.11, T=T, lamp_lo=0.07) + mpc_log_runs(c, th_eq, de_eq, coef, rel_ref, 3, 7110, T, 0.11, gam=gam, lamp_lo=0.07)
    eps = envelope_from(envelope_traj(c, cal, coef))
    for i in range(nreal):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
        de0 = de_eq + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        for nm in ('C0', 'A2', 'A3'):
            ctrl = static_ctrl(c) if nm == 'C0' else make_net_mpc(c, nm, coef, rel_ref, eps, gam=gam)
            lg = simulate_plant(c, th0, de0, w0, ctrl, T, load)
            ang = np.array([np.max(np.abs(t[c['lf']] - t[c['lt']])) for t in lg['th']])
            rows.append((i, nm, float(np.degrees(ang.max())), float(np.sum(np.maximum(0, ang - gam)) * TS), float(np.mean(ang > gam)), float(np.sqrt(np.mean(lg['w'][len(lg['w']) // 3:] ** 2)))))
        print(i, rows[-3:], flush=True)
    csv('case9_tube_test.csv', ['run', 'controller', 'angle_peak_deg', 'viol_integral_rad_s', 'frac_time_over_limit', 'w_rms'], rows)
    for nm in ('C0', 'A2', 'A3'):
        r = [x for x in rows if x[1] == nm]
        print(nm, 'peak', np.mean([x[2] for x in r]), 'viol', np.mean([x[3] for x in r]), 'frac', np.mean([x[4] for x in r]), 'wrms', np.mean([x[5] for x in r]))
    json.dump(dict(limit_deg=limit_deg, eps_rad=eps.tolist()), open(RES / 'case9_tube_test.json', 'w'))

if __name__ == '__main__':
    main()
