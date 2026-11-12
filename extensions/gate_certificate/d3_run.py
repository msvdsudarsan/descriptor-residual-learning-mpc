"""D3: i.i.d. closed-loop trajectories under the deployed controller A3(lambda): tightening eps = lambda * eps_prod (case9, stress load scale 2.75)."""
import sys, time
sys.path.insert(0, 'code')
import numpy as np
from network_cl_recalibration import *
OUTD = Path('.')
lim = float(sys.argv[1]); lams = [float(x) for x in sys.argv[2].split(',')]; n = int(sys.argv[3]); seed0 = int(sys.argv[4])
name = 'case9'; T = 6.0; gam = np.radians(lim)
f = OUTD / 'coef_case9_stress.npy'
c = build_case(name, load_scale=STRESS[name]); th_eq, de_eq, _ = equilibrium(c); rel_ref = de_eq - np.sum(c['M'] * de_eq) / np.sum(c['M'])
if f.exists(): coef = np.load(f)
else:
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3); coef, _, _ = fit_net(c, train, val, th_eq); np.save(f, coef)
eps_prod = np.loadtxt(RES / 'case9_tube_envelope_stress_recal.csv', delimiter=',', skiprows=1)[:, 1]    # [0, eps_1..eps_5] production policy-matched envelope (rad)
print('eps_prod (rad)', eps_prod, ' deg', np.degrees(eps_prod), flush=True)
for lam in lams:
    t0 = time.time(); rows = []
    rng = np.random.default_rng(seed0)       # same disturbances for every lambda (common random numbers)
    for i in range(n):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_eq + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        ctrl = make_net_mpc(c, 'A3', coef, rel_ref, lam * eps_prod, gam=gam, analytic=True)
        lg = simulate_plant(c, th0, de0, w0, ctrl, T, load)
        ang = np.array([np.max(np.abs(t[c['lf']] - t[c['lt']])) for t in lg['th']])
        de_rel = lg['de'] - (lg['de'] @ c['M'] / np.sum(c['M']))[:, None]
        rows.append([np.degrees(ang.max()), float(np.mean(ang > gam)), float(np.sum(np.maximum(0, ang - gam)) * TS), float(np.sqrt(np.mean(lg['w'][len(lg['w']) // 3:] ** 2))),
                     float(np.sqrt(np.mean((de_rel - rel_ref[None, :]) ** 2))), float(np.sum((lg['u'] / c['umax']) ** 2) * TS), float(np.argmax(ang)), float(np.degrees(ang[0]))])
    R = np.array(rows); np.save(OUTD / f'd3_lim{int(lim)}_lam{lam:g}_n{n}_s{seed0}.npy', R)
    print('limit %g lambda %g n %d: frac>0 %d, frac>0.05 %d, mean peak %.3f, mean w_rms %.4f, argmax-peak sample mean %.1f, initial angle mean %.3f deg; %.1f s' % (lim, lam, n, (R[:, 1] > 0).sum(), (R[:, 1] > 0.05).sum(), R[:, 0].mean(), R[:, 3].mean(), R[:, 6].mean(), R[:, 7].mean(), time.time() - t0), flush=True)
