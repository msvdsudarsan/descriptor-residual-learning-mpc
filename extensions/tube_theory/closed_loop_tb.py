"""T-B: closed loop, case9 at load scale 2.75, line-angle limit 23 deg, actuator authority umax x scale.  Controllers: C0 static law, A2 (no margin), A3p (paper conformal envelope on total angle error),
A3h (hybrid margin H2 from conformal_hybrid.py: q_span + calibrated load term).  Paired realizations (same IC/load for all controllers).
Usage: python3 closed_loop_tb.py <umax_scale> [nreal]"""
import sys, json; sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from env import *
scale = float(sys.argv[1]); nreal = int(sys.argv[2]) if len(sys.argv) > 2 else 10; LIM = 23.0; gam = np.radians(LIM)
c, th_eq, de_eq, coef = make_env('case9', stress=True); c['umax'] = c['umax'] * scale
rel_ref = de_eq - np.sum(c['M'] * de_eq) / np.sum(c['M'])
H = json.load(open('results/hybrid_margin_case9_stress.json')); eps_h = np.r_[0.0, H['H2']]
eps_p = pd.read_csv('results/case9_tube_envelope_stress.csv')['eps_line_angle_rad'].values
print(f'umax scale {scale}; eps paper (deg):', np.round(np.degrees(eps_p), 2), ' eps hybrid H2 (deg):', np.round(np.degrees(eps_h), 2))
rng = np.random.default_rng(6001); rows = []
for i in range(nreal):
    load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_eq + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
    th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
    for nm, mk in (('C0', None), ('A2', np.zeros(6)), ('A3p', eps_p), ('A3h', eps_h)):
        ctrl = static_ctrl(c) if nm == 'C0' else make_net_mpc(c, 'A3' if nm.startswith('A3') else 'A2', coef, rel_ref, mk, gam=gam)
        lg = simulate_plant(c, th0, de0, w0, ctrl, 6.0, load)
        ang = np.array([np.max(np.abs(t[c['lf']] - t[c['lt']])) for t in lg['th']]); dd = lg['de'] - lg['de'].mean(1, keepdims=True) - (rel_ref - rel_ref.mean())[None, :]
        rows.append(dict(run=i, ctrl=nm, peak_deg=np.degrees(ang.max()), frac_over=float(np.mean(ang > gam)), viol_int=float(np.sum(np.maximum(0, ang - gam)) * TS),
                         w_rms=float(np.sqrt(np.mean(lg['w'][len(lg['w']) // 3:] ** 2))), rot_rms=float(np.sqrt(np.mean(dd ** 2))), u_rms=float(np.sqrt(np.mean((lg['u'] / c['umax']) ** 2)))))
    print(i, [(r['ctrl'], round(r['peak_deg'], 2), round(r['frac_over'], 3)) for r in rows[-4:]], flush=True)
df = pd.DataFrame(rows); df.to_csv(f'results/closed_loop_tb_x{scale:g}.csv', index=False)
def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = 2.262 * x.std(ddof=1) / np.sqrt(n) if n > 9 else 2.365 * x.std(ddof=1) / np.sqrt(n); return m, h
print(f'\n=== umax x{scale:g}, limit {LIM} deg, {nreal} paired realizations; mean +- 95% t-interval ===')
for nm in ('C0', 'A2', 'A3p', 'A3h'):
    d = df[df.ctrl == nm]; print('%-4s' % nm, *['%s %.4g+-%.2g' % (k, *ci(d[k])) for k in ('peak_deg', 'frac_over', 'viol_int', 'w_rms', 'rot_rms', 'u_rms')])
for nm in ('A3p', 'A3h'):
    a = df[df.ctrl == 'A2'].sort_values('run'); b = df[df.ctrl == nm].sort_values('run')
    print('paired diff %s - A2:' % nm, *['%s %.3g+-%.2g' % (k, *ci(b[k].values - a[k].values)) for k in ('peak_deg', 'frac_over', 'viol_int', 'w_rms', 'u_rms')])
