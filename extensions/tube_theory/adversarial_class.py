"""What the load class {lamp<=0.08, omega in [1,6]} (with learned-predictor model mismatch) can really produce: attained worst case over the class.
Any deterministic bound valid for the whole class is >= this number; so (bound / attained) is the part of the looseness that is the bound's fault."""
import sys; sys.path.insert(0, 'code')
import numpy as np
from env import *
name = sys.argv[1]; N = 10; LAMP = 0.08
c, th_eq, de_eq, coef = make_env(name); mode, use_res = MODES['A2']; rng = np.random.default_rng(77)
ts = TS * np.arange(1, N + 1); worst_ang = np.zeros(N); worst_sp = np.zeros(N); typ_ang = [[] for _ in range(N)]; arg = None
for seed in range(4):
    de0 = de_eq + rng.normal(0, 0.10, c['ng']); w0 = rng.normal(0, 0.3, c['ng']); th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
    for om in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        for ph in np.linspace(0, 2 * np.pi, 5)[:-1]:
            load = (LAMP, om, ph); lg = simulate_plant(c, th0, de0, w0, static_ctrl(c), 4.0, load)
            for k in range(0, len(lg['t']) - 1 - N, 2):
                Pd_k = c['Pd'] * (1 + LAMP * np.sin(om * lg['t'][k] + ph)); de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
                for j in range(N):
                    for _ in range(NSUB):
                        de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
                    dp, tp = lg['de'][k + j + 1], lg['th'][k + j + 1]
                    a = np.abs(angles(c, th, de) - angles(c, tp, dp)).max(); worst_sp[j] = max(worst_sp[j], np.ptp(de - dp)); typ_ang[j].append(a)
                    if a > worst_ang[j]: worst_ang[j] = a
K = constants(c, coef, 45.0, 2.4); T = tube(K, 0.1806 if name == 'case9' else 0.36, LAMP, 6.0, 1.5, ts, fixed_omega=1.5)
print(name, 'attained worst case over the class (lamp=0.08 fixed, omega in 1..6, 4 phases, 4 ICs) vs bound (rho_sup, Omega_bar=1.5)')
print(' j  t   attained max angle err | attained max span | median angle err | Theorem-1 angle bound | bound/attained')
for j in range(N): print('%2d %.1f   %.4g | %.4g | %.4g | %.4g | %.1f' % (j + 1, ts[j], worst_ang[j], worst_sp[j], np.median(typ_ang[j]), T['ang'][j], T['ang'][j] / worst_ang[j]))
