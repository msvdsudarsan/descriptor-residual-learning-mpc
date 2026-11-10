"""Analytic gradient of the projected-predictor cost against central differences when the clipping of the learned residual is inactive and active.
The clip level (last column of the coefficient array) is lowered until the clip is active for most samples.  Output: results/gradient_clip_check.csv"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import network_experiments as ne
from network_gradient import mpc_cost_net_analytic

rows = []
for name in ('case9', 'case39'):
    c = ne.build_case(name, load_scale=ne.STRESS[name]); th, de, _ = ne.equilibrium(c); ng = c['ng']
    rel_ref = de - np.sum(c['M'] * de) / np.sum(c['M']); N = 5
    for clip in (1.0, 0.05, 0.01):
        errs = []
        for rep in range(6):
            rng = np.random.default_rng(10 + rep)
            coef = np.zeros((ng, ne.NF + 1)); coef[:, :ne.NF] = rng.normal(0, 0.05, (ng, ne.NF)); coef[:, ne.NF] = clip
            de0 = de + rng.normal(0, 0.1, ng); w0 = rng.normal(0, 0.3, ng); th0, _ = ne.solve_theta(th, de0, c['Pd'], *ne.args_of(c), 1e-12, 30)
            U = rng.normal(0, 0.5, N * ng) * np.tile(c['umax'], N)
            a = (de0, w0, th0, np.zeros(ng), N, ne.TS, ne.NSUB, 1, 1, coef, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'],
                 rel_ref, c['umax'], 0.5, 20.0, 0.02, 0.2, np.radians(20.0), np.array([0, .015, .03, .04, .055, .06]), 3e3)
            J1, g1 = mpc_cost_net_analytic(U, *a); gfd = np.zeros_like(U)
            for j in range(len(U)):
                e = np.zeros_like(U); e[j] = 1e-6; gfd[j] = (ne.mpc_cost_net(U + e, *a) - ne.mpc_cost_net(U - e, *a)) / 2e-6
            errs.append(np.linalg.norm(g1 - gfd) / np.linalg.norm(gfd))
        rows.append((name, clip, float(np.max(errs)), float(np.median(errs))))
        print(name, 'clip', clip, 'max rel error %.2e  median %.2e' % (np.max(errs), np.median(errs)), flush=True)
ne.csv('gradient_clip_check.csv', ['case', 'clip_level', 'max_rel_error', 'median_rel_error'], rows)
