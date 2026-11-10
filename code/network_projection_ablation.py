"""Prediction ablation: tangent update (A1) vs Newton corrector (A2) vs a Spotorno-type predict-project step (S).
S: after each RK4 step with the tangent (first-order) algebraic update, the pair u=(delta,theta) is projected orthogonally onto
g(delta,theta)=0 by the fixed-Jacobian iteration z = u~ - A^T lam, A=[dg/d delta, dg/d theta] evaluated at u~ (fast variant), 6 iterations or |g|<1e-10.
Same residual coefficients (stored nominal fit), same test runs (seed 2001, 12 runs, load amplitude up to 0.08) as the prediction ablation of network_experiments.py.
Usage: PYTHONPATH=code python3 code/network_projection_ablation.py case9 results/projection_ablation
"""
import sys, os, csv as _csv
import numpy as np
from network_experiments import *   # noqa
import csv as _csv


def proj_step(c, de, th, Pd, iters=6, tol=1e-10):
    ng = c['ng']; n = th.shape[0]; gb = c['gbus']
    u = np.r_[de, th]
    def gfun(v):
        g, _ = alg_res_jac(v[ng:], v[:ng], Pd, *args_of(c)[:3], gb, c['bg'], False); return g
    g, Jm = alg_res_jac(th, de, Pd, *args_of(c)[:3], gb, c['bg'], True)
    A = np.zeros((n, ng + n))
    for i in range(ng):
        A[gb[i], i] = c['bg'][i] * np.cos(de[i] - th[gb[i]])
    A[:, ng:] = -Jm
    AAt = A @ A.T
    lam = np.zeros(n); z = u.copy()
    for _ in range(iters):
        g = gfun(z)
        if np.max(np.abs(g)) < tol: break
        lam = lam + np.linalg.solve(AAt, g)
        z = u - A.T @ lam
    return z[:ng], z[ng:]


def predict_S(c, runs, coef, N=10, stride=4, variant='S'):
    mode, use_res = 0, 1
    err = [[] for _ in range(N)]; gmax = []
    for load, lg in runs:
        K = len(lg['t']) - 1
        for k in range(0, K - N, stride):
            Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy(); gm = 0.0
            for j in range(N):
                u = lg['u'][k + j]
                for _ in range(NSUB):
                    de, w, th = pred_step(de, w, th, u, TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
                    de, th = proj_step(c, de, th, Pd_k)
                coi_p = np.sum(c['M'] * de) / np.sum(c['M']); de_t = lg['de'][k + j + 1]; coi_t = np.sum(c['M'] * de_t) / np.sum(c['M'])
                err[j].append(np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + j + 1]])
                g, _ = alg_res_jac(th, de, Pd_k, *args_of(c)[:3], c['gbus'], c['bg'], False)
                gm = max(gm, float(np.max(np.abs(g))))
            gmax.append(gm)
    return [np.array(e) for e in err], np.array(gmax)


def rmse(e, ng): return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1) / ng)))


if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    c = build_case(name, load_scale=1.0); th_eq, de_eq, _ = equilibrium(c)
    coef = np.load(f'results/coef_nominal_{name}.npy')
    runs = make_test_runs(c, th_eq, de_eq, 12, 2001, 0.08)
    res = {}
    for nm in ('A0', 'A1', 'A2'):
        res[nm] = predict_windows(c, runs, nm, coef)[:2]
    res['S'] = predict_S(c, runs, coef)
    ng = c['ng']
    rows = []
    for nm in ('A0', 'A1', 'A2', 'S'):
        for j in (0, 4, 9):
            rows.append((nm, (j + 1) * TS, rmse(res[nm][0][j], ng), float(np.mean(res[nm][1])), float(np.max(res[nm][1]))))
    # paired per-run ratio at 1 s, S/A2 and A1/A2: per-run RMSE using window order is not stored per run; use window-wise paired ratio of squared errors
    wise = {}
    for nm in ('A1', 'S'):
        a = np.sum(res[nm][0][9] ** 2, axis=1); b = np.sum(res['A2'][0][9] ** 2, axis=1)
        wise[nm] = (float(np.sqrt(np.mean(a) / np.mean(b))), float(np.median(np.sqrt(a / b))))
    with open(f'{out}/{name}_projection_ablation.csv', 'w', newline='') as f:
        w = _csv.writer(f); w.writerow(['model', 'horizon_s', 'state_rmse', 'mean_window_max_abs_g', 'max_abs_g']); w.writerows(rows)
    with open(f'{out}/{name}_projection_ablation_pairs.csv', 'w', newline='') as f:
        w = _csv.writer(f); w.writerow(['model', 'rmse_ratio_to_A2_1s', 'median_window_ratio_1s']); [w.writerow([k, *v]) for k, v in wise.items()]
    for r in rows: print(r)
    print(wise)
