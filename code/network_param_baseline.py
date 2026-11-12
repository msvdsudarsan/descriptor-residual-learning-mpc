"""Phase 3 baseline: does the library residual beat a parametric update of the existing physics?
The 'parametric' correction re-estimates the damping coefficient (omega term) or damping plus a cubic speed term (friction-like), each with a constant (the residual has a large mean), by ridge regression on the
same weak-form training data, and is used in the same projected predictor and the same MPC (A2 structure) as the library residual.
Prediction: one-second state RMSE relative to A0 on in-family test runs (nominal loading, 8 s, seeds 9100+).  Control: RMS speed deviation on the closed-loop
realizations of the main study (elevated loading, seed 6001) with the analytic-gradient A2 controller."""
import sys
import numpy as np
from network_cl_recalibration import setup, metrics
from network_baselines_loaddyn import stack, fit_ridge, scenario_runs
from network_validity_gate import *

VARIANTS = {'library residual (paper)': None, 'damping + constant': [6, 11], 'damping + cubic speed + constant': [6, 8, 11]}   # feature indices: 6 = omega, 8 = omega^3, 11 = constant (a damping-only fit without the constant is biased: the residual has a large mean)


def restricted(TH, Y, THv, Yv, ng, idx):
    coef = np.zeros((ng, NF + 1))
    for i in range(ng):
        A = TH[:, i, :][:, idx]; b = Y[:, i]; best = None
        for rg in (1e-8, 1e-6, 1e-4, 1e-2):
            cc = np.linalg.solve(A.T @ A + rg * len(b) * np.eye(len(idx)), A.T @ b); e = np.sqrt(np.mean((Yv[:, i] - THv[:, i, :][:, idx] @ cc) ** 2))
            if best is None or e < best[0]: best = (e, cc)
        coef[i, idx] = best[1]; coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i]))
    return coef


def run(name):
    rows_p, rows_c = [], []
    # ---- prediction at nominal loading
    c, th_eq, de_eq, coef_lib = nominal_fit(name)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    TH, Y = stack([weak_form_net(c, t, th_eq) for t in train]); THv, Yv = stack([weak_form_net(c, t, th_eq) for t in val])
    cf = {k: (coef_lib if v is None else restricted(TH, Y, THv, Yv, c['ng'], v)) for k, v in VARIANTS.items()}
    nrun = 8 if name == 'case9' else 5
    for scen, runs in (('in-family', make_test_runs(c, th_eq, de_eq, nrun, 9100, 0.06, T=8.0)), ('saliency x1.5', scenario_runs(c, th_eq, de_eq, 'saliency x1.5', nrun, 9200)), ('load lag 1.0 s', scenario_runs(c, th_eq, de_eq, 'load-lag tau=1.0', nrun, 9300))):
        e0 = np.vstack(predict_windows(c, runs, 'A0', coef_lib, stride=4)[0]); r0 = np.sqrt(np.mean(e0 ** 2))
        for k, co in cf.items():
            e = np.vstack(predict_windows(c, runs, 'A2', co, stride=4)[0]); rows_p.append((scen, k, float(np.sqrt(np.mean(e ** 2)) / r0)))
            print(name, scen, 'prediction ratio to A0', k, rows_p[-1][2], flush=True)
    csv(f'{name}_param_baseline_prediction.csv', ['scenario', 'correction', 'rmse_ratio_to_A0'], rows_p)
    # ---- closed loop at elevated loading
    cs, th_s, de_s, coef_s, rel_ref = setup(name); T = 6.0 if name == 'case9' else 5.0; nreal = 6 if name == 'case9' else 4
    trs = make_training(cs, th_s, de_s, 30, 1001, 1e-3); vls = make_training(cs, th_s, de_s, 8, 1002, 1e-3)
    THs, Ys = stack([weak_form_net(cs, t, th_s) for t in trs]); THsv, Ysv = stack([weak_form_net(cs, t, th_s) for t in vls])
    cfs = {k: (coef_s if v is None else restricted(THs, Ys, THsv, Ysv, cs['ng'], v)) for k, v in VARIANTS.items()}
    rng = np.random.default_rng(6001)
    for i in range(nreal):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_s + rng.normal(0, 0.03, cs['ng']); w0 = rng.normal(0, 0.1, cs['ng'])
        th0, _ = solve_theta(th_s, de0, cs['Pd'], *args_of(cs), 1e-12, 30)
        for k in ['C0 static law', 'A0 physics only'] + list(VARIANTS):
            if k.startswith('C0'): ctrl = static_ctrl(cs)
            elif k.startswith('A0'): ctrl = make_net_mpc(cs, 'A0', coef_s, rel_ref, np.zeros(6), analytic=False)
            else: ctrl = make_net_mpc(cs, 'A2', cfs[k], rel_ref, np.zeros(6), analytic=True)
            m = metrics(cs, simulate_plant(cs, th0, de0, w0, ctrl, T, load), GAMMA); rows_c.append((i, k, m['w_rms'], m['peak']))
    csv(f'{name}_param_baseline_closed_loop.csv', ['run', 'controller', 'w_rms', 'peak_deg'], rows_c)
    for k in ['C0 static law', 'A0 physics only'] + list(VARIANTS):
        print(name, 'closed loop', k, 'mean w_rms', float(np.mean([r[2] for r in rows_c if r[1] == k])), flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['case9', 'case39']): run(nm)
