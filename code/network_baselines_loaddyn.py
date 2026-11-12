"""Stronger identification baselines and unmodelled load dynamics (case9 / case39).

All residual models are used inside the SAME Newton-projected RK4 predictor, so the comparison isolates the learned residual.
Training data and validation data are those of the main study (30 + 8 noisy trajectories, noise 1e-3).  Methods:
  P0   physics model with projection, no residual
  OURS sparse weak-form regression (the method of the paper; coefficients of results/coef_nominal_*.npy)
  SF   the same sparse regression on noisy finite differences (strong form), same library and same model selection
  RIDGE dense ridge regression, weak form, same 13-term library, no sparsity
  MLP  one small neural network per generator (2 x 32 tanh), inputs sin(psi), cos(psi), w, u, trained on the weak-form targets
Test scenarios (plant differs from the predictor, which always uses the nominal physics and the nominal load law):
  in-family     plant of the study
  load-lag      the plant load follows the nominal fluctuation through a first-order lag of time constant tau_L
  saliency x1.5 the unmodelled saliency term of the plant is 1.5 times larger than in the training plant
Metric: pooled RMSE of the state error over a 1 s horizon (centre-of-inertia angle removed), and its ratio to the physics-only
predictor A0 of the paper (first-order algebraic update, no residual).  Reduced lossless models only.
"""
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor
from network_validity_gate import *

NSEEDS = {'case9': 8, 'case39': 5}
TAUS = (0.5, 1.0)


def strong_form(c, tr, th_eq):
    de, w, U, Pd, h = tr['de'], tr['w'], tr['U'], tr['Pd'], tr['h']
    K = U.shape[0]; th = np.zeros((K + 1, c['n'])); guess = th_eq
    for k in range(K + 1):
        guess, _ = solve_theta(guess, de[k], Pd[k], *args_of(c), 1e-11, 30); th[k] = guess
    Y = np.zeros((K, c['ng'])); TH = np.zeros((K, c['ng'], NF))
    for k in range(K):
        a = swing_acc(de[k], w[k], U[k], th[k], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
        Y[k] = (w[k + 1] - w[k]) / h - a
        for i in range(c['ng']):
            TH[k, i] = feats(de[k, i] - th[k, c['gbus'][i]], w[k, i], U[k, i])
    return TH, Y


def stack(pairs):
    return np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])


def fit_sparse(TH, Y, THv, Yv, ng, ridges=(1e-6, 1e-4, 1e-2), threshes=(0.002, 0.01, 0.03, 0.1, 0.3)):
    coef = np.zeros((ng, NF + 1))
    for i in range(ng):
        cands = []
        for rg in ridges:
            for tq in threshes:
                cc = stlsq(TH[:, i, :], Y[:, i], ridge=rg, thresh=tq)
                cands.append((np.sqrt(np.mean((Yv[:, i] - THv[:, i, :] @ cc) ** 2)), int(np.sum(cc != 0)), cc))
        eb = min(x[0] for x in cands)
        cc = min([x for x in cands if x[0] <= 1.05 * eb], key=lambda x: (x[1], x[0]))[2]
        coef[i, :NF] = cc; coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i]))
    return coef


def fit_ridge(TH, Y, THv, Yv, ng, ridges=(1e-6, 1e-4, 1e-2, 1e-1, 1.0)):
    coef = np.zeros((ng, NF + 1))
    for i in range(ng):
        A, b = TH[:, i, :], Y[:, i]; best = None
        for rg in ridges:
            cc = np.linalg.solve(A.T @ A + rg * len(b) * np.eye(NF), A.T @ b)
            e = np.sqrt(np.mean((Yv[:, i] - THv[:, i, :] @ cc) ** 2))
            if best is None or e < best[0]:
                best = (e, cc)
        coef[i, :NF] = best[1]; coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i]))
    return coef


class MLPRes:
    IDX = [0, 1, 6, 9]                                   # sin psi, cos psi, w, u in the feature vector

    def __init__(self, TH, Y, ng):
        self.nets, self.mu, self.sd, self.cb = [], [], [], 1.5 * np.max(np.abs(Y), axis=0)
        for i in range(ng):
            X = TH[:, i, :][:, self.IDX]; mu, sd = X.mean(0), X.std(0) + 1e-9
            m = MLPRegressor(hidden_layer_sizes=(32, 32), activation='tanh', alpha=1e-4, max_iter=400, early_stopping=True,
                             n_iter_no_change=20, random_state=0)
            m.fit((X - mu) / sd, Y[:, i]); self.nets.append(m); self.mu.append(mu); self.sd.append(sd)
        self.W = [[w.copy() for w in m.coefs_] for m in self.nets]; self.B = [[b.copy() for b in m.intercepts_] for m in self.nets]

    def __call__(self, de, w, u, th, c):
        out = np.zeros(len(de))
        for i in range(len(de)):
            f = feats(de[i] - th[c['gbus'][i]], w[i], u[i])[self.IDX]
            h = (f - self.mu[i]) / self.sd[i]
            h = np.tanh(h @ self.W[i][0] + self.B[i][0]); h = np.tanh(h @ self.W[i][1] + self.B[i][1])
            out[i] = np.clip((h @ self.W[i][2] + self.B[i][2])[0], -self.cb[i], self.cb[i])
        return out


def coef_res(coef):
    return lambda de, w, u, th, c: res_acc(de, w, u, th, c['gbus'], coef)


def rk4_proj(c, de, w, th, u, h, Pd, res):
    def acc(d, ww, t_):
        a = swing_acc(d, ww, u, t_, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
        return a + res(d, ww, u, t_, c) if res is not None else a
    a1 = acc(de, w, th); d2 = de + .5 * h * w; w2 = w + .5 * h * a1
    th2 = solve_theta(th, d2, Pd, *args_of(c), 1e-10, 12)[0]
    a2 = acc(d2, w2, th2); d3 = de + .5 * h * w2; w3 = w + .5 * h * a2
    th3 = solve_theta(th2, d3, Pd, *args_of(c), 1e-10, 12)[0]
    a3 = acc(d3, w3, th3); d4 = de + h * w3; w4 = w + h * a3
    th4 = solve_theta(th3, d4, Pd, *args_of(c), 1e-10, 12)[0]
    a4 = acc(d4, w4, th4)
    dn = de + h * (w + 2 * w2 + 2 * w3 + w4) / 6; wn = w + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
    thn = solve_theta(th4, dn, Pd, *args_of(c), 1e-10, 12)[0]
    return dn, wn, thn


def generic_sqerr(c, runs, res, N=10, stride=4):
    """Sum of squared state errors and count, per run (projected predictor with residual function res)."""
    out = []
    for load, lg in runs:
        s, n = 0.0, 0
        for k in range(0, len(lg['t']) - 1 - N, stride):
            Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
            for j in range(N):
                u = lg['u'][k + j]
                for _ in range(NSUB):
                    de, w, th = rk4_proj(c, de, w, th, u, TS / NSUB, Pd_k, res)
                coi_p = np.sum(c['M'] * de) / np.sum(c['M']); de_t = lg['de'][k + j + 1]; coi_t = np.sum(c['M'] * de_t) / np.sum(c['M'])
                e = np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + j + 1]]
                s += float(e @ e); n += e.size
        out.append((s, n))
    return out


def a0_sqerr(c, runs, coef):
    out = []
    for r in runs:
        e = predict_windows(c, [r], 'A0', coef, stride=4)[0]; E = np.vstack(e); out.append((float(np.sum(E ** 2)), E.size))
    return out


def scenario_runs(c, th_eq, de_eq, kind, n, seed, T=8.0):
    rng = np.random.default_rng(seed); runs = []
    cp = dict(c)
    if kind == 'saliency x1.5':
        cp['kappa'] = c['kappa'] * 1.5
    for _ in range(n):
        de0 = de_eq + rng.normal(0, 0.10, c['ng']); w0 = rng.normal(0, 0.3, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        load = (rng.uniform(0.0, 0.06), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28)); lp = load
        if kind.startswith('load-lag'):
            tau = float(kind.split('=')[1]); a = load[1] * tau
            lp = (load[0] / np.sqrt(1 + a * a), load[1], load[2] - np.arctan(a))
        lg = simulate_plant(cp, th0, de0, w0, static_ctrl(c), T, lp)
        runs.append((load, lg))                          # the predictor is given the nominal (unlagged) load law `load`
    return runs


def run(name):
    c, th_eq, de_eq, coef_o = nominal_fit(name)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    TH, Y = stack([weak_form_net(c, t, th_eq) for t in train]); THv, Yv = stack([weak_form_net(c, t, th_eq) for t in val])
    THs, Ys = stack([strong_form(c, t, th_eq) for t in train]); THsv, Ysv = stack([strong_form(c, t, th_eq) for t in val])
    methods = {'P0 (physics; projected)': None,
               'OURS (sparse; weak form)': coef_res(coef_o),
               'SF (sparse; strong form)': coef_res(fit_sparse(THs, Ys, THsv, Ysv, c['ng'])),
               'RIDGE (dense; weak form)': coef_res(fit_ridge(TH, Y, THv, Yv, c['ng'])),
               'MLP (2x32; weak form)': MLPRes(TH, Y, c['ng'])}
    # consistency: the generic predictor with the stored coefficients reproduces predict_windows('A2')
    chk = scenario_runs(c, th_eq, de_eq, 'in-family', 1, 4242)
    e2 = np.vstack(predict_windows(c, chk, 'A2', coef_o, stride=4)[0]); g = generic_sqerr(c, chk, methods['OURS (sparse; weak form)'])[0]
    print(name, 'consistency (A2 vs generic) sq-error:', float(np.sum(e2 ** 2)), g[0], flush=True)
    assert abs(float(np.sum(e2 ** 2)) - g[0]) <= 1e-6 * g[0], 'generic predictor does not reproduce A2'
    rows = []
    for kind in ['in-family'] + [f'load-lag tau={t}' for t in TAUS] + ['saliency x1.5']:
        runs = scenario_runs(c, th_eq, de_eq, kind, NSEEDS[name], 7700)
        a0 = a0_sqerr(c, runs, coef_o); A0 = sum(s for s, _ in a0); N0 = sum(n for _, n in a0)
        for m, res in methods.items():
            sq = generic_sqerr(c, runs, res); S = sum(s for s, _ in sq); N = sum(n for _, n in sq)
            per = [np.sqrt((s / n) / (s0 / n0)) for (s, n), (s0, n0) in zip(sq, a0)]
            rows.append((kind, m, float(np.sqrt(S / N)), float(np.sqrt((S / N) / (A0 / N0))), float(np.min(per)), float(np.max(per))))
            print(name, rows[-1], flush=True)
    csv(f'{name}_baselines_loaddyn.csv', ['scenario', 'method', 'rmse_1s', 'ratio_to_A0', 'min_run_ratio', 'max_run_ratio'], rows)


def run_noise(name, noises=(1e-3, 1e-2, 3e-2)):
    """Same four regressors trained with measurement noise sigma on the training and validation trajectories; in-family test."""
    c, th_eq, de_eq, _ = nominal_fit(name); rows = []
    runs = scenario_runs(c, th_eq, de_eq, 'in-family', NSEEDS[name], 7700)
    for sg in noises:
        train = make_training(c, th_eq, de_eq, 30, 1001, sg); val = make_training(c, th_eq, de_eq, 8, 1002, sg)
        TH, Y = stack([weak_form_net(c, t, th_eq) for t in train]); THv, Yv = stack([weak_form_net(c, t, th_eq) for t in val])
        THs, Ys = stack([strong_form(c, t, th_eq) for t in train]); THsv, Ysv = stack([strong_form(c, t, th_eq) for t in val])
        co = fit_sparse(TH, Y, THv, Yv, c['ng']); cs = fit_sparse(THs, Ys, THsv, Ysv, c['ng'])
        meth = {'OURS (sparse; weak form)': (coef_res(co), int(np.count_nonzero(co[:, :NF]))),
                'SF (sparse; strong form)': (coef_res(cs), int(np.count_nonzero(cs[:, :NF]))),
                'RIDGE (dense; weak form)': (coef_res(fit_ridge(TH, Y, THv, Yv, c['ng'])), NF * c['ng']),
                'MLP (2x32; weak form)': (MLPRes(TH, Y, c['ng']), -1)}
        a0 = a0_sqerr(c, runs, co); A0 = sum(s for s, _ in a0); N0 = sum(n for _, n in a0)
        for m, (res, nz) in meth.items():
            sq = generic_sqerr(c, runs, res); S = sum(s for s, _ in sq); N = sum(n for _, n in sq)
            rows.append((sg, m, float(np.sqrt((S / N) / (A0 / N0))), nz)); print(name, rows[-1], flush=True)
    csv(f'{name}_baselines_noise.csv', ['noise_sigma', 'method', 'ratio_to_A0', 'active_terms'], rows)


if __name__ == '__main__':
    if sys.argv[1:2] == ['noise']:
        for nm in (sys.argv[2:] or ['case9', 'case39']): run_noise(nm)
    else:
        for nm in (sys.argv[1:] or ['case9', 'case39']): run(nm)
