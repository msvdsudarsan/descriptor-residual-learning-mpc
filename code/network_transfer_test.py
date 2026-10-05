"""N9: does a residual trained across a range of loadings transfer better than one trained at a single loading?
The predictor knows the loading and the dispatch (so only the learned residual differs between variants)."""
import numpy as np
from network_experiments import *

MU_TRAIN = (0.6, 0.8, 1.0, 1.25, 1.5)
MU_TEST = {'train': (0.6, 1.0, 1.5), 'interpolation': (0.7, 0.9, 1.1, 1.4), 'extrapolation': (1.8,)}


def case_at(name, mu):
    c = build_case(name, load_scale=mu); th, de, res = equilibrium(c)
    assert res < 1e-8
    return c, th, de


def fit_pooled(name, mus, ntr=8, nval=3, seed0=1100):
    TH, Y, THv, Yv = [], [], [], []
    for i, mu in enumerate(mus):
        c, th, de = case_at(name, mu)
        for tr, bucket in ((make_training(c, th, de, ntr, seed0 + i, 1e-3), 0), (make_training(c, th, de, nval, seed0 + 50 + i, 1e-3), 1)):
            for t in tr:
                a, b = weak_form_net(c, t, th)
                (TH if bucket == 0 else THv).append(a); (Y if bucket == 0 else Yv).append(b)
    TH = np.concatenate(TH); Y = np.concatenate(Y); THv = np.concatenate(THv); Yv = np.concatenate(Yv)
    ng = TH.shape[1]; coef = np.zeros((ng, NF + 1))
    for i in range(ng):
        cands = []
        for rg in (1e-6, 1e-4, 1e-2):
            for tq in (0.002, 0.01, 0.03, 0.1, 0.3):
                cc = stlsq(TH[:, i, :], Y[:, i], ridge=rg, thresh=tq)
                cands.append((np.sqrt(np.mean((Yv[:, i] - THv[:, i, :] @ cc) ** 2)), int(np.sum(cc != 0)), cc))
        eb = min(x[0] for x in cands)
        cc = min([x for x in cands if x[0] <= 1.05 * eb], key=lambda x: (x[1], x[0]))[2]
        coef[i, :NF] = cc; coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i]))
    return coef


def fit_single(name, mu=1.0):
    c, th, de = case_at(name, mu)
    tr = make_training(c, th, de, 30, 1001, 1e-3); va = make_training(c, th, de, 8, 1002, 1e-3)
    return fit_net(c, tr, va, th)[0]


def main(name):
    coef_nom = fit_single(name); coef_pool = fit_pooled(name, MU_TRAIN)
    rows = []
    for kind, mus in MU_TEST.items():
        for mu in mus:
            c, th, de = case_at(name, mu)
            runs = make_test_runs(c, th, de, 4, 3100 + int(100 * mu), 0.06)
            r = {}
            for lab, nm, cf in (('A0', 'A0', coef_nom), ('nom', 'A2', coef_nom), ('pool', 'A2', coef_pool)):
                e = predict_windows(c, runs, nm, cf)[0]
                r[lab] = float(np.sqrt(np.mean(np.vstack(e) ** 2)))
            rows.append((kind, mu, r['A0'], r['nom'], r['pool'], r['nom'] / r['A0'], r['pool'] / r['A0']))
            print(name, rows[-1], flush=True)
    csv(f'{name}_transfer.csv', ['kind', 'load_scale', 'rmse_A0', 'rmse_A2_single_loading', 'rmse_A2_pooled', 'ratio_single', 'ratio_pooled'], rows)


if __name__ == '__main__':
    for nm in ('case9', 'case39'):
        main(nm)
