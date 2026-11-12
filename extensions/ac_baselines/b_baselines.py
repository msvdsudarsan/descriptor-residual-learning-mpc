"""Task 2 (lossless repo model).  B1: learned algebraic map (ridge, no Newton projection) vs A0/A1/A2, prediction only.
Usage: python b_baselines.py b1 case9|case39"""
import sys, json, time
import numpy as np
from network_cl_recalibration import setup, metrics          # noqa (imports the repo pipeline)
from network_baselines_loaddyn import stack, scenario_runs
from network_validity_gate import *

OUT = RES   # agent_ac/results (code was copied; RES = ROOT/'results' with ROOT = agent_ac)


def make_runs(c, th_eq, de_eq, n, seed, lamp_hi, T=8.0, ic=0.10, wic=0.3):
    rng = np.random.default_rng(seed); runs = []
    for _ in range(n):
        de0 = de_eq + rng.normal(0, ic, c['ng']); w0 = rng.normal(0, wic, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        load = (rng.uniform(0, lamp_hi), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        runs.append((load, simulate_plant(c, th0, de0, w0, static_ctrl(c), T, load)))
    return runs


# ----------------------------------------------------------------------------- learned algebraic map
class AlgMap:
    """theta_hat = dbar + W phi(d, fac),  d = delta - dbar, phi = [1, d, fac-1, (quadratic: d_i d_j, (fac-1) d_i)].  Ridge, alpha by validation."""
    def __init__(self, quad): self.quad = quad

    def phi(self, de, fac):
        d = de - de.mean(); f = [1.0]; f += list(d); f.append(fac - 1.0)
        if self.quad:
            ng = len(d); f += [d[i] * d[j] for i in range(ng) for j in range(i, ng)]; f += list((fac - 1.0) * d)
        return np.array(f)

    def data(self, trs, c):
        X, Y = [], []
        for t in trs:
            fac = t['Pd'].sum(1) / c['Pd'].sum()
            for k in range(t['de'].shape[0]):
                X.append(self.phi(t['de'][k], fac[k])); Y.append(t['th_true'][k] - t['de'][k].mean())
        return np.array(X), np.array(Y)

    def fit(self, tr, va, c):
        X, Y = self.data(tr, c); Xv, Yv = self.data(va, c)
        self.sc = np.sqrt(np.mean(X ** 2, axis=0)) + 1e-12; self.sc[0] = 1.0
        best = None
        for a in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
            A = X / self.sc; W = np.linalg.solve(A.T @ A + a * len(Y) * np.eye(A.shape[1]), A.T @ Y)
            e = np.sqrt(np.mean((Yv - (Xv / self.sc) @ W) ** 2))
            if best is None or e < best[0]: best = (e, a, W)
        self.val_rmse, self.alpha, self.W = best
        self.nfeat = X.shape[1]; return self

    def __call__(self, de, fac):
        return de.mean() + (self.phi(de, fac) / self.sc) @ self.W


def pred_step_map(c, de, w, th, u, h, fac, coef, amap, anchored):
    """RK4 with the learned map supplying theta at every stage (no Newton).  anchored: theta = theta0 + map(d) - map(d0)."""
    A = c
    d0 = de.copy(); m0 = amap(d0, fac)
    def thf(d): return th + (amap(d, fac) - m0) if anchored else amap(d, fac)
    def acc(d, ww, t_):
        return swing_acc(d, ww, u, t_, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric']) + res_acc(d, ww, u, t_, c['gbus'], coef)
    th1 = th if anchored else thf(de)
    a1 = acc(de, w, th1); d2 = de + 0.5 * h * w; w2 = w + 0.5 * h * a1; t2 = thf(d2)
    a2 = acc(d2, w2, t2); d3 = de + 0.5 * h * w2; w3 = w + 0.5 * h * a2; t3 = thf(d3)
    a3 = acc(d3, w3, t3); d4 = de + h * w3; w4 = w + h * a3; t4 = thf(d4)
    a4 = acc(d4, w4, t4)
    dn = de + h * (w + 2 * w2 + 2 * w3 + w4) / 6.0; wn = w + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
    return dn, wn, thf(dn)


def predict_map(c, runs, coef, amap, anchored, N=10, stride=4, facscale=1.0):
    err = [[] for _ in range(N)]; ang = [[] for _ in range(N)]; gmax = []
    for load, lg in runs:
        K = len(lg['t']) - 1
        for k in range(0, K - N, stride):
            fac_k = 1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]); Pd_k = c['Pd'] * fac_k
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy(); gm = 0.0
            for j in range(N):
                u = lg['u'][k + j]
                for _ in range(NSUB):
                    de, w, th = pred_step_map(c, de, w, th, u, TS / NSUB, fac_k * facscale, coef, amap, anchored)
                dt = lg['de'][k + j + 1]; coi_p = np.sum(c['M'] * de) / np.sum(c['M']); coi_t = np.sum(c['M'] * dt) / np.sum(c['M'])
                err[j].append(np.r_[(de - coi_p) - (dt - coi_t), w - lg['w'][k + j + 1]])
                g, _ = alg_res_jac(th, de, Pd_k, *args_of(c)[:3], c['gbus'], c['bg'], False); gm = max(gm, float(np.max(np.abs(g))))
                tt = lg['th'][k + j + 1]
                dth = (th[c['lf']] - th[c['lt']]) - (tt[c['lf']] - tt[c['lt']]); dps = (de - th[c['gbus']]) - (dt - tt[c['gbus']])
                ang[j].append(max(float(np.max(np.abs(dth))), float(np.max(np.abs(dps)))))
            gmax.append(gm)
    return [np.array(e) for e in err], np.array(gmax), [np.array(a) for a in ang]


def summarize(c, res):
    es, gm, an = res
    return dict(rmse_1s=float(np.sqrt(np.mean(np.sum(es[-1] ** 2, axis=1) / c['ng']))), rmse_all=float(np.sqrt(np.mean(np.vstack(es) ** 2))),
                ang_rms_1s=float(np.sqrt(np.mean(an[-1] ** 2))), ang_max_1s=float(np.max(an[-1])), ang_rms_all=float(np.sqrt(np.mean(np.concatenate(an) ** 2))),
                g_mean=float(np.mean(gm)), g_max=float(np.max(gm)))


def b1(name):
    t0 = time.time(); c, th_eq, de_eq, coef = nominal_fit(name)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    maps = {'B1 linear map': AlgMap(False).fit(train, val, c), 'B1 quadratic map': AlgMap(True).fit(train, val, c)}
    for k, m in maps.items(): print(name, k, 'features', m.nfeat, 'alpha', m.alpha, 'val rmse theta (rad)', m.val_rmse, flush=True)
    nrun = 8 if name == 'case9' else 5
    sets = {'in-family (ic 0.10)': make_runs(c, th_eq, de_eq, nrun, 9100, 0.06),
            'wide initial state (ic 0.25)': make_runs(c, th_eq, de_eq, nrun, 9400, 0.06, ic=0.25, wic=0.6)}
    # structural changes KNOWN to the predictor (it is given the new network / load level); learned maps were trained at the nominal network only
    sets_c = {k: (c, v, 1.0) for k, v in sets.items()}
    cm = build_case(name, load_scale=2.0); thm, dem, rm = equilibrium(cm)
    if rm < 1e-8:
        sets_c['loading x2 (mu=2) known to predictor'] = (cm, make_runs(cm, thm, dem, nrun, 9500, 0.06), 2.0)
    nl = len(c['lf']); lines = [int(x) for x in np.linspace(0, nl - 1, 4 if name == 'case9' else 5).astype(int)]; ro = []
    for l in lines:
        cp = build_case(name); cp['lb'] = cp['lb'].copy(); cp['lb'][l] = 0.0
        try: thp, dep, rp = equilibrium(cp)
        except Exception: continue
        if rp < 1e-8 and np.all(np.isfinite(thp)): ro += [(cp, r) for r in make_runs(cp, thp, dep, 2, 9600 + l, 0.06)]
    if ro:
        sets_c['line outage known to predictor'] = ('outage', ro, 1.0)
    out = {}
    for sname, (cc, runs, fs) in sets_c.items():
        res = {}
        if cc == 'outage':
            parts = {}
            def over(fn):
                return [fn(cp, [r]) for cp, r in runs]
            def merge(lst):
                es = [np.vstack([x[0][j] for x in lst]) for j in range(10)]; gm = np.concatenate([x[1] for x in lst]); an = [np.concatenate([x[2][j] for x in lst]) for j in range(10)]
                return es, gm, an
            for nm in ('A0', 'A1', 'A2'):
                res[nm] = summarize(c, merge(over(lambda cp, r, nm=nm: predict_windows(cp, r, nm, coef, stride=4)[:3])))
            for mn, m in maps.items():
                for anch in (False, True):
                    res[f'{mn}{" anchored" if anch else ""}'] = summarize(c, merge(over(lambda cp, r, m=m, anch=anch: predict_map(cp, r, coef, m, anch))))
            out[sname] = res
            for k, v in res.items():
                print(name, sname, k, 'rmse_1s %.4f (ratio to A0 %.3f)  ang_rms_1s %.5f ang_max_1s %.5f  g_mean %.2e g_max %.2e' % (v['rmse_1s'], v['rmse_1s'] / res['A0']['rmse_1s'], v['ang_rms_1s'], v['ang_max_1s'], v['g_mean'], v['g_max']), flush=True)
            continue
        for nm in ('A0', 'A1', 'A2'):
            res[nm] = summarize(cc, predict_windows(cc, runs, nm, coef, stride=4)[:3])
        for mn, m in maps.items():
            for anch in (False, True):
                res[f'{mn}{" anchored" if anch else ""}'] = summarize(cc, predict_map(cc, runs, coef, m, anch, facscale=fs))
        out[sname] = res
        for k, v in res.items():
            print(name, sname, k, 'rmse_1s %.4f (ratio to A0 %.3f)  ang_rms_1s %.5f ang_max_1s %.5f  g_mean %.2e g_max %.2e' % (v['rmse_1s'], v['rmse_1s'] / res['A0']['rmse_1s'], v['ang_rms_1s'], v['ang_max_1s'], v['g_mean'], v['g_max']), flush=True)
    out['map_info'] = {k: dict(nfeat=m.nfeat, alpha=m.alpha, val_rmse_theta=float(m.val_rmse)) for k, m in maps.items()}
    (OUT / f'{name}_B1.json').write_text(json.dumps(out, indent=1, default=float)); print('time', time.time() - t0)


if __name__ == '__main__':
    if sys.argv[1] == 'b1': b1(sys.argv[2])
