"""AC pipeline: plant simulation, weak-form residual identification (same library / STLSQ selection as the repo) and the
A0 / A1 / A2 one-second prediction ablation (paper's N3) on the AC descriptor model.
A0: physics only (damping 0.8 D, no saliency / friction), tangent update of z=(theta,V);  A1: + residual, tangent update;
A2: + residual + Newton projection onto g(z, delta)=0 at every RK stage.  Plant: saliency (x V) + saturating friction + load fluctuation
(P and Q scaled together).  Usage: python ac_pipeline.py case9|case39 [loadmodel pq|z]"""
import sys, json, time
from pathlib import Path
import numpy as np
from common import stlsq, conformal_quantile, bootstrap_ci
from ac_model import *

RES = Path(__file__).resolve().parents[1] / 'results'
TS, NSUB = 0.1, 2
MODES = {'A0': (0, 0), 'A1': (0, 1), 'A2': (1, 1)}


def simulate_plant(c, z0, de0, w0, controller, T, load, hp=0.0125, Ts=TS):
    lamp, lom, lph = load; ng = c['ng']; nsteps = int(round(T / Ts)); nsub = int(round(Ts / hp))
    net, dyn = net_of(c), dyn_of(c)
    de, w, z = de0.copy(), w0.copy(), z0.copy(); u_prev = np.zeros(ng); t = 0.0
    L = dict(t=[], de=[], w=[], z=[], u=[])
    for k in range(nsteps):
        u = controller(t, de, w, z, u_prev)
        for key, val in (('t', t), ('de', de.copy()), ('w', w.copy()), ('z', z.copy()), ('u', u.copy())):
            L[key].append(val)
        for _ in range(nsub):
            de, w, z = plant_step(de, w, z, u, hp, t, net, dyn, lamp, lom, lph); t += hp
        u_prev = u
    for key, val in (('t', t), ('de', de.copy()), ('w', w.copy()), ('z', z.copy()), ('u', u_prev.copy())):
        L[key].append(val)
    return {k: np.array(v) for k, v in L.items()}


def static_ctrl(c):
    return lambda t, de, w, z, up: np.clip(-c['D'] * w, -c['umax'], c['umax'])


def make_training(c, z_eq, de_eq, n_traj, seed, noise, h=0.025, T=3.0, ic=0.20, wic=0.8, lamp=0.04):
    rng = np.random.default_rng(seed); out = []; net = net_of(c)
    for _ in range(n_traj):
        de0 = de_eq + rng.normal(0, ic, c['ng']); w0 = rng.normal(0, wic, c['ng'])
        z0, _, _ = solve_z(z_eq, de0, 1.0, net, 1e-12, 40)
        load = (rng.uniform(0, lamp), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        ctrl = lambda t, de, w, z, up: np.clip(-c['D'] * w + rng.normal(0, 0.35, c['ng']) * c['umax'], -c['umax'], c['umax'])
        lg = simulate_plant(c, z0, de0, w0, ctrl, T, load, hp=h / 2, Ts=h)
        de_m = lg['de'] + rng.normal(0, noise, lg['de'].shape); w_m = lg['w'] + rng.normal(0, noise, lg['w'].shape)
        fac = 1 + load[0] * np.sin(load[1] * lg['t'] + load[2])
        out.append(dict(de=de_m, w=w_m, U=lg['u'][:-1], fac=fac, h=h, de_true=lg['de'], w_true=lg['w'], z_true=lg['z']))
    return out


def weak_form(c, tr, z_eq):
    de, w, U, fac, h = tr['de'], tr['w'], tr['U'], tr['fac'], tr['h']
    net, dynp = net_of(c), dyn_of(c, True); K = U.shape[0]; n = c['n']
    z = np.zeros((K + 1, 2 * n)); guess = z_eq
    for k in range(K + 1):
        guess, _, _ = solve_z(guess, de[k], fac[k], net, 1e-11, 40); z[k] = guess
    Y = np.zeros((K, c['ng'])); TH = np.zeros((K, c['ng'], NF)); gb = c['gbus']
    for k in range(K):
        a0 = swing_acc(de[k], w[k], U[k], z[k], net, dynp, False); a1 = swing_acc(de[k + 1], w[k + 1], U[k], z[k + 1], net, dynp, False)
        Y[k] = (w[k + 1] - w[k]) / h - 0.5 * (a0 + a1)
        for i in range(c['ng']):
            TH[k, i] = 0.5 * (feats(de[k, i] - z[k, gb[i]], w[k, i], U[k, i]) + feats(de[k + 1, i] - z[k + 1, gb[i]], w[k + 1, i], U[k, i]))
    return TH, Y


def fit_net(c, train, val, z_eq, ridges=(1e-6, 1e-4, 1e-2), threshes=(0.002, 0.01, 0.03, 0.1, 0.3)):
    Tr = [weak_form(c, t, z_eq) for t in train]; Va = [weak_form(c, t, z_eq) for t in val]
    TH = np.concatenate([a for a, _ in Tr]); Y = np.concatenate([b for _, b in Tr])
    THv = np.concatenate([a for a, _ in Va]); Yv = np.concatenate([b for _, b in Va])
    coef = np.zeros((c['ng'], NF + 1)); info = []
    for i in range(c['ng']):
        cands = []
        for rg in ridges:
            for tq in threshes:
                cc = stlsq(TH[:, i, :], Y[:, i], ridge=rg, thresh=tq)
                cands.append((np.sqrt(np.mean((Yv[:, i] - THv[:, i, :] @ cc) ** 2)), int(np.sum(cc != 0)), cc))
        ebest = min(x[0] for x in cands)
        e, _, cc = min([x for x in cands if x[0] <= 1.05 * ebest], key=lambda x: (x[1], x[0]))
        coef[i, :NF] = cc; coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i])); info.append(e)
    # residual rms of the true (not noisy) hidden term on the training windows, for the report
    return coef, float(np.mean(info)), float(np.sqrt(np.mean(Y ** 2)))


def make_test_runs(c, z_eq, de_eq, n, seed, lamp_hi, T=6.0, lamp_lo=0.0):
    rng = np.random.default_rng(seed); runs = []; net = net_of(c)
    for _ in range(n):
        de0 = de_eq + rng.normal(0, 0.10, c['ng']); w0 = rng.normal(0, 0.3, c['ng'])
        z0, _, _ = solve_z(z_eq, de0, 1.0, net, 1e-12, 40)
        load = (rng.uniform(lamp_lo, lamp_hi), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        runs.append((load, simulate_plant(c, z0, de0, w0, static_ctrl(c), T, load)))
    return runs


def predict_windows(c, runs, name, coef, N=10, stride=4):
    mode, use_res = MODES[name]; net, dynp = net_of(c), dyn_of(c, True); n, ng = c['n'], c['ng']; gb = c['gbus']
    err_state = [[] for _ in range(N)]; gmax = []; angerr = [[] for _ in range(N)]; verr = [[] for _ in range(N)]
    for load, lg in runs:
        K = len(lg['t']) - 1
        for k in range(0, K - N, stride):
            fac_k = 1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2])
            de, w, z = lg['de'][k].copy(), lg['w'][k].copy(), lg['z'][k].copy(); gm = 0.0
            for j in range(N):
                u = lg['u'][k + j]
                for _ in range(NSUB):
                    de, w, z = pred_step(de, w, z, u, TS / NSUB, fac_k, mode, use_res, coef, net, dynp)
                zt = lg['z'][k + j + 1]; de_t = lg['de'][k + j + 1]
                coi_p = np.sum(c['M'] * de) / np.sum(c['M']); coi_t = np.sum(c['M'] * de_t) / np.sum(c['M'])
                err_state[j].append(np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + j + 1]])
                g, _ = ac_res_jac(z, de, fac_k, net, False); gm = max(gm, float(np.max(np.abs(g))))
                dth = (z[c['lf']] - z[c['lt']]) - (zt[c['lf']] - zt[c['lt']]); dps = (de - z[gb]) - (de_t - zt[gb])
                angerr[j].append(max(float(np.max(np.abs(dth))), float(np.max(np.abs(dps)))))
                verr[j].append(float(np.max(np.abs(z[n:] - zt[n:]))))
            gmax.append(gm)
    return [np.array(e) for e in err_state], np.array(gmax), [np.array(a) for a in angerr], [np.array(v) for v in verr]


def run(name, tag, npw, nqw):
    t0 = time.time(); c = build_ac(name, npw=npw, nqw=nqw); z_eq, de_eq, pf = nominal_point(c)
    train = make_training(c, z_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, z_eq, de_eq, 8, 1002, 1e-3)
    coef, val_rmse, y_rms = fit_net(c, train, val, z_eq)
    print(name, tag, 'fit done; residual-target rms', y_rms, 'val rmse after fit', val_rmse, 'active terms', int(np.sum(coef[:, :NF] != 0)), '/', coef[:, :NF].size, 'time', time.time() - t0, flush=True)
    nrun = 12 if name == 'case9' else 8
    runs = make_test_runs(c, z_eq, de_eq, nrun, 2001, 0.08)             # as the paper's N3: load fluctuation up to 0.08 (2x training max)
    out = {}
    for nm in ('A0', 'A1', 'A2'):
        out[nm] = predict_windows(c, runs, nm, coef)
    rows = {}
    for nm in out:
        es = out[nm][0]
        rows[nm] = dict(rmse_h=[float(np.sqrt(np.mean(np.sum(e ** 2, axis=1) / c['ng']))) for e in es],
                        ang_rms_h=[float(np.sqrt(np.mean(a ** 2))) for a in out[nm][2]], ang_max_h=[float(np.max(a)) for a in out[nm][2]],
                        V_rms_h=[float(np.sqrt(np.mean(v ** 2))) for v in out[nm][3]], gmax_mean=float(np.mean(out[nm][1])), gmax_max=float(np.max(out[nm][1])))
    # paired per-run ratio A2/A0 and A1/A0 (all horizons pooled, as in the paper's paired table)
    ratios = {'A2': [], 'A1': []}
    for r in runs:
        e0 = np.mean(np.vstack(predict_windows(c, [r], 'A0', coef)[0]) ** 2)
        for nm in ('A1', 'A2'):
            ratios[nm].append(float(np.sqrt(np.mean(np.vstack(predict_windows(c, [r], nm, coef)[0]) ** 2) / e0)))
    summ = dict(case=name, tag=tag, npw=npw, nqw=nqw, nruns=nrun, rows=rows, paired_ratio_ci={k: bootstrap_ci(np.array(v)) for k, v in ratios.items()},
                paired_ratio_runs=ratios, ratio_1s_A1_over_A0=rows['A1']['rmse_h'][-1] / rows['A0']['rmse_h'][-1], ratio_1s_A2_over_A0=rows['A2']['rmse_h'][-1] / rows['A0']['rmse_h'][-1],
                active_terms=int(np.sum(coef[:, :NF] != 0)), residual_target_rms=y_rms, val_rmse_after_fit=val_rmse, runtime_s=time.time() - t0)
    print(json.dumps({k: summ[k] for k in ('case', 'tag', 'ratio_1s_A1_over_A0', 'ratio_1s_A2_over_A0', 'paired_ratio_ci')}, default=float), flush=True)
    for nm in rows:
        print(nm, '1s state rmse', rows[nm]['rmse_h'][-1], 'ang rms 1s', rows[nm]['ang_rms_h'][-1], 'V rms 1s', rows[nm]['V_rms_h'][-1], 'gmax mean/max', rows[nm]['gmax_mean'], rows[nm]['gmax_max'], flush=True)
    (RES / f'{name}_ac_{tag}_prediction.json').write_text(json.dumps(summ, indent=1, default=float))
    np.save(RES / f'coef_ac_{name}_{tag}.npy', coef)


if __name__ == '__main__':
    name = sys.argv[1]; lm = sys.argv[2] if len(sys.argv) > 2 else 'pq'
    run(name, lm, *((0.0, 0.0) if lm == 'pq' else (2.0, 2.0)))
