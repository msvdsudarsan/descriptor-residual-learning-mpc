"""AC recalibrated-physics comparison (see PROTOCOL.md).  Usage (repo root):
PYTHONPATH=code:extensions/ac_baselines python3 extensions/ac_recal/ac_recal_comparison.py case9|case39 pq|z LAMBDA [NTEST]
Writes extensions/ac_recal/results/<case>_<lm>_lam<LAMBDA>.json (also on failure: status and reason are recorded)."""
import sys, json, time, traceback
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from common import stlsq, bootstrap_ci
from ac_model import *
from ac_pipeline import make_training, make_test_runs, fit_net, TS, NSUB

OUT = Path(__file__).resolve().parent / 'results'; OUT.mkdir(exist_ok=True)
LAMS = (1e-8, 1e-6, 1e-4, 1e-2); BND = 0.5
METHODS = {'A0': (0, 0, 'orig', False), 'A1': (0, 1, 'orig', False), 'P0': (1, 0, 'orig', False), 'P1': (1, 0, 'recal', False),
           'P2': (1, 1, 'orig', False), 'P3': (1, 1, 'recal', True)}      # name: (mode, use_res, physics, uses P3 residual)


def weak_form_pe(c, tr, z_eq):
    """As ac_pipeline.weak_form, plus PE = trapezoid mean of the electrical power b*E*V*sin(psi) per window (needed to refit M, D, Pm)."""
    de, w, U, fac, h = tr['de'], tr['w'], tr['U'], tr['fac'], tr['h']
    net, dynp = net_of(c), dyn_of(c, True); K = U.shape[0]; n = c['n']; gb = c['gbus']; bgE = c['bgE']
    z = np.zeros((K + 1, 2 * n)); guess = z_eq
    for k in range(K + 1):
        guess, _, _ = solve_z(guess, de[k], fac[k], net, 1e-11, 40); z[k] = guess
    Y = np.zeros((K, c['ng'])); TH = np.zeros((K, c['ng'], NF)); PE = np.zeros((K, c['ng']))
    for k in range(K):
        a0 = swing_acc(de[k], w[k], U[k], z[k], net, dynp, False); a1 = swing_acc(de[k + 1], w[k + 1], U[k], z[k + 1], net, dynp, False)
        Y[k] = (w[k + 1] - w[k]) / h - 0.5 * (a0 + a1)
        for i in range(c['ng']):
            j = gb[i]; p0 = de[k, i] - z[k, j]; p1 = de[k + 1, i] - z[k + 1, j]
            TH[k, i] = 0.5 * (feats(p0, w[k, i], U[k, i]) + feats(p1, w[k + 1, i], U[k, i]))
            PE[k, i] = 0.5 * (bgE[i] * z[k, n + j] * np.sin(p0) + bgE[i] * z[k + 1, n + j] * np.sin(p1))
    return TH, Y, PE


def recal_physics(c, TH, Y, PE, THv, Yv, PEv):
    ng = c['ng']; M0, D0, P0 = c['M'].copy(), c['Dphys'].copy(), c['Pm'].copy(); Mn, Dn, Pn = M0.copy(), D0.copy(), P0.copy(); info = []
    for i in range(ng):
        th0 = np.array([M0[i], D0[i], P0[i]])
        def tgt(T, y, pe): return y + (P0[i] + T[:, i, 9] - D0[i] * T[:, i, 6] - pe[:, i]) / M0[i]
        def mod(p, T, pe): return (p[2] + T[:, i, 9] - p[1] * T[:, i, 6] - pe[:, i]) / p[0]
        z, zv = tgt(TH, Y[:, i], PE), tgt(THv, Yv[:, i], PEv); n = len(z)
        sc = np.where(np.abs(th0) > 1e-3, np.abs(th0), 0.5 * np.median(np.abs(P0)))
        lo = np.minimum(th0 - BND * sc, th0 + BND * sc); hi = np.maximum(th0 - BND * sc, th0 + BND * sc); lo[:2] = np.maximum(lo[:2], 1e-6)
        best = None
        for lam in LAMS:
            f = lambda p: np.r_[(z - mod(p, TH, PE)), np.sqrt(lam * n) * (p - th0) / sc]
            s = least_squares(f, th0, bounds=(lo, hi), x_scale=sc, method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12)
            ev = np.sqrt(np.mean((zv - mod(s.x, THv, PEv)) ** 2))
            if best is None or ev < best[0]: best = (ev, lam, s)
        ev, lam, s = best; Mn[i], Dn[i], Pn[i] = s.x
        info.append(dict(gen=i, M0=float(M0[i]), D0=float(D0[i]), P0=float(P0[i]), M=float(s.x[0]), D=float(s.x[1]), Pm=float(s.x[2]), lam=lam, status=int(s.status),
                         at_bound=[bool(abs(s.x[k] - lo[k]) < 1e-9 * sc[k] or abs(s.x[k] - hi[k]) < 1e-9 * sc[k]) for k in range(3)], val_rmse=float(ev)))
    c2 = dict(c); c2['M'], c2['Dphys'], c2['Pm'] = Mn, Dn, Pn
    return c2, info


def predict_run(cp, Mref, run, mode, use_res, coef, N=10, stride=4):
    load, lg = run; net, dynp = net_of(cp), dyn_of(cp, True); n, ng = cp['n'], cp['ng']; gb = cp['gbus']; K = len(lg['t']) - 1
    es = [[] for _ in range(N)]; ang = [[] for _ in range(N)]; ver = [[] for _ in range(N)]; gm_w = []; nwin = 0; t0 = time.perf_counter()
    for k in range(0, K - N, stride):
        fac_k = 1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2])
        de, w, z = lg['de'][k].copy(), lg['w'][k].copy(), lg['z'][k].copy(); gm = 0.0
        for j in range(N):
            u = lg['u'][k + j]
            for _ in range(NSUB): de, w, z = pred_step(de, w, z, u, TS / NSUB, fac_k, mode, use_res, coef, net, dynp)
            zt = lg['z'][k + j + 1]; de_t = lg['de'][k + j + 1]
            coi_p = np.sum(Mref * de) / np.sum(Mref); coi_t = np.sum(Mref * de_t) / np.sum(Mref)
            es[j].append(np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + j + 1]])
            g, _ = ac_res_jac(z, de, fac_k, net, False); gm = max(gm, float(np.max(np.abs(g))))
            dth = (z[cp['lf']] - z[cp['lt']]) - (zt[cp['lf']] - zt[cp['lt']]); dps = (de - z[gb]) - (de_t - zt[gb])
            ang[j].append(max(float(np.max(np.abs(dth))), float(np.max(np.abs(dps))))); ver[j].append(float(np.max(np.abs(z[n:] - zt[n:]))))
        gm_w.append(gm); nwin += 1
    return [np.array(e) for e in es], [np.array(a) for a in ang], [np.array(v) for v in ver], np.array(gm_w), (time.perf_counter() - t0) / max(nwin, 1)


def main(name, lm, lam, ntest):
    tag = f'{name}_{lm}_lam{lam:.1f}' + ('' if ntest == 12 else f'_n{ntest}'); rec = dict(case=name, loadmodel=lm, lam=lam, ntest=ntest, status='OK', reason=''); t00 = time.time()
    try:
        npw = 0.0 if lm == 'pq' else 2.0
        c = build_ac(name, load_scale=lam, npw=npw, nqw=npw); z_eq, de_eq, pfres = nominal_point(c); rec['pf_residual'] = float(pfres)
        if not np.isfinite(pfres) or pfres > 1e-8:
            rec.update(status='FAILED', reason=f'nominal power flow residual {pfres:.2e}'); (OUT / f'{tag}.json').write_text(json.dumps(rec, indent=1)); return
        rec['V_min'] = float(np.min(c['V0'])); rec['V_max'] = float(np.max(c['V0']))
        _, J0, _ = None, None, None
        g0, Jg = ac_res_jac(z_eq, de_eq, 1.0, net_of(c), True); rec['sigma_min_Jg'] = float(np.linalg.svd(Jg, compute_uv=False)[-1])
        t = time.time(); train = make_training(c, z_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, z_eq, de_eq, 8, 1002, 1e-3); rec['t_data_s'] = time.time() - t
        t = time.time(); coef2, v2, yrms = fit_net(c, train, val, z_eq); rec['t_fit_residual_s'] = time.time() - t; rec['residual_target_rms'] = yrms; rec['val_rmse_P2'] = v2
        t = time.time(); Tr = [weak_form_pe(c, tr, z_eq) for tr in train]; Va = [weak_form_pe(c, tr, z_eq) for tr in val]
        TH = np.concatenate([a for a, _, _ in Tr]); Y = np.concatenate([b for _, b, _ in Tr]); PE = np.concatenate([d for _, _, d in Tr])
        THv = np.concatenate([a for a, _, _ in Va]); Yv = np.concatenate([b for _, b, _ in Va]); PEv = np.concatenate([d for _, _, d in Va])
        c2, info = recal_physics(c, TH, Y, PE, THv, Yv, PEv); rec['t_fit_recal_s'] = time.time() - t; rec['recal_params'] = info
        rec['recal_gens_at_bound'] = int(sum(any(d['at_bound']) for d in info))
        t = time.time(); coef3, v3, _ = fit_net(c2, train, val, z_eq); rec['t_fit_P3_s'] = time.time() - t; rec['val_rmse_P3'] = v3
        rec['val_rmse_P1'] = float(np.sqrt(np.mean([d['val_rmse'] ** 2 for d in info]))); rec['val_rmse_A0'] = float(np.sqrt(np.mean(Yv ** 2)))
        rec['active_terms_P2'] = int(np.sum(coef2[:, :NF] != 0)); rec['active_terms_P3'] = int(np.sum(coef3[:, :NF] != 0))
        runs = make_test_runs(c, z_eq, de_eq, ntest, 2001, 0.08)
        coefs = {'A0': coef2, 'A1': coef2, 'P0': coef2, 'P1': coef2, 'P2': coef2, 'P3': coef3}; res = {m: [] for m in METHODS}; Mref = c['M']
        for m, (mode, use_res, phys, _) in METHODS.items():      # warm-up (JIT compilation) so that the timings below exclude it; output discarded
            predict_run(c2 if phys == 'recal' else c, Mref, runs[0], mode, use_res, coefs[m])
        for ri, run in enumerate(runs):
            for m, (mode, use_res, phys, _) in METHODS.items():
                cp = c2 if phys == 'recal' else c
                try:
                    out = predict_run(cp, Mref, run, mode, use_res, coefs[m]); fin = all(np.all(np.isfinite(e)) for e in out[0])
                except Exception as ex:
                    out, fin = None, False; res[m].append(dict(run=ri, finite=False, error=repr(ex))); continue
                es, ang, ver, gm_w, tw = out
                d = dict(run=ri, finite=bool(fin), mse_pooled=float(np.mean(np.vstack(es) ** 2)) if fin else None, mse_1s=float(np.mean(es[-1] ** 2)) if fin else None,
                         sse_h=[float(np.sum(e ** 2)) for e in es], n_h=[int(e.size) for e in es], ang_sq_1s=float(np.mean(ang[-1] ** 2)), V_sq_1s=float(np.mean(ver[-1] ** 2)),
                         gmax_mean=float(np.mean(gm_w)), gmax_max=float(np.max(gm_w)), sec_per_window=float(tw),
                         newton_fail_windows=int(np.sum(~np.isfinite(gm_w)) + (np.sum(gm_w > 1e-6) if mode == 1 else 0)), n_windows=int(len(gm_w)))
                res[m].append(d)
        rec['runs'] = res
        # summaries
        summ = {}
        for m in METHODS:
            ok = [d for d in res[m] if d.get('finite')]
            if ok:
                sse = np.sum([d['sse_h'] for d in ok], axis=0); cnt = np.sum([d['n_h'] for d in ok], axis=0)
                summ[m] = dict(rmse_1s=float(np.sqrt(sse[-1] / cnt[-1])), rmse_h=[float(np.sqrt(a / b)) for a, b in zip(sse, cnt)], ang_rms_1s=float(np.sqrt(np.mean([d['ang_sq_1s'] for d in ok]))),
                               V_rms_1s=float(np.sqrt(np.mean([d['V_sq_1s'] for d in ok]))), gmax_mean=float(np.mean([d['gmax_mean'] for d in ok])), gmax_max=float(max(d['gmax_max'] for d in ok)),
                               sec_per_window=float(np.mean([d['sec_per_window'] for d in ok])), newton_fail_windows=int(sum(d['newton_fail_windows'] for d in ok)), n_finite_runs=len(ok), n_runs=len(res[m]))
            else: summ[m] = dict(n_finite_runs=0, n_runs=len(res[m]))
        common_ok = [ri for ri in range(len(runs)) if all(res[m][ri].get('finite') for m in METHODS)]; rec['n_common_runs'] = len(common_ok); rec['n_excluded_runs'] = len(runs) - len(common_ok)
        for a, b in (('P2', 'P1'), ('P3', 'P1'), ('P3', 'P2'), ('P1', 'P0'), ('P2', 'A0'), ('P1', 'A0'), ('P0', 'A0')):
            if len(common_ok) >= 3:
                for key in ('mse_pooled', 'mse_1s'):
                    r = np.array([np.sqrt(res[a][ri][key] / res[b][ri][key]) for ri in common_ok]); summ.setdefault('paired', {})[f'{a}/{b} {key}'] = dict(zip(('mean', 'lo', 'hi'), bootstrap_ci(r)), per_run=[float(x) for x in r])
        rec['summary'] = summ
    except Exception as ex:
        rec.update(status='FAILED', reason=repr(ex), trace=traceback.format_exc())
    rec['runtime_s'] = time.time() - t00; (OUT / f'{tag}.json').write_text(json.dumps(rec, indent=1, default=float))
    print(tag, rec['status'], rec['reason'], round(rec['runtime_s']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 12)
