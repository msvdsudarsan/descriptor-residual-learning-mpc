"""B2: coverage of the TRUE algebraic-angle error (max over all line angles and rotor-terminal angles, the repo's definition) by
margins built on A2 predictions (lossless repo model), in-family and out-of-family.
Margins (horizon step j = 1..10, window start k >= 32 so that 3.2 s of load history exist):
  CONST   : 95% split-conformal quantile of the algebraic-angle error at step j, calibrated in-family   (constant margin)
  SPAN    : 95% conformal quantile of span(rotor-angle prediction error) at step j  (state-error envelope only, Lemma 1 without load term)
  LEM_D   : SPAN + s_load(k) * R_design                     (Lemma 1 + frozen-load term; s_load from the INVERSE JACOBIAN at the window state,
                                                            R_design = declared peak-to-peak load swing of the in-family generator = 0.12)
  LEM_O   : SPAN + s_load(k) * R_k                          (same, R_k = observed peak-to-peak of the metered load factor over the last 3.2 s)
  LEM_OC  : SPAN + s_load(eq) * R_k                         (as LEM_O but Jacobian frozen at the calibration equilibrium = isolates the Jacobian-aware part)
  LEMCAL_J: SPAN + kappa_j * s_load(k) * R_k, kappa_j = 95% conformal quantile on the calibration set (same average size as CONST in-family)
  LEMCAL_C: same with the Jacobian frozen at the calibration equilibrium
  ORACLE  : SPAN + exact frozen-load change (uses the true future load; diagnostic only)
s_load(k) = max( max_lines |a_l^T J^-1 Pd|, max_i |(J^-1 Pd)_{gbus(i)}| ), J = -dg/dtheta at the plant state at k.
Usage: python b2_margins.py case9|case39"""
import sys, json, time
import numpy as np
from network_baselines_loaddyn import scenario_runs
from network_validity_gate import *

OUT = RES
KMIN, HIST = 32, 32
R_DESIGN = 0.12


def s_load(c, th, de, Pd):
    _, J = alg_res_jac(th, de, Pd, *args_of(c)[:3], c['gbus'], c['bg'], True)
    v = np.linalg.solve(J, Pd)
    return float(max(np.max(np.abs(v[c['lf']] - v[c['lt']])), np.max(np.abs(v[c['gbus']]))))


def windows(c, runs, coef, N=10, stride=4):
    """Per window: err_ang[N], span_e[N], oracle_frozen[N], s_k, R_k.  Predictor: A2."""
    rows = []
    for load, lg in runs:
        K = len(lg['t']) - 1; fac = 1 + load[0] * np.sin(load[1] * lg['t'] + load[2])
        for k in range(KMIN, K - N, stride):
            Rk = float(fac[k - HIST:k + 1].max() - fac[k - HIST:k + 1].min()); Pd_k = c['Pd'] * fac[k]
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
            ea = np.zeros(N); sp = np.zeros(N); orc = np.zeros(N)
            for j in range(N):
                u = lg['u'][k + j]
                for _ in range(NSUB):
                    de, w, th = pred_step(de, w, th, u, TS / NSUB, Pd_k, 1, 1, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
                dt = lg['de'][k + j + 1]; tt = lg['th'][k + j + 1]
                dth = (th[c['lf']] - th[c['lt']]) - (tt[c['lf']] - tt[c['lt']]); dps = (de - th[c['gbus']]) - (dt - tt[c['gbus']])
                ea[j] = max(float(np.max(np.abs(dth))), float(np.max(np.abs(dps))))
                e = de - dt; sp[j] = float(e.max() - e.min())
                # exact frozen-load term: theta(delta_true, Pd_k) vs plant theta (Pd_true)
                tf, _ = solve_theta(tt, dt, Pd_k, *args_of(c), 1e-12, 30)
                a = np.abs((tf[c['lf']] - tf[c['lt']]) - (tt[c['lf']] - tt[c['lt']])); b = np.abs((dt - tf[c['gbus']]) - (dt - tt[c['gbus']]))
                orc[j] = max(float(a.max()), float(b.max()))
            rows.append((ea, sp, orc, s_load(c, lg['th'][k], lg['de'][k], Pd_k), Rk))
    return dict(ea=np.array([r[0] for r in rows]), sp=np.array([r[1] for r in rows]), orc=np.array([r[2] for r in rows]),
                s=np.array([r[3] for r in rows]), R=np.array([r[4] for r in rows]))


def margins(cal, s_eq, W):
    q = lambda X: np.array([conformal_quantile(X[:, j], 0.05) for j in range(X.shape[1])])
    qc, qs = q(cal['ea']), q(cal['sp'])
    ok = cal['R'] > 1e-9; den = cal['s'] * cal['R']; den_c = s_eq * cal['R']
    kap = np.array([conformal_quantile(np.maximum(cal['ea'][ok, j] - qs[j], 0) / den[ok], 0.05) for j in range(cal['ea'].shape[1])])
    kap_c = np.array([conformal_quantile(np.maximum(cal['ea'][ok, j] - qs[j], 0) / den_c[ok], 0.05) for j in range(cal['ea'].shape[1])])
    return dict(qc=qc, qs=qs, s_eq=s_eq, kap=kap, kap_c=kap_c)


def coverage(M, W):
    n = W['ea'].shape[0]
    mg = {'CONST': np.tile(M['qc'], (n, 1)), 'SPAN': np.tile(M['qs'], (n, 1)),
          'LEM_D': M['qs'][None, :] + (W['s'] * R_DESIGN)[:, None], 'LEM_O': M['qs'][None, :] + (W['s'] * W['R'])[:, None],
          'LEM_OC': M['qs'][None, :] + (M['s_eq'] * W['R'])[:, None],
          'LEMCAL_J': M['qs'][None, :] + M['kap'][None, :] * (W['s'] * W['R'])[:, None], 'LEMCAL_C': M['qs'][None, :] + M['kap_c'][None, :] * (M['s_eq'] * W['R'])[:, None], 'ORACLE': M['qs'][None, :] + W['orc']}
    out = {}
    for k, m in mg.items():
        cov = W['ea'] <= m
        out[k] = dict(cov_j5=float(cov[:, 4].mean()), cov_j10=float(cov[:, 9].mean()), cov_all=float(cov.mean()), margin_over_err_j10=float(m[:, 9].mean() / W['ea'][:, 9].mean()), margin_mean_j10=float(m[:, 9].mean()))
    out['_n_windows'] = n; out['_err_mean_j10'] = float(W['ea'][:, 9].mean()); out['_err_q95_j10'] = float(np.quantile(W['ea'][:, 9], 0.95))
    # additional diagnostic: does the exact decomposition err <= span + frozen hold (Lemma 1 + exact load term)?
    out['_lemma_oracle_holds_frac_all'] = float(np.mean(W['ea'] <= W['sp'] + W['orc'] + 1e-9))
    return out


def concat(Ws):
    return {k: np.concatenate([w[k] for w in Ws]) for k in Ws[0]}


def run(name):
    t0 = time.time(); c, th_eq, de_eq, coef = nominal_fit(name); nr = 20 if name == 'case9' else 24
    cal = windows(c, make_runs_b(c, th_eq, de_eq, nr, 7001, 0.06), coef)
    _, J = alg_res_jac(th_eq, de_eq, c['Pd'], *args_of(c)[:3], c['gbus'], c['bg'], True)
    s_eq = s_load(c, th_eq, de_eq, c['Pd']); M = margins(cal, s_eq, None)
    print(name, 'calibration windows', cal['ea'].shape[0], 'qc', np.round(M['qc'], 4), 'qs', np.round(M['qs'], 4), 's_eq', s_eq, 'mean s_k', cal['s'].mean(), 'range s_k', cal['s'].min(), cal['s'].max(), flush=True)
    res = {'_calibration': dict(n=int(cal['ea'].shape[0]), qc=M['qc'], qs=M['qs'], s_eq=s_eq, s_k_min=float(cal['s'].min()), s_k_max=float(cal['s'].max()))}
    scen = {}
    scen['in-family (fresh seed)'] = windows(c, make_runs_b(c, th_eq, de_eq, nr, 7002, 0.06), coef)
    scen['load fluctuation x2 (amp<=0.12)'] = windows(c, make_runs_b(c, th_eq, de_eq, nr, 7003, 0.12), coef)
    # saliency x1.5 in the plant
    rs = scenario_runs(c, th_eq, de_eq, 'saliency x1.5', nr, 7004, T=8.0)
    scen['saliency x1.5'] = windows(c, rs, coef)
    # line outage in the plant, nominal predictor
    nl = len(c['lf']); lines = list(range(nl)) if name == 'case9' else list(np.linspace(0, nl - 1, 10).astype(int)); Wl = []; used = []
    for l in lines:
        cp = build_case(name); cp['lb'] = cp['lb'].copy(); cp['lb'][l] = 0.0
        try:
            thp, dep, rp = equilibrium(cp)
        except Exception:
            continue
        if not (rp < 1e-8) or not np.all(np.isfinite(thp)): continue
        runs = make_runs_b(cp, thp, dep, 3, 7100 + l, 0.06)
        # predictor keeps the NOMINAL network c; the sensitivity s_load and the plant states use the nominal predictor model
        Wl.append(windows(c, runs, coef)); used.append(l)
    scen['line outage (plant), nominal predictor'] = concat(Wl)
    # loading x2: plant and predictor at load scale 2 (dispatch scaled), coef/margins from mu=1
    cm = build_case(name, load_scale=2.0); thm, dem, rm = equilibrium(cm)
    if rm < 1e-8:
        scen['loading x2 (mu=2)'] = windows(cm, make_runs_b(cm, thm, dem, nr, 7005, 0.06), coef)
    for sname, W in scen.items():
        res[sname] = coverage(M, W); r = res[sname]
        print(name, sname, 'n', r['_n_windows'], 'err_mean_j10 %.4f' % r['_err_mean_j10'], 'lemma-oracle-decomp holds %.3f' % r['_lemma_oracle_holds_frac_all'])
        for k in ('CONST', 'SPAN', 'LEM_D', 'LEM_O', 'LEM_OC', 'LEMCAL_J', 'LEMCAL_C', 'ORACLE'):
            print('    %-7s cov j5 %.3f  j10 %.3f  all %.3f   margin/err(j10) %.2f' % (k, r[k]['cov_j5'], r[k]['cov_j10'], r[k]['cov_all'], r[k]['margin_over_err_j10']), flush=True)
    res['_outage_lines_used'] = used
    (OUT / f'{name}_B2.json').write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))); print('time', time.time() - t0)


def make_runs_b(c, th_eq, de_eq, n, seed, lamp_hi, T=8.0):
    from b_baselines import make_runs
    return make_runs(c, th_eq, de_eq, n, seed, lamp_hi, T=T)


if __name__ == '__main__':
    run(sys.argv[1])
