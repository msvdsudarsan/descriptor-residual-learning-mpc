"""Recalibrated-physics baseline (protocol BASELINE_PROTOCOL). NEW script; no existing code is edited.
NOTE (v1): P1c is evaluated inside the SAME Newton-projected predictor/A2 controller as P2 with a zero residual (a first version used the unprojected 'A0' mode for P1c; superseded, kept in results/baseline_recal_superseded_P1c_unprojected).
Models: A0 original physics | P2 library residual (stored coef) | P1a/P1b damping recalibration (restricted()) | P1c bounded physical recalibration of (M_i, D_i, Pm_i)
        | P3 = P1c + library residual fitted on P1c's remaining weak-form defect (fit_net, unchanged).
Prediction metric: same as network_param_baseline.py (stride 4, N=10), but the centre-of-inertia weighting always uses the ORIGINAL M.
Usage: python3 network_recal_baseline.py case9|case39 [pred|cl|both]"""
import sys, json, time, traceback
import numpy as np
from scipy.optimize import least_squares
from network_param_baseline import VARIANTS, restricted
from network_cl_recalibration import setup, metrics
from network_baselines_loaddyn import stack, scenario_runs
from network_validity_gate import *

OUT = RES / 'baseline_recal'; OUT.mkdir(exist_ok=True)
LAMS = (1e-8, 1e-6, 1e-4, 1e-2); BND = 0.5        # +-50 % of nominal (protocol)


def recal_physics(c, TH, Y, THv, Yv):
    """Per generator, bounded least squares on theta=(M, D, Pm); weak-form target z = Y + a_nominal (exact, from the stored features)."""
    ng = c['ng']; M0, D0, P0 = c['M'].copy(), c['Dphys'].copy(), c['Pm'].copy()
    Mn, Dn, Pn = M0.copy(), D0.copy(), P0.copy(); info = []; sq = 0.0; cnt = 0
    for i in range(ng):
        th0 = np.array([M0[i], D0[i], P0[i]]); bg = c['bg'][i]
        def tgt(T, y): return y + (P0[i] + T[:, i, 9] - D0[i] * T[:, i, 6] - bg * T[:, i, 0]) / M0[i]
        def mod(p, T): return (p[2] + T[:, i, 9] - p[1] * T[:, i, 6] - bg * T[:, i, 0]) / p[0]
        z, zv = tgt(TH, Y[:, i]), tgt(THv, Yv[:, i]); n = len(z)
        sc = np.where(np.abs(th0) > 1e-3, np.abs(th0), 0.5 * np.median(np.abs(P0)))      # zero-nominal rule (not triggered for case9/case39)
        lo = np.minimum(th0 - BND * sc, th0 + BND * sc); hi = np.maximum(th0 - BND * sc, th0 + BND * sc)
        lo[:2] = np.maximum(lo[:2], 1e-6)                                                  # M>0, D>0 (bounds already positive)
        best = None
        for lam in LAMS:
            f = lambda p: np.r_[(z - mod(p, TH)), np.sqrt(lam * n) * (p - th0) / sc]
            s = least_squares(f, th0, bounds=(lo, hi), x_scale=sc, method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12)
            ev = np.sqrt(np.mean((zv - mod(s.x, THv)) ** 2))
            if best is None or ev < best[0]: best = (ev, lam, s)
        ev, lam, s = best; Mn[i], Dn[i], Pn[i] = s.x
        sq += float(np.sum((zv - mod(s.x, THv)) ** 2)); cnt += len(zv)
        info.append(dict(gen=i, M0=M0[i], D0=D0[i], P0=P0[i], M=s.x[0], D=s.x[1], Pm=s.x[2], lam=lam, status=int(s.status), nfev=int(s.nfev),
                         at_bound=[bool(abs(s.x[k] - lo[k]) < 1e-9 * sc[k] or abs(s.x[k] - hi[k]) < 1e-9 * sc[k]) for k in range(3)], val_rmse=float(ev)))
    c2 = dict(c); c2['M'], c2['Dphys'], c2['Pm'] = Mn, Dn, Pn
    return c2, info, float(np.sqrt(sq / cnt))


def val_rmse_linear(THv, Yv, coef, ng):
    return float(np.sqrt(np.mean(np.concatenate([(Yv[:, i] - THv[:, i, :] @ coef[i, :NF]) ** 2 for i in range(ng)]))))


def pred_run(cp, Mref, run, name, coef, N=10, stride=4):
    """Squared error sum and count for one run (predictor built from cp; COI weighting with Mref)."""
    mode, use_res = MODES[name]; load, lg = run; K = len(lg['t']) - 1; ss = 0.0; n = 0
    for k in range(0, K - N, stride):
        Pd_k = cp['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
        de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
        for j in range(N):
            u = lg['u'][k + j]
            for _ in range(NSUB): de, w, th = pred_step(de, w, th, u, TS / NSUB, Pd_k, mode, use_res, coef, *args_of(cp), cp['M'], cp['Dphys'], cp['Pm'], cp['kappa'], cp['fric'])
            coi_p = np.sum(Mref * de) / np.sum(Mref); de_t = lg['de'][k + j + 1]; coi_t = np.sum(Mref * de_t) / np.sum(Mref)
            e = np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + j + 1]]
            ss += float(np.sum(e ** 2)); n += len(e)
    return ss, n


def csvw(path, head, rows):
    with open(path, 'w') as f:
        f.write(','.join(head) + '\n')
        for r in rows: f.write(','.join(str(x) for x in r) + '\n')


def prediction(name):
    c, th_eq, de_eq, coef_lib = nominal_fit(name)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    TH, Y = stack([weak_form_net(c, t, th_eq) for t in train]); THv, Yv = stack([weak_form_net(c, t, th_eq) for t in val])
    cf = {k: (coef_lib if v is None else restricted(TH, Y, THv, Yv, c['ng'], v)) for k, v in VARIANTS.items()}
    c1c, info, v1c = recal_physics(c, TH, Y, THv, Yv)
    csvw(OUT / f'{name}_prediction_P1c_params.csv', list(info[0].keys()), [list(d.values()) for d in info])
    vsel = {'A0 physics (reference)': float(np.sqrt(np.mean(Yv ** 2))), 'P1a damping + constant': val_rmse_linear(THv, Yv, cf['damping + constant'], c['ng']),
            'P1b damping + cubic speed + constant': val_rmse_linear(THv, Yv, cf['damping + cubic speed + constant'], c['ng']), 'P1c physical recalibration': v1c}
    pick = min(['P1a damping + constant', 'P1b damping + cubic speed + constant', 'P1c physical recalibration'], key=lambda k: vsel[k])
    # P3: library residual on top of P1c (fit_net unchanged)
    coef3 = fit_net(c1c, train, val, th_eq)[0]
    json.dump(dict(validation_weak_form_rmse=vsel, selected_P1=pick, P1c_status=[d['status'] for d in info], P1c_at_bound=[d['at_bound'] for d in info]), open(OUT / f'{name}_prediction_selection.json', 'w'), indent=1)
    print(name, 'validation weak-form RMSE', vsel, '| selected', pick, flush=True)
    models = [('A0', c, 'A0', coef_lib), ('P2 library residual', c, 'A2', coef_lib), ('P1a', c, 'A2', cf['damping + constant']), ('P1b', c, 'A2', cf['damping + cubic speed + constant']),
              ('P1c', c1c, 'A2', np.zeros((c['ng'], NF + 1))), ('P3', c1c, 'A2', coef3)]
    nrun = 8 if name == 'case9' else 5; rows = []; fails = []
    for scen, runs in (('in-family', make_test_runs(c, th_eq, de_eq, nrun, 9100, 0.06, T=8.0)), ('saliency x1.5', scenario_runs(c, th_eq, de_eq, 'saliency x1.5', nrun, 9200)), ('load lag 1.0 s', scenario_runs(c, th_eq, de_eq, 'load-lag tau=1.0', nrun, 9300))):
        for ri, run in enumerate(runs):
            for mname, cp, mode, co in models:
                try: ss, n = pred_run(cp, c['M'], run, mode, co)
                except Exception as ex: ss, n = float('nan'), 0; fails.append((scen, ri, mname, repr(ex)))
                rows.append((scen, ri, mname, ss, n))
            print(name, scen, 'run', ri, flush=True)
    csvw(OUT / f'{name}_prediction_runs.csv', ['scenario', 'run', 'model', 'sumsq', 'n'], rows)
    json.dump(fails, open(OUT / f'{name}_prediction_failures.json', 'w'))
    # stored-value check (pooled ratios of P2/P1a/P1b vs stored CSV)
    import csv as _c
    st = {(r['scenario'], r['correction']): float(r['rmse_ratio_to_A0']) for r in _c.DictReader(open(RES / f'{name}_param_baseline_prediction.csv'))}
    chk = {}
    for scen in ('in-family', 'saliency x1.5', 'load lag 1.0 s'):
        R = [r for r in rows if r[0] == scen]; tot = lambda m: (sum(r[3] for r in R if r[2] == m), sum(r[4] for r in R if r[2] == m))
        a0 = tot('A0'); rat = lambda m: np.sqrt((tot(m)[0] / tot(m)[1]) / (a0[0] / a0[1]))
        for m, key in (('P2 library residual', 'library residual (paper)'), ('P1a', 'damping + constant'), ('P1b', 'damping + cubic speed + constant')): chk[f'{scen} | {m}'] = (float(rat(m)), st[(scen, key)], abs(float(rat(m)) - st[(scen, key)]))
    json.dump(chk, open(OUT / f'{name}_prediction_stored_check.json', 'w'), indent=1); print(name, 'stored-value check max abs diff', max(v[2] for v in chk.values()), flush=True)


def energy(lg): return float(TS * np.sum(lg['u'][:-1] ** 2))


def closed_loop(name):
    cs, th_s, de_s, coef_s, rel_ref = setup(name); T = 6.0 if name == 'case9' else 5.0; nreal = 6 if name == 'case9' else 4
    trs = make_training(cs, th_s, de_s, 30, 1001, 1e-3); vls = make_training(cs, th_s, de_s, 8, 1002, 1e-3)
    THs, Ys = stack([weak_form_net(cs, t, th_s) for t in trs]); THsv, Ysv = stack([weak_form_net(cs, t, th_s) for t in vls])
    cfs = {k: (coef_s if v is None else restricted(THs, Ys, THsv, Ysv, cs['ng'], v)) for k, v in VARIANTS.items()}
    c1c, info, v1c = recal_physics(cs, THs, Ys, THsv, Ysv)
    csvw(OUT / f'{name}_closed_loop_P1c_params.csv', list(info[0].keys()), [list(d.values()) for d in info])
    coef3 = fit_net(c1c, trs, vls, th_s)[0]
    json.dump(dict(validation_weak_form_rmse_P1c=v1c, A0=float(np.sqrt(np.mean(Ysv ** 2))), P1a=val_rmse_linear(THsv, Ysv, cfs['damping + constant'], cs['ng']), P1b=val_rmse_linear(THsv, Ysv, cfs['damping + cubic speed + constant'], cs['ng']),
                   P1c_status=[d['status'] for d in info], P1c_at_bound=[d['at_bound'] for d in info]), open(OUT / f'{name}_closed_loop_selection.json', 'w'), indent=1)
    gen = lambda rng: ((rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)), rng.normal(0, 0.03, cs['ng']), rng.normal(0, 0.1, cs['ng']))
    def ctrl_of(k, mult=None):
        if k == 'C0 static law': return static_ctrl(cs)
        if k == 'C0 energy-matched': kg = mult * cs['D']; return lambda t, de, w, th, up, Pd: np.clip(-kg * w, -cs['umax'], cs['umax'])
        if k == 'A0 physics only': return make_net_mpc(cs, 'A0', coef_s, rel_ref, np.zeros(6), analytic=False)
        if k == 'P1c physical recalibration': return make_net_mpc(c1c, 'A2', np.zeros((cs['ng'], NF + 1)), rel_ref, np.zeros(6), analytic=True)      # same Newton-projected predictor/controller structure as P2, zero residual
        if k == 'P3 residual on P1c': return make_net_mpc(c1c, 'A2', coef3, rel_ref, np.zeros(6), analytic=True)
        return make_net_mpc(cs, 'A2', cfs[k], rel_ref, np.zeros(6), analytic=True)
    # pilot (seed 80000, 10 trajectories, disjoint from the test realizations 6001): static gain matched to P2's input energy
    rng = np.random.default_rng(80000); pil = []
    for _ in range(10):
        load, d0, w0 = gen(rng); de0 = de_s + d0; th0, _ = solve_theta(th_s, de0, cs['Pd'], *args_of(cs), 1e-12, 30); pil.append((load, de0, w0, th0))
    run_p = lambda ctrl_fn: np.mean([energy(simulate_plant(cs, th0, de0, w0, ctrl_fn(), T, load)) for load, de0, w0, th0 in pil])
    e_target = run_p(lambda: ctrl_of('library residual (paper)')); print(name, 'pilot P2 energy', e_target, flush=True)
    lo_, hi_ = np.log(0.05), np.log(20.0); g_hist = []
    for _ in range(14):
        mid = 0.5 * (lo_ + hi_); e = run_p(lambda: ctrl_of('C0 energy-matched', np.exp(mid))); g_hist.append((float(np.exp(mid)), float(e)))
        if e < e_target: lo_ = mid
        else: hi_ = mid
    gstar = float(np.exp(0.5 * (lo_ + hi_))); json.dump(dict(pilot_seed=80000, n_pilot=10, e_target=e_target, gain_multiple_of_D=gstar, history=g_hist), open(OUT / f'{name}_c0_energy_gain.json', 'w'), indent=1)
    print(name, 'C0 energy-matched gain multiple', gstar, flush=True)
    ctrls = ['C0 static law', 'A0 physics only', 'library residual (paper)', 'damping + constant', 'damping + cubic speed + constant', 'P1c physical recalibration', 'P3 residual on P1c', 'C0 energy-matched']
    rng = np.random.default_rng(6001); rows = []; fails = []
    for i in range(nreal):
        load, d0, w0 = gen(rng); de0 = de_s + d0; th0, _ = solve_theta(th_s, de0, cs['Pd'], *args_of(cs), 1e-12, 30)
        for k in ctrls:
            try:
                lg = simulate_plant(cs, th0, de0, w0, ctrl_of(k, gstar), T, load); m = metrics(cs, lg, GAMMA)
                if not np.all(np.isfinite(lg['w'])): raise FloatingPointError('non-finite speed')
                rows.append((i, k, m['w_rms'], m['peak'], m['viol'], m['frac'], energy(lg), m['solve_ms']))
            except Exception as ex:
                rows.append((i, k, 'nan', 'nan', 'nan', 'nan', 'nan', 'nan')); fails.append((i, k, repr(ex)))
            print(name, 'run', i, k, rows[-1][2], flush=True)
    csvw(OUT / f'{name}_closed_loop_runs.csv', ['run', 'controller', 'w_rms', 'peak_deg', 'viol_integral', 'frac_over_gamma', 'input_energy', 'solve_ms'], rows)
    json.dump(fails, open(OUT / f'{name}_closed_loop_failures.json', 'w'))
    import csv as _c
    st = {(int(r['run']), r['controller']): float(r['w_rms']) for r in _c.DictReader(open(RES / f'{name}_param_baseline_closed_loop.csv'))}
    chk = {f'run{r[0]} {r[1]}': (r[2], st[(r[0], r[1])], abs(float(r[2]) - st[(r[0], r[1])])) for r in rows if (r[0], r[1]) in st and r[2] != 'nan'}
    json.dump(chk, open(OUT / f'{name}_closed_loop_stored_check.json', 'w'), indent=1); print(name, 'CL stored-value check max abs diff w_rms', max(v[2] for v in chk.values()), flush=True)


if __name__ == '__main__':
    nm = sys.argv[1]; part = sys.argv[2] if len(sys.argv) > 2 else 'both'
    if part in ('pred', 'both'): prediction(nm)
    if part in ('cl', 'both'): closed_loop(nm)
