"""Dispatch reconciliation, recovery and mild-shift tests for the residual trust gate (case9 / case39).

A. Atlas with the gate boundary: for every atlas cell the share of samples at which the gate (steady mismatch, no reconciliation) falls back.
B. Prediction tests: mild and large unannounced steps of load and dispatch, and a step that is later undone (recovery).
   Variants: A0, A2, A2 with the true dispatch (oracle), the plain gate, and the gate with dispatch reconciliation.
C. Closed loop after a mild unannounced step.
"""
import sys
import numpy as np
import pandas as pd
from network_validity_gate import *
from network_trust import Reconciler, reconcile_series, innovation_score

A_GATE = 0.25                         # chosen on the validation scenarios (results/*_gate_validation.csv); the regularity weight b is not used
T_REST = 6.0
SEG_DURING = (T_SHIFT + 0.6, T_REST - 1.0)


# ----------------------------------------------------------------------------- A. atlas with gate boundary
def with_pd(lg, cp, load):
    lamp, lom, lph = load
    lg = dict(lg); lg['Pd'] = np.array([cp['Pd'] * (1 + lamp * np.sin(lom * t + lph)) for t in lg['t']]); return lg


def eta_ref(c0):
    _, Jn = alg_res_jac(np.zeros(c0['n']), np.zeros(c0['ng']), c0['Pd'], *args_of(c0)[:3], c0['gbus'], c0['bg'], True)
    return float(np.cos(GAMMA) * np.linalg.eigvalsh(Jn).min())


def atlas_gate(name, c0, coef, q0):
    at = pd.read_csv(RES / f'{name}_atlas.csv'); rows = []; bref = eta_ref(c0); lam_nom = regularity(c0, *equilibrium(c0)[:2])[0]
    for mu in sorted(at.load_scale.unique()):
        cp = build_case(name, load_scale=mu); thp, dep, _ = equilibrium(cp)
        runs = make_test_runs(cp, thp, dep, 4, 8100 + int(100 * mu), 0.06)
        eta = float(at[at.load_scale == mu].lambda_min_ratio.iloc[0] * lam_nom / bref)
        for kap in sorted(at.dispatch_error.unique()):
            ratio = float(at[(at.load_scale == mu) & (at.dispatch_error == kap)].ratio_A2_over_A0.iloc[0]); fb = []
            for rr in runs:
                load, lg = rr
                lg = with_pd(lg, cp, load); cq = with_pm(cp, kap)
                tau, q, _ = reconcile_series(cq, coef, lg, q0, A_GATE, False, np.random.default_rng(5))
                fb.append(np.mean(tau[1:] < 0.5))
            rows.append((mu, kap, eta, ratio, float(np.mean(fb))))
        print(name, 'atlas gate mu', mu, flush=True)
    csv(f'{name}_atlas_gate.csv', ['load_scale', 'dispatch_error', 'eta_J', 'ratio_A2_over_A0', 'gate_fallback_share'], rows)
    return rows


# ----------------------------------------------------------------------------- B. prediction tests
def make_runs(c0, th_eq, de_eq, mu1, t_restore, n, seed, lamp_hi, T, name):
    c1 = build_case(name, load_scale=mu1) if mu1 is not None else c0
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        de0 = de_eq + rng.normal(0, 0.10, c0['ng']); w0 = rng.normal(0, 0.3, c0['ng'])
        th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
        load = (rng.uniform(0.0, lamp_hi), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        lg = simulate_shift(c0, c1, T_SHIFT, th0, de0, w0, static_ctrl(c0), T, load, t_restore=t_restore)
        out.append((load, lg, c1))
    return out


def true_pm(t, c0, c1, t_restore):
    return c1['Pm'] if (t >= T_SHIFT - 1e-9 and (t_restore is None or t < t_restore - 1e-9)) else c0['Pm']


def variant_errors(c0, coef, run, t_restore, tau_g, tau_r, rho_r, N=10, stride=4):
    load, lg, c1 = run; out = []
    for k in range(1, len(lg['t']) - 1 - N, stride):
        Pd_k = lg['Pd'][k]; Pm_nom = c0['Pm']; Pm_true = true_pm(lg['t'][k], c0, c1, t_restore)
        spec = ((0, 0, coef * 0, Pm_nom), (1, 1, coef, Pm_nom), (1, 1, coef, Pm_true), (1, 1, coef * tau_g[k], Pm_nom), (1, 1, coef * tau_r[k], rho_r[k] * Pm_nom))
        err = []
        for mode, ur, cf, Pm in spec:
            cf = cf.copy(); cf[:, NF] = coef[:, NF]
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
            for j in range(N):
                for _ in range(NSUB):
                    de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, mode, ur, cf, *args_of(c0), c0['M'], c0['Dphys'], Pm, c0['kappa'], c0['fric'])
            coi_p = np.sum(c0['M'] * de) / np.sum(c0['M']); de_t = lg['de'][k + N]; coi_t = np.sum(c0['M'] * de_t) / np.sum(c0['M'])
            err.append(np.mean(np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + N]] ** 2))
        out.append((lg['t'][k],) + tuple(err))
    return np.array(out)


def evaluate_runs(c0, coef, runs, t_restore, q0, seed, sig=SIG_MEAS):
    res = []
    for i, run in enumerate(runs):
        lg = run[1]
        tg, qg, _ = reconcile_series(c0, coef, lg, q0, A_GATE, False, np.random.default_rng(seed + i), sig)
        tr, qr, rr = reconcile_series(c0, coef, lg, q0, A_GATE, True, np.random.default_rng(seed + i), sig)
        W = variant_errors(c0, coef, run, t_restore, tg, tr, rr)
        res.append(dict(t=lg['t'], tau_g=tg, tau_r=tr, rho=rr, q_r=qr, q_g=qg, W=W))
    return res


def seg_rmse(res, lo, hi):
    E = np.vstack([r['W'][(r['W'][:, 0] >= lo) & (r['W'][:, 0] <= hi)] for r in res])
    return np.sqrt(E[:, 1:].mean(axis=0))                     # A0, A2, oracle, gate, reconcile


def first_time(t, mask, t_from):
    k = np.where((t >= t_from - 1e-9) & mask)[0]
    return float(t[k[0]]) if len(k) else np.nan


def event_metrics(res, t_event, t_end):
    """Detection (q > 1 after the event), minimum trust, and the time from detection until the trust is back at 0.8."""
    det, mint, ret, miss = [], [], [], 0
    for r in res:
        t = r['t']; win = (t >= t_event - 1e-9) & (t <= t_end)
        td = first_time(t, (r['q_r'] > 1.0) & win, t_event)
        if np.isnan(td): miss += 1; continue
        det.append(td - t_event); mint.append(float(r['tau_r'][win].min()))
        tr = first_time(t, (r['tau_r'] >= 0.8) & (t > td + 1e-9) & win, td); ret.append(tr - td if not np.isnan(tr) else np.nan)
    n = len(res)
    return (1 - miss / n, float(np.nanmean(det)) if det else np.nan, float(np.mean(mint)) if mint else np.nan, float(np.nanmean(ret)) if ret else np.nan)


def prediction_tests(name, c0, th_eq, de_eq, coef, q0):
    nrun = 6 if name == 'case9' else 4
    scen = [('no change', None, None, 0.04, 8.0, 9501), ('false-shift stress', None, None, 0.08, 8.0, 9502)]
    for mu1 in (1.03, 1.05, 1.10, 0.8, 1.25, 1.5):
        scen.append((f'persistent mu {mu1}', mu1, None, 0.06, 8.0, 9510 + int(100 * mu1)))
    for mu1 in (1.10, 1.25):
        scen.append((f'recovery mu {mu1}', mu1, T_REST, 0.06, 10.0, 9530 + int(100 * mu1)))
    rows = []; keep = {}
    for lab, mu1, tr_, lamp, T, seed in scen:
        runs = make_runs(c0, th_eq, de_eq, mu1, tr_, nrun, seed, lamp, T, name)
        sig = 3 * SIG_MEAS if 'stress' in lab else SIG_MEAS
        res = evaluate_runs(c0, coef, runs, tr_, q0, seed + 5, sig); keep[lab] = (runs, res)
        if mu1 is None:
            rm = seg_rmse(res, 0.4, T - 1.0); det = (np.nan,) * 4
            ff = float(np.mean(np.concatenate([r['tau_r'][r['t'] > 0.4] < 0.5 for r in res]))); rho_est = float(np.mean([np.mean(np.abs(r['rho'] - 1)) for r in res]))
            rows.append((lab, mu1, *rm, rho_est, ff, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            hi = T - 1.0 if tr_ is None else T_REST - 1.0
            rm = seg_rmse(res, T_SHIFT + 0.6, hi)
            dr, dd, mt, rt = event_metrics(res, T_SHIFT, T_SHIFT + 2.0)
            rho_est = float(np.mean([np.mean(r['rho'][(r['t'] >= T_SHIFT + 0.6) & (r['t'] <= (T - 0.5 if tr_ is None else T_REST - 0.1))]) for r in res]))
            ff = float(np.mean(np.concatenate([r['tau_r'][(r['t'] > 0.4) & (r['t'] < T_SHIFT)] < 0.5 for r in res])))
            if tr_ is not None:
                rm2 = seg_rmse(res, T_REST + 0.6, T - 1.0); dr2, dd2, mt2, rt2 = event_metrics(res, T_REST, T_REST + 2.0)
            else:
                rm2 = (np.nan,) * 5; dr2 = dd2 = mt2 = rt2 = np.nan
            rows.append((lab, mu1, *rm, rho_est, ff, dr, dd, mt, rt, *rm2[1:2], dr2, dd2))
        print(name, lab, [round(float(x), 3) if x == x else x for x in rows[-1][2:]], flush=True)
    hdr = ['scenario', 'mu_after', 'rmse_A0', 'rmse_A2', 'rmse_oracle', 'rmse_gate', 'rmse_reconcile', 'mean_rho_estimate', 'false_fallback_rate',
           'detection_rate', 'detection_delay_s', 'min_trust', 'retrust_delay_s', 'rmse_A2_after_restore', 'detection_rate_restore', 'detection_delay_restore_s']
    csv(f'{name}_reconcile_prediction.csv', hdr, rows)
    # rmse after restoration for all variants (recovery scenarios only)
    rec_rows = []
    for lab in [s[0] for s in scen if s[2] is not None]:
        runs, res = keep[lab]; rm = seg_rmse(res, T_REST + 0.6, 10.0 - 1.0); rec_rows.append((lab,) + tuple(float(x) for x in rm))
    csv(f'{name}_reconcile_after_restore.csv', ['scenario', 'rmse_A0', 'rmse_A2', 'rmse_oracle', 'rmse_gate', 'rmse_reconcile'], rec_rows)
    # traces for the figure
    runs, res = keep['recovery mu 1.25']; r0 = res[0]
    np.savez(RES / f'{name}_recovery_trace.npz', t=r0['t'], tau_g=r0['tau_g'], tau_r=r0['tau_r'], rho=r0['rho'], W=r0['W'], mu1=1.25, t_shift=T_SHIFT, t_rest=T_REST)
    return rows


# ----------------------------------------------------------------------------- C. closed loop, mild shift
def make_rec_mpc(c, coef, Pm_fn, ref_fn, q0, a, kind, rng, N=5, maxiter=12, qd=0.5, qw=20.0, r=0.02, rd=0.2, pen=3e3):
    """Projected-predictor MPC with exact gradient.  kind: 'A2' (tau = 1), 'gate' (trust gate), 'rec' (gate with dispatch reconciliation), 'pp' (tau = 0)."""
    ng = c['ng']; st = {'U': np.zeros(N * ng), 'prev': None, 'tau': [], 'rho': []}
    rec = Reconciler(c, coef, q0, a, kind == 'rec') if kind in ('gate', 'rec') else None
    bounds = [(-c['umax'][i], c['umax'][i]) for _ in range(N) for i in range(ng)]; eps0 = np.zeros(N + 1)

    def ctrl(t, de, w, th, up, Pd):
        dm = de + rng.normal(0, SIG_MEAS, ng); wm = w + rng.normal(0, SIG_MEAS, ng)
        tau, rho = (0.0 if kind == 'pp' else 1.0), 1.0
        if rec is not None and st['prev'] is not None:
            tau, _, rho = rec.step(st['prev'][0], (dm, wm), up, st['prev'][1])
        elif rec is not None:
            rho = rec.rho
        st['prev'] = ((dm, wm), Pd.copy()); st['tau'].append(tau); st['rho'].append(rho)
        Pm = rho * c['Pm'] if kind == 'rec' else Pm_fn(t)
        cf = coef * tau; cf[:, NF] = coef[:, NF]
        U0 = np.r_[st['U'][ng:], st['U'][-ng:]]
        args = (de, w, th, up, N, TS, NSUB, 1, 1, cf, Pd, c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['Dphys'], Pm, c['kappa'], c['fric'],
                ref_fn(t), c['umax'], qd, qw, r, rd, GAMMA, eps0, pen)
        res = minimize(lambda U: mpc_cost_net_analytic(U, *args), U0, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=maxiter))
        st['U'] = res.x
        return res.x[:ng].copy()
    ctrl.state = st
    return ctrl


def closed_loop_mild(name, c0, th_eq, de_eq, coef, q0):
    nreal = 6 if name == 'case9' else 4; Tcl = 6.0 if name == 'case9' else 5.5; rows = []; runs_out = []
    for mu1 in (1.05, 1.10):
        c1 = build_case(name, load_scale=mu1); th1, de1, _ = equilibrium(c1)
        ref0 = de_eq - np.sum(c0['M'] * de_eq) / np.sum(c0['M']); ref1 = de1 - np.sum(c1['M'] * de1) / np.sum(c1['M'])
        ref_fn = lambda t: ref0 if t < T_SHIFT - 1e-9 else ref1
        pm_nom = lambda t: c0['Pm']; pm_true = lambda t: c1['Pm'] if t >= T_SHIFT - 1e-9 else c0['Pm']
        th0w, _ = solve_theta(th_eq, de_eq, c0['Pd'], *args_of(c0), 1e-12, 30)
        for kd in ('A2', 'rec'): make_rec_mpc(c0, coef, pm_nom, ref_fn, q0, A_GATE, kd, np.random.default_rng(0))(0.0, de_eq, np.zeros(c0['ng']), th0w, np.zeros(c0['ng']), c0['Pd'])
        rng = np.random.default_rng(6201); cols = {}
        for i in range(nreal):
            load = (rng.uniform(0.04, 0.08), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
            de0 = de_eq + rng.normal(0, 0.03, c0['ng']); w0 = rng.normal(0, 0.1, c0['ng'])
            th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
            for nm in ('C0', 'A0', 'A2', 'gate', 'rec', 'oracle'):
                if nm == 'C0': ctrl = static_ctrl(c0)
                elif nm == 'A0':
                    cA = dict(c0); b0 = make_net_mpc(c0, 'A0', coef, ref0, np.zeros(6)); b1 = make_net_mpc(c0, 'A0', coef, ref1, np.zeros(6))
                    ctrl = (lambda b0, b1: (lambda t, de, w, th, up, Pd: (b0 if t < T_SHIFT - 1e-9 else b1)(t, de, w, th, up, Pd)))(b0, b1)
                elif nm == 'oracle': ctrl = make_rec_mpc(c0, coef, pm_true, ref_fn, q0, A_GATE, 'A2', np.random.default_rng(77 + i))
                else: ctrl = make_rec_mpc(c0, coef, pm_nom, ref_fn, q0, A_GATE, nm, np.random.default_rng(77 + i))
                lg = simulate_shift(c0, c1, T_SHIFT, th0, de0, w0, ctrl, Tcl, load)
                post = lg['t'] >= T_SHIFT + 0.5
                m = dict(w_rms=float(np.sqrt(np.mean(lg['w'][post] ** 2))), u_energy=float(np.sum((lg['u'][post] / c0['umax']) ** 2) * TS),
                         sat=float(np.mean(np.abs(lg['u'][post]) >= 0.999 * c0['umax'])), solve_ms=lg['solve'] * 1e3)
                if nm in ('gate', 'rec'):
                    tt = np.array(ctrl.state['tau']); m['mean_tau'] = float(np.mean(tt[post[:len(tt)]])); m['min_tau'] = float(tt[post[:len(tt)]].min())
                    m['rho_end'] = float(np.mean(ctrl.state['rho'][-5:]))
                cols.setdefault(nm, []).append(m); runs_out.append((mu1, i, nm, m))
            print(name, 'mild', mu1, 'run', i, {n: round(v[-1]['w_rms'], 4) for n, v in cols.items()}, flush=True)
        for nm, ms in cols.items():
            rows.append((mu1, nm, float(np.mean([m['w_rms'] for m in ms])), float(np.mean([m['u_energy'] for m in ms])), float(np.mean([m['sat'] for m in ms])),
                         float(np.mean([m['solve_ms'] for m in ms])), float(np.mean([m.get('mean_tau', np.nan) for m in ms])), float(np.mean([m.get('min_tau', np.nan) for m in ms])), float(np.mean([m.get('rho_end', np.nan) for m in ms]))))
    csv(f'{name}_reconcile_closed_loop.csv', ['mu_after', 'controller', 'w_rms', 'u_energy', 'input_at_bound_share', 'solve_ms', 'mean_tau_after', 'min_tau_after', 'rho_end'], rows)
    csv(f'{name}_reconcile_closed_loop_runs.csv', ['mu_after', 'run', 'controller', 'w_rms', 'u_energy', 'input_at_bound_share'], [(a, b, c, m['w_rms'], m['u_energy'], m['sat']) for a, b, c, m in runs_out])
    return rows


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['case9', 'case39']):
        c0, th_eq, de_eq, coef = nominal_fit(nm); lam_ref = regularity(c0, th_eq, de_eq)[0]
        q0, _ = calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, nm); print(nm, 'q0', q0, flush=True)
        atlas_gate(nm, c0, coef, q0)
        prediction_tests(nm, c0, th_eq, de_eq, coef, q0)
        closed_loop_mild(nm, c0, th_eq, de_eq, coef, q0)
    print('done')
