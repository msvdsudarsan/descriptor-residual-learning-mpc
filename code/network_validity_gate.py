"""Validity atlas, residual trust gate (SP-RTG) and its closed-loop test for case9 / case39.

A. Atlas: ratio A2/A0 of the 1 s prediction error over load scale mu (descriptor regularity axis) and dispatch error kappa
   (operating-compatibility axis).  The plant follows the load with its dispatch; the predictor is told the load and uses
   the dispatch Pm_plant * (1 + kappa).  kappa = 0 is a known dispatch; kappa = 1/mu - 1 reproduces Table 5.
B. Gate: prediction experiments with an abrupt load-and-dispatch step at 3 s, announced to the predictor or not, plus
   no-change and false-shift stress runs.  Gate constants (a, b) are chosen on validation scenarios, then frozen.
C. Closed loop: C0, A0, A2, A2 with tau = 0 (physics + projection) and SP-RTG after the same step.
"""
import sys, time
import numpy as np
from network_trust import *
from network_gradient import mpc_cost_net_analytic

MUS = (0.6, 1.0, 1.5, 2.0, 2.5, 3.0)
KAPPAS = (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)
T_RUN, T_SHIFT = 8.0, 3.0
A_GRID, B_GRID = (0.25, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0)
TAU_TRUST, TAU_FALL = 0.8, 0.2


def nominal_fit(name):
    f = RES / f'coef_nominal_{name}.npy'
    c = build_case(name); th, de, _ = equilibrium(c)
    if f.exists():
        return c, th, de, np.load(f)
    train = make_training(c, th, de, 30, 1001, 1e-3); val = make_training(c, th, de, 8, 1002, 1e-3)
    coef = fit_net(c, train, val, th)[0]; np.save(f, coef)
    return c, th, de, coef


def with_pm(c, kappa):
    cq = dict(c); cq['Pm'] = c['Pm'] * (1 + kappa); return cq


# ----------------------------------------------------------------------------- A. atlas
def atlas(name, c0, coef):
    rows = []; lam_ref = regularity(c0, *equilibrium(c0)[:2])[0]
    for mu in MUS:
        cp = build_case(name, load_scale=mu); thp, dep, rp = equilibrium(cp)
        if rp > 1e-8: continue
        lam, cond, amax, _ = regularity(cp, thp, dep)
        runs = make_test_runs(cp, thp, dep, 4, 8100 + int(100 * mu), 0.06)
        for kap in KAPPAS:
            cq = with_pm(cp, kap); r0 = []; r2 = []
            for rr in runs:
                e0 = predict_windows(cq, [rr], 'A0', coef, stride=6)[0]; e2 = predict_windows(cq, [rr], 'A2', coef, stride=6)[0]
                r0.append(np.mean(np.vstack(e0) ** 2)); r2.append(np.mean(np.vstack(e2) ** 2))
            r0 = np.array(r0); r2 = np.array(r2)
            ok = np.isfinite(r0) & np.isfinite(r2)
            ratio = float(np.sqrt(np.mean(r2[ok]) / np.mean(r0[ok]))) if ok.any() else float('nan')
            rows.append((mu, kap, lam / lam_ref, cond, np.degrees(amax) if amax < 3 else amax, ratio, float(np.sqrt(np.mean(r0[ok]))), float(np.sqrt(np.mean(r2[ok])))))
        print(name, 'atlas mu', mu, [round(r[5], 2) for r in rows if r[0] == mu], flush=True)
    csv(f'{name}_atlas.csv', ['load_scale', 'dispatch_error', 'lambda_min_ratio', 'cond_J', 'max_line_angle_deg', 'ratio_A2_over_A0', 'rmse_A0', 'rmse_A2'], rows)
    return rows


# ----------------------------------------------------------------------------- B. gate, prediction level
def make_shift_runs(c0, th_eq, de_eq, mu1, n, seed, lamp_hi, announced, name):
    c1 = build_case(name, load_scale=mu1) if mu1 is not None else c0
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        de0 = de_eq + rng.normal(0, 0.10, c0['ng']); w0 = rng.normal(0, 0.3, c0['ng'])
        th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
        load = (rng.uniform(0.0, lamp_hi), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        lg = simulate_shift(c0, c1, T_SHIFT, th0, de0, w0, static_ctrl(c0), T_RUN, load)
        pm_fn = (lambda t, c1=c1, c0=c0: c1['Pm'] if t >= T_SHIFT - 1e-9 else c0['Pm']) if announced else (lambda t, c0=c0: c0['Pm'])
        out.append((load, lg, pm_fn))
    return out


def window_errors(c0, coef, run, tau, N=10, stride=4, k_from=0):
    """1 s prediction error per window (A0, A2, gated, physics+projection) with the predictor's own dispatch belief."""
    load, lg, pm_fn = run
    out = []
    for k in range(k_from, len(lg['t']) - 1 - N, stride):
        Pd_k = lg['Pd'][k]; Pm = pm_fn(lg['t'][k]); err = []
        for mode, use_res, cf in ((0, 0, coef * 0), (1, 1, coef), (1, 1, coef * tau[k]), (1, 1, coef * 0)):
            cf = cf.copy(); cf[:, NF] = coef[:, NF]
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
            for j in range(N):
                for _ in range(NSUB):
                    de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, mode, use_res, cf, *args_of(c0), c0['M'], c0['Dphys'], Pm, c0['kappa'], c0['fric'])
            coi_p = np.sum(c0['M'] * de) / np.sum(c0['M']); de_t = lg['de'][k + N]; coi_t = np.sum(c0['M'] * de_t) / np.sum(c0['M'])
            err.append(np.mean(np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + N]] ** 2))
        out.append((lg['t'][k],) + tuple(err))
    return np.array(out)


def calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, name):
    runs = make_shift_runs(c0, th_eq, de_eq, None, 10, 9001, 0.08, True, name); rng = np.random.default_rng(9002); mx = []
    for load, lg, pm_fn in runs:
        _, q, _, s = gate_series(c0, coef, pm_fn, lg, 1.0, 1.0, 0.0, lam_ref, rng)
        sm = np.array([np.mean(s[max(1, k - M_SMOOTH + 1):k + 1]) for k in range(M_SMOOTH, len(s))]); mx.append(sm.max())
    return conformal_quantile(np.array(mx), 0.05), np.array(mx)


def evaluate(c0, coef, runs, q0, a, b, lam_ref, seed, k_shift_from, sig=SIG_MEAS):
    rng = np.random.default_rng(seed); res = []
    for run in runs:
        load, lg, pm_fn = run
        tau, q, r, s = gate_series(c0, coef, pm_fn, lg, q0, a, b, lam_ref, rng, sig=sig)
        W = window_errors(c0, coef, run, tau, k_from=1)
        res.append(dict(tau=tau, q=q, W=W, t=lg['t']))
    return res


def summarize(res, t_from):
    """Aggregate window errors with t >= t_from: RMSE of A0, A2, gate, physics+projection; trust-state duty cycle; detection delay."""
    E = np.vstack([r['W'][r['W'][:, 0] >= t_from] for r in res])
    rm = np.sqrt(E[:, 1:].mean(axis=0))
    taus = np.concatenate([r['tau'][r['t'] >= t_from] for r in res])
    return rm, float(np.mean(taus >= TAU_TRUST)), float(np.mean((taus < TAU_TRUST) & (taus >= TAU_FALL))), float(np.mean(taus < TAU_FALL))


def detection_delay(res):
    d = []
    for r in res:
        k = np.where((r['t'] >= T_SHIFT - 1e-9) & (r['tau'] < 0.5))[0]
        d.append(r['t'][k[0]] - T_SHIFT if len(k) else np.nan)
    return np.array(d)


def gate_experiments(name, c0, th_eq, de_eq, coef):
    lam_ref = regularity(c0, th_eq, de_eq)[0]
    q0, mx = calibrate_q0(c0, th_eq, de_eq, coef, lam_ref, name); print(name, 'q0', q0, mx, flush=True)
    nrun = 6 if name == 'case9' else 4
    # --- validation: choose (a, b)
    val = []
    for mu1, ann, seed in ((0.8, True, 9101), (1.5, True, 9102), (0.8, False, 9103), (1.25, False, 9104), (1.5, False, 9105)):
        val.append(make_shift_runs(c0, th_eq, de_eq, mu1, 3, seed, 0.06, ann, name))
    grid = []
    for a in A_GRID:
        for b in B_GRID:
            sc = []
            for runs in val:
                rm, *_ = summarize(evaluate(c0, coef, runs, q0, a, b, lam_ref, 9200, 1), T_SHIFT + 0.3)
                sc.append(rm[2] / rm[0])                                      # gated RMSE over A0 RMSE
            grid.append((a, b, float(np.mean(sc))))
            print(name, 'val', grid[-1], flush=True)
    csv(f'{name}_gate_validation.csv', ['a', 'b', 'mean_ratio_gate_over_A0'], grid)
    # parsimony rule: among all (a, b) whose validation score is within 0.5% of the best, take the smallest b and then the smallest a
    top = min(g[2] for g in grid); best = min([g for g in grid if g[2] <= 1.005 * top], key=lambda x: (x[1], x[0])); a_star, b_star = best[0], best[1]
    print(name, 'chosen', a_star, b_star, flush=True)
    # --- test scenarios
    rows = []; keep = {}
    scen = [('no change', None, True, 0.04, 9301), ('false-shift stress (8% load 3x noise)', None, True, 0.08, 9302)]
    for mu1 in (0.8, 1.25, 1.5):
        scen.append((f'announced mu {mu1}', mu1, True, 0.06, 9310 + int(10 * mu1))); scen.append((f'unannounced mu {mu1}', mu1, False, 0.06, 9320 + int(10 * mu1)))
    for lab, mu1, ann, lamp, seed in scen:
        runs = make_shift_runs(c0, th_eq, de_eq, mu1, nrun, seed, lamp, ann, name)
        sig = 3 * SIG_MEAS if 'stress' in lab else SIG_MEAS
        res = evaluate(c0, coef, runs, q0, a_star, b_star, lam_ref, seed + 7, 1, sig=sig)
        pre = summarize(res, 0.0) if mu1 is None else None
        post = summarize(res, T_SHIFT + 0.3) if mu1 is not None else summarize(res, T_SHIFT + 0.3)
        rm, d_tr, d_ca, d_fb = post
        # false fallback rate: windows with tau < 0.5 before the shift (or anywhere when there is no shift)
        t_lim = T_SHIFT if mu1 is not None else T_RUN
        ff = float(np.mean(np.concatenate([r['tau'][(r['t'] < t_lim) & (r['t'] > 0.4)] < 0.5 for r in res])))
        dd = detection_delay(res) if mu1 is not None else np.array([np.nan])
        rows.append((lab, rm[0], rm[1], rm[2], rm[3], rm[1] / rm[0], rm[2] / rm[0], rm[2] / rm[1], d_tr, d_ca, d_fb, ff, float(np.nanmean(dd)) if np.isfinite(dd).any() else float('nan')))
        keep[lab] = res
        print(name, lab, [round(x, 3) for x in rows[-1][1:]], flush=True)
    csv(f'{name}_gate_prediction.csv', ['scenario', 'rmse_A0', 'rmse_A2', 'rmse_gate', 'rmse_physics_projection', 'ratio_A2_A0', 'ratio_gate_A0', 'ratio_gate_A2', 'duty_trusted', 'duty_cautious', 'duty_fallback', 'false_fallback_rate', 'detection_delay_s'], rows)
    return dict(q0=q0, a=a_star, b=b_star, lam_ref=lam_ref, rows=rows, keep=keep, grid=grid)


# ----------------------------------------------------------------------------- C. gate, closed loop
def make_gated_mpc(c, coef, Pm_fn, ref_fn, q0, a, b, lam_ref, mode_tau, rng, N=5, maxiter=12, qd=0.5, qw=20.0, r=0.02, rd=0.2, pen=3e3):
    """Projected-predictor MPC with exact gradient.  mode_tau: 'gate' (SP-RTG), float (fixed tau, 1.0 = A2, 0.0 = physics + projection)."""
    ng = c['ng']; st = {'U': np.zeros(N * ng), 'prev': None, 'hist': [], 'tau': []}
    bounds = [(-c['umax'][i], c['umax'][i]) for _ in range(N) for i in range(ng)]
    eps0 = np.zeros(N + 1)

    def ctrl(t, de, w, th, up, Pd):
        dm = de + rng.normal(0, SIG_MEAS, ng); wm = w + rng.normal(0, SIG_MEAS, ng)
        tau = 1.0 if not isinstance(mode_tau, str) else 1.0
        if st['prev'] is not None:
            s = innovation_score(c, coef, Pm_fn(t - TS), st['prev'][0], up, st['prev'][1], (dm, wm))
            st['hist'].append(s); qk = np.mean(st['hist'][-M_SMOOTH:]) / q0
            cc = dict(c); cc['Pd'] = Pd
            rk = regularity_score(cc, dm, lam_ref)
            tau = trust(qk, rk, a, b) if isinstance(mode_tau, str) else float(mode_tau)
        elif not isinstance(mode_tau, str):
            tau = float(mode_tau)
        st['prev'] = ((dm, wm), Pd.copy()); st['tau'].append(tau)
        cf = coef * tau; cf[:, NF] = coef[:, NF]
        U0 = np.r_[st['U'][ng:], st['U'][-ng:]]
        args = (de, w, th, up, N, TS, NSUB, 1, 1, cf, Pd, c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['Dphys'], Pm_fn(t), c['kappa'], c['fric'],
                ref_fn(t), c['umax'], qd, qw, r, rd, GAMMA, eps0, pen)
        res = minimize(lambda U: mpc_cost_net_analytic(U, *args), U0, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=maxiter))
        st['U'] = res.x
        return res.x[:ng].copy()
    ctrl.state = st
    return ctrl


def closed_loop(name, c0, th_eq, de_eq, coef, gate, mu1=1.5):
    c1 = build_case(name, load_scale=mu1); th1, de1, _ = equilibrium(c1)
    ref0 = de_eq - np.sum(c0['M'] * de_eq) / np.sum(c0['M']); ref1 = de1 - np.sum(c1['M'] * de1) / np.sum(c1['M'])
    ref_fn = lambda t: ref0 if t < T_SHIFT - 1e-9 else ref1                      # angle reference follows the schedule; dispatch does not
    nreal = 6 if name == 'case9' else 4
    # warm-up so that no timing contains compilation
    rng_w = np.random.default_rng(0)
    th0w, _ = solve_theta(th_eq, de_eq, c0['Pd'], *args_of(c0), 1e-12, 30)
    make_gated_mpc(c0, coef, lambda t: c0['Pm'], ref_fn, gate['q0'], gate['a'], gate['b'], gate['lam_ref'], 'gate', rng_w)(0.0, de_eq, np.zeros(c0['ng']), th0w, np.zeros(c0['ng']), c0['Pd'])
    make_net_mpc(c0, 'A0', coef, ref0, np.zeros(6))(0.0, de_eq, np.zeros(c0['ng']), th0w, np.zeros(c0['ng']), c0['Pd'])
    out = []; traces = {}
    for ann in (False, True):
        pm_fn = (lambda t: c1['Pm'] if t >= T_SHIFT - 1e-9 else c0['Pm']) if ann else (lambda t: c0['Pm'])
        rng = np.random.default_rng(6101)
        for i in range(nreal):
            load = (rng.uniform(0.04, 0.08), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
            de0 = de_eq + rng.normal(0, 0.03, c0['ng']); w0 = rng.normal(0, 0.1, c0['ng'])
            th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
            for nm in ('C0', 'A0', 'A2', 'physics+projection', 'SP-RTG'):
                if nm == 'C0': ctrl = static_ctrl(c0)
                elif nm == 'A0':
                    base = make_net_mpc(c0, 'A0', coef, ref0, np.zeros(6))
                    def ctrl(t, de, w, th, up, Pd, base=base):
                        cq = base; return cq(t, de, w, th, up, Pd)
                    # A0 uses the fixed nominal reference of make_net_mpc; re-create with the shifted reference after the step
                    c0a = dict(c0); c0a['Pm'] = c1['Pm'] if ann else c0['Pm']
                    base1 = make_net_mpc(c0a, 'A0', coef, ref1, np.zeros(6))
                    ctrl = (lambda b0, b1: (lambda t, de, w, th, up, Pd: (b0 if t < T_SHIFT - 1e-9 else b1)(t, de, w, th, up, Pd)))(base, base1)
                else:
                    mt = {'A2': 1.0, 'physics+projection': 0.0, 'SP-RTG': 'gate'}[nm]
                    ctrl = make_gated_mpc(c0, coef, pm_fn, ref_fn, gate['q0'], gate['a'], gate['b'], gate['lam_ref'], mt, np.random.default_rng(77 + i))
                lg = simulate_shift(c0, c1, T_SHIFT, th0, de0, w0, ctrl, 6.0 if name == 'case9' else 5.5, load)
                post = lg['t'] >= T_SHIFT + 0.5
                ang = np.degrees([np.max(np.abs(t_[c1['lf']] - c1['lt'] * 0 - t_[c1['lt']])) for t_ in lg['th']])
                m = dict(w_rms=float(np.sqrt(np.mean(lg['w'][post] ** 2))), u_energy=float(np.sum((lg['u'][post] / c0['umax']) ** 2) * TS), peak=float(ang.max()), solve_ms=lg['solve'] * 1e3)
                if nm == 'SP-RTG':
                    tt = np.array(ctrl.state['tau']); m['mean_tau_post'] = float(np.mean(tt[lg['t'][:len(tt)] >= T_SHIFT + 0.5])); m['mean_tau_pre'] = float(np.mean(tt[lg['t'][:len(tt)] < T_SHIFT]))
                    if i == 0: traces[('SP-RTG', ann)] = (lg['t'][:len(tt)].copy(), tt.copy())
                out.append((('announced' if ann else 'unannounced'), i, nm, m))
            print(name, 'ann' if ann else 'unann', 'run', i, {o[2]: round(o[3]['w_rms'], 3) for o in out[-5:]}, flush=True)
    keys = ['w_rms', 'u_energy', 'peak', 'solve_ms']
    rows = []
    for ann in ('unannounced', 'announced'):
        for nm in ('C0', 'A0', 'A2', 'physics+projection', 'SP-RTG'):
            ms = [o[3] for o in out if o[0] == ann and o[2] == nm]
            rows.append((ann, nm) + tuple(float(np.mean([m[k] for m in ms])) for k in keys) + (float(np.mean([m.get('mean_tau_post', np.nan) for m in ms])),))
    csv(f'{name}_gate_closed_loop.csv', ['dispatch', 'controller'] + keys + ['mean_tau_post'], rows)
    csv(f'{name}_gate_closed_loop_runs.csv', ['dispatch', 'run', 'controller'] + keys, [(o[0], o[1], o[2]) + tuple(o[3][k] for k in keys) for o in out])
    return rows, traces


# ----------------------------------------------------------------------------- figures
def atlas_figure(res):
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, nm in zip(axs, ('case9', 'case39')):
        rows = res[nm]['atlas']; mus = sorted(set(r[0] for r in rows)); kaps = sorted(set(r[1] for r in rows))
        Z = np.full((len(kaps), len(mus)), np.nan)
        for r in rows: Z[kaps.index(r[1]), mus.index(r[0])] = r[5]
        im = ax.imshow(np.log2(Z), origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
        for i in range(len(kaps)):
            for j in range(len(mus)):
                if np.isfinite(Z[i, j]): ax.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', fontsize=5.5, color='k')
        lamr = {r[0]: r[2] for r in rows}
        ax.set_xticks(range(len(mus))); ax.set_xticklabels([f'{m:g}\n({lamr[m]:.2f})' for m in mus], fontsize=6)
        ax.set_yticks(range(len(kaps))); ax.set_yticklabels([f'{k:+.1f}' for k in kaps], fontsize=6)
        ax.set_xlabel('load scale $\\mu$  (and $\\lambda_{\\min}(J)/\\lambda_{\\min}^{\\rm nom}$)'); ax.set_ylabel('dispatch error $\\kappa$')
        ax.set_title(f'({"ab"[("case9", "case39").index(nm)]}) {nm}'); ax.grid(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label('$\\log_2$ (A2/A0 error ratio)', fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, 'fig_validity_atlas')


def gate_figure(res):
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 4.2), sharex='col')
    for k, nm in enumerate(('case9', 'case39')):
        keep = res[nm]['gate']['keep']; r = keep['unannounced mu 1.5'][0]; ra = keep['announced mu 1.5'][0]
        ax = axs[0, k]
        ax.axvspan(0, 8, color='none'); ax.axvline(T_SHIFT, color='k', ls=':', lw=.8)
        ax.plot(r['t'], r['tau'], color='#d62728', lw=1.1, label='unannounced dispatch')
        ax.plot(ra['t'], ra['tau'], color='#2ca02c', lw=1.1, label='announced dispatch')
        ax.axhspan(TAU_TRUST, 1.02, color='#2ca02c', alpha=.07); ax.axhspan(TAU_FALL, TAU_TRUST, color='#ff7f0e', alpha=.07); ax.axhspan(-0.02, TAU_FALL, color='#d62728', alpha=.07)
        ax.set_ylabel('trust $\\tau$'); ax.set_title(f'({"ab"[k]}) {nm}, step to $\\mu=1.5$ at 3 s'); ax.legend(fontsize=6, loc='center right')
        ax = axs[1, k]; W = r['W']
        for col, lab, c_ in ((1, 'A0', COL['A0']), (2, 'A2', COL['A2']), (3, 'SP-RTG', COL['A3'])):
            ax.semilogy(W[:, 0], np.sqrt(W[:, col]), color=c_, lw=1.1, label=lab)
        ax.axvline(T_SHIFT, color='k', ls=':', lw=.8); ax.set_ylabel('1 s prediction RMSE'); ax.set_xlabel('window start [s]'); ax.legend(fontsize=6, ncol=3)
    fig.tight_layout(); save(fig, 'fig_trust_gate')


if __name__ == '__main__':
    which = sys.argv[1:] or ['case9', 'case39']
    res = {}
    for nm in which:
        c0, th_eq, de_eq, coef = nominal_fit(nm)
        af = RES / f'{nm}_atlas.csv'
        if af.exists():
            res[nm] = {'atlas': [tuple(float(x) for x in l.split(',')) for l in af.read_text().strip().split('\n')[1:]]}
        else:
            res[nm] = {'atlas': atlas(nm, c0, coef)}
        res[nm]['gate'] = gate_experiments(nm, c0, th_eq, de_eq, coef)
        cl_rows, traces = closed_loop(nm, c0, th_eq, de_eq, coef, res[nm]['gate'])
        print(nm, 'closed loop', cl_rows, flush=True)
    if len(res) == 2:
        atlas_figure(res); gate_figure(res)
    print('done')
