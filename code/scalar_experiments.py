"""Scalar-benchmark experiments E1-E5. Writes CSVs to ../results and figures to ../figures."""
import sys, time, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common import *
from scalar_benchmark import *

import os
SEV = float(os.environ.get('PHYS_SEVERITY', '1.0')); TAG = '' if SEV == 1.0 else f'_sev{SEV:g}'
ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / 'results', ROOT / 'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 120,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
S = MPCSettings
ZERO = np.zeros(1); E0 = np.zeros((1, 3), dtype=np.int64)
UFF = equilibrium_input(D_REF, PHYS_PM, PHYS_KV, PHYS_C3)       # feedforward computed from the physics-only model
COL = {'A0': '#7f7f7f', 'A1': '#1f77b4', 'A2': '#2ca02c', 'A3': '#d62728', 'C0': '#9467bd'}


def save(fig, name):
    name = name + TAG
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def csv(name, header, rows):
    name = name.replace('.csv', TAG + '.csv')
    with open(RES / name, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'{x:.8g}' if isinstance(x, (float, np.floating)) else str(x) for x in r) + '\n')


def model_args(name, coef, E):
    mode, use_res = {'A0': (0, 0), 'A1': (0, 1), 'A2': (1, 1), 'A3': (1, 1)}[name]
    return (coef, E) if use_res else (ZERO, E0), mode, use_res


def closed_loop_runs(n, seed, amp_lo, amp_hi):
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        dist = (rng.uniform(amp_lo, amp_hi), rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi))
        x0 = (rng.uniform(0.15, 0.30), rng.uniform(-0.05, 0.05))
        log, _ = run_closed_loop(lambda t, d, w, up: static_law(d, w, UFF), x0, 12.0, dist)
        runs.append((dist, x0, log))
    return runs


def make_mpc(name, coef, E, eps=None):
    (c_, E_), mode, use_res = model_args(name, coef, E)
    P, _, _ = terminal_weight(c_, E_, use_res, D_REF, S.Ts, S.qd, S.qw, S.r)
    e = np.zeros((S.N + 1, 2)) if (eps is None or name != 'A3') else eps
    st = {'U': np.zeros(S.N)}

    def ctrl(t, d, w, uprev):
        U0 = np.r_[st['U'][1:], st['U'][-1]]
        a = (d, w, uprev, S.N, S.Ts, S.nsub, mode, use_res, c_, E_, D_REF, S.qd, S.qw, S.r, S.rd,
             P[0, 0], P[0, 1], P[1, 1], D_MAX, W_MAX, e, S.pen)
        r = minimize(lambda U: mpc_cost_grad(U, *a), U0, jac=True, method='L-BFGS-B',
                     bounds=[(-U_MAX, U_MAX)] * S.N, options=dict(maxiter=60))
        st['U'] = r.x
        return float(r.x[0])
    return ctrl


def kstep_errors(runs, name, coef, E, stride=3):
    (c_, E_), mode, use_res = model_args(name, coef, E)
    err = [[] for _ in range(S.N)]; gres = []
    for dist, x0, log in runs:
        for k in range(0, len(log) - S.N - 1, stride):
            U = log[k:k + S.N, 4].copy()
            o = rollout(log[k, 1], log[k, 2], U, log[max(k - 1, 0), 4] if k > 0 else U[0], S.Ts, S.nsub, mode, use_res, c_, E_)
            for j in range(S.N):
                err[j].append((o[j + 1, 0] - log[k + j + 1, 1], o[j + 1, 1] - log[k + j + 1, 2]))
            gres.append(max(abs(g_res(o[j, 0], o[j, 2], (log[k - 1, 4] if (j == 0 and k > 0) else (U[0] if j == 0 else U[j - 1])))) for j in range(S.N + 1)))
    return [np.array(e) for e in err], np.array(gres)


def run_metrics(log):
    e = log[:, 1] - D_REF
    return dict(iae=float(np.sum(np.abs(e)) * S.Ts), last=float(np.mean(np.abs(e[-30:]))), dpeak=float(log[:, 1].max()),
                viol=float(np.sum(np.maximum(0, log[:, 1] - D_MAX)) * S.Ts), u2=float(np.sum(log[:, 4] ** 2) * S.Ts),
                wpeak=float(np.abs(log[:, 2]).max()))


def main():
    t_all = time.time()
    summary = {}
    # ------------------------------------------------------------------ E1 residual identification
    train = make_identification_data(24, 11, 1e-3); val = make_identification_data(8, 12, 1e-3)
    fit = fit_residual(train, val); coef, E = fit['coef'], fit['E']
    test_pts, test_tr = collect_test_points(16, 13)
    ex = np.array([exact_residual(*p) for p in test_pts]); hat = np.array([res_eval(p[0], p[1], p[2], coef, E) for p in test_pts])
    # measurement-based residual errors (noise and training-level disturbance included), kept per trajectory
    def weak_res_list(trs):
        out = []
        for tr in trs:
            Th, y = weak_form_targets(tr, E); out.append(y - Th @ coef)
        return out

    def block_max(arrs, L=20):                       # non-overlapping blocks of 1 s
        return np.array([np.max(np.abs(a[i:i + L])) for a in arrs for i in range(0, len(a) - L + 1, L)])
    cal_list = weak_res_list(make_identification_data(10, 15, 1e-3))      # independent calibration set
    test_list = weak_res_list(make_identification_data(16, 14, 1e-3))     # independent test set
    rho_pt = conformal_quantile(np.abs(np.concatenate(cal_list)), 0.05)
    cover_pt = float(np.mean(np.abs(np.concatenate(test_list)) <= rho_pt))
    rho_cal = conformal_quantile(block_max(cal_list), 0.05)
    cover = float(np.mean(block_max(test_list) <= rho_cal))
    test_meas = np.concatenate(test_list)
    rms_ex = float(np.sqrt(np.mean(ex ** 2))); rms_err = float(np.sqrt(np.mean((ex - hat) ** 2)))
    nact = int(np.sum(coef != 0))
    csv('scalar_residual_fit.csv', ['quantity', 'value'], [
        ('library_degree', fit['degree']), ('library_terms', len(coef)), ('active_terms', nact),
        ('ridge', fit['ridge']), ('threshold', fit['thresh']),
        ('residual_rms_true_test', rms_ex), ('residual_rms_error_after_fit', rms_err),
        ('error_reduction_percent', 100 * (1 - rms_err / rms_ex)), ('max_abs_error_test', float(np.max(np.abs(ex - hat)))),
        ('rho_pointwise_95', rho_pt), ('coverage_pointwise_test', cover_pt),
        ('rho_block_95', rho_cal), ('coverage_block_test', cover), ('n_calibration_blocks', len(block_max(cal_list))), ('n_test_blocks', len(block_max(test_list)))])
    summary.update(rms_ex=rms_ex, rms_err=rms_err, rho=rho_cal, cover=cover, rho_pt=rho_pt, cover_pt=cover_pt, nact=nact, deg=fit['degree'])
    print('E1', summary)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax[0].scatter(ex, hat, s=3, alpha=.4, color=COL['A1']); lim = [ex.min(), ex.max()]; ax[0].plot(lim, lim, 'k--', lw=.8)
    ax[0].set_xlabel('true residual $d$'); ax[0].set_ylabel('learned $\\hat d$'); ax[0].set_title('(a) held-out residual')
    ax[1].hist(np.abs(test_meas), bins=40, color='#aaaaaa'); ax[1].axvline(rho_pt, color=COL['A3'], lw=1.2, label=f'$\\rho_{{pt}}$={rho_pt:.3f}')
    ax[1].set_xlabel('|measured residual error|'); ax[1].set_title(f'(b) pointwise bound, coverage {cover_pt:.3f}'); ax[1].legend()
    dgrid = np.linspace(0, 1.1, 60); wgrid = np.linspace(-0.7, 0.7, 60); DD, WW = np.meshgrid(dgrid, wgrid)
    ut = -0.5
    Z = np.array([[exact_residual(a, b, ut) - res_eval(a, b, ut, coef, E) for a in dgrid] for b in wgrid])
    im = ax[2].contourf(DD, WW, Z, 20, cmap='RdBu_r'); plt.colorbar(im, ax=ax[2], shrink=.8)
    ax[2].set_xlabel('$\\delta$'); ax[2].set_ylabel('$\\omega$'); ax[2].set_title('(c) error map, $u=-0.5$')
    fig.tight_layout(); save(fig, 'fig_scalar_learning')

    # ------------------------------------------------------------------ trajectory-level, policy-matched k-step envelope
    def mpc_runs(n, seed, lo, hi):
        rng = np.random.default_rng(seed); out = []
        for _ in range(n):
            dist = (rng.uniform(lo, hi), rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi)); x0 = (rng.uniform(0.15, 0.30), rng.uniform(-0.05, 0.05))
            log, _ = run_closed_loop(make_mpc('A2', coef, E), x0, 12.0, dist); out.append((dist, x0, log))
        return out

    def per_run_max(runs):
        """max over windows of |k-step error| for every run: array (n_runs, N, 2)."""
        res = []
        for r in runs:
            er, _ = kstep_errors([r], 'A2', coef, E)
            res.append([[np.max(np.abs(er[j][:, 0])), np.max(np.abs(er[j][:, 1]))] for j in range(S.N)])
        return np.array(res)
    calib_runs = closed_loop_runs(10, 21, 0.0, 0.12) + mpc_runs(10, 22, 0.0, 0.12)        # 10 static-law + 10 MPC trajectories
    pm = per_run_max(calib_runs)
    eps = np.zeros((S.N + 1, 2))
    for j in range(S.N):
        eps[j + 1, 0] = conformal_quantile(pm[:, j, 0], 0.05); eps[j + 1, 1] = conformal_quantile(pm[:, j, 1], 0.05)
    csv('scalar_tube_envelope.csv', ['k', 'eps_delta', 'eps_omega'], [(k, eps[k, 0], eps[k, 1]) for k in range(S.N + 1)])
    cov_rows = []
    for src, runs_ in (('static law', closed_loop_runs(16, 31, 0.0, 0.12)), ('MPC (A2)', mpc_runs(16, 32, 0.0, 0.12))):
        pt = per_run_max(runs_)
        cov = lambda j, o: float(np.mean(pt[:, j, o] <= eps[j + 1, o]))
        joint = float(np.mean(np.all(pt <= eps[1:, :][None, :, :], axis=(1, 2))))
        cov_rows.append((src, len(runs_), cov(5, 0), cov(5, 1), cov(11, 0), cov(11, 1), joint))
    csv('scalar_envelope_coverage.csv', ['test_source', 'n_traj', 'cover_delta_k6', 'cover_omega_k6', 'cover_delta_k12', 'cover_omega_k12', 'cover_joint_all_k'], cov_rows)
    print('tube eps', eps[[1, 6, 12]], 'coverage', cov_rows)

    # ------------------------------------------------------------------ E2 prediction ablation
    test_runs = closed_loop_runs(16, 31, 0.0, 0.12)
    rows = []; ablation = {}
    for nm in ['A0', 'A1', 'A2']:
        er, gr = kstep_errors(test_runs, nm, coef, E)
        ablation[nm] = (er, gr)
        for j in range(S.N):
            rows.append((nm, (j + 1) * S.Ts, float(np.sqrt(np.mean(np.sum(er[j] ** 2, axis=1)))), float(np.mean(np.abs(er[j][:, 0]))), float(np.mean(np.abs(er[j][:, 1])))))
    csv('scalar_prediction_ablation.csv', ['model', 'horizon_s', 'state_rmse', 'mean_abs_delta_err', 'mean_abs_omega_err'], rows)
    g_rows = [(nm, float(np.mean(ablation[nm][1])), float(np.max(ablation[nm][1]))) for nm in ablation]
    csv('scalar_algebraic_residual.csv', ['model', 'mean_max_abs_g', 'max_abs_g'], g_rows)
    # per-run paired RMSE at the 1.2 s horizon, bootstrap over runs
    def per_run_rmse(nm):
        out = []
        for ri, (dist, x0, log) in enumerate(test_runs):
            er, _ = kstep_errors([(dist, x0, log)], nm, coef, E)
            out.append(np.sqrt(np.mean(np.sum(np.vstack(er) ** 2, axis=1))))
        return np.array(out)
    r0, r1, r2 = per_run_rmse('A0'), per_run_rmse('A1'), per_run_rmse('A2')
    ratio = r2 / r0
    summary['pred_ratio'] = bootstrap_ci(ratio)
    csv('scalar_prediction_paired.csv', ['run', 'rmse_A0', 'rmse_A1', 'rmse_A2'], [(i, r0[i], r1[i], r2[i]) for i in range(len(r0))])
    print('E2 ratio A2/A0', summary['pred_ratio'], 'g', g_rows)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    hz = (np.arange(S.N) + 1) * S.Ts
    for nm in ['A0', 'A1', 'A2']:
        ax[0].plot(hz, [np.sqrt(np.mean(np.sum(ablation[nm][0][j] ** 2, axis=1))) for j in range(S.N)], marker='o', ms=2.5, color=COL[nm], label=nm)
    ax[0].set_xlabel('prediction horizon [s]'); ax[0].set_ylabel('state RMSE'); ax[0].set_title('(a) multi-step prediction error'); ax[0].legend(ncol=3)
    ax[1].semilogy(np.arange(1, len(r0) + 1), r0, 'o', color=COL['A0'], label='A0'); ax[1].semilogy(np.arange(1, len(r0) + 1), r2, 's', color=COL['A2'], label='A2')
    ax[1].set_xlabel('test realisation'); ax[1].set_ylabel('state RMSE (1.2 s windows)'); ax[1].set_title('(b) paired comparison'); ax[1].legend()
    fig.tight_layout(); save(fig, 'fig_scalar_ablation')

    # ------------------------------------------------------------------ E3 closed loop
    R = 16; rng = np.random.default_rng(41)
    conds = [(rng.uniform(0.02, 0.12), rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.30), rng.uniform(-0.05, 0.05)) for _ in range(R)]
    names = ['C0', 'A0', 'A1', 'A2', 'A3']; res = {n: [] for n in names}; logs = {}
    for i, (a, om, ph, d0, w0) in enumerate(conds):
        for n in names:
            ctrl = (lambda t, d, w, up: static_law(d, w, UFF)) if n == 'C0' else make_mpc(n, coef, E, eps)
            log, ts = run_closed_loop(ctrl, (d0, w0), 12.0, (a, om, ph))
            m = run_metrics(log); m['solve_ms'] = ts * 1e3; res[n].append(m)
            if i == 0: logs[n] = log
    keys = ['iae', 'last', 'dpeak', 'viol', 'u2', 'wpeak', 'solve_ms']
    rows = []
    for n in names:
        row = [n]
        for k in keys:
            mu, lo, hi = bootstrap_ci(np.array([m[k] for m in res[n]]))
            row += [mu, lo, hi]
        rows.append(row)
    hdr = ['controller'] + [f'{k}_{s}' for k in keys for s in ('mean', 'lo', 'hi')]
    csv('scalar_closed_loop.csv', hdr, rows)
    csv('scalar_closed_loop_runs.csv', ['run', 'controller'] + keys, [(i, n) + tuple(res[n][i][k] for k in keys) for n in names for i in range(R)])
    summary['cl'] = {n: {k: float(np.mean([m[k] for m in res[n]])) for k in keys} for n in names}
    # paired differences vs A0
    pair = []
    for n in ['C0', 'A1', 'A2', 'A3']:
        for k in ['iae', 'last', 'viol']:
            diff = np.array([res[n][i][k] - res['A0'][i][k] for i in range(R)])
            pair.append((n, k, *bootstrap_ci(diff)))
    csv('scalar_closed_loop_paired_vs_A0.csv', ['controller', 'metric', 'mean_diff', 'lo', 'hi'], pair)
    print('E3', json.dumps(summary['cl'], indent=1)[:1500])
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for n in names:
        ax[0].plot(logs[n][:, 0], logs[n][:, 1], color=COL[n], lw=1.1, label=n)
    ax[0].axhline(D_REF, color='k', lw=.6, ls=':'); ax[0].axhline(D_MAX, color='k', lw=.6, ls='--')
    ax[0].set_xlabel('time [s]'); ax[0].set_ylabel('$\\delta$'); ax[0].set_title('(a) one realisation'); ax[0].legend(ncol=2, fontsize=6)
    for n in names:
        ax[1].plot(logs[n][:, 0], logs[n][:, 4], color=COL[n], lw=1.0)
    ax[1].set_xlabel('time [s]'); ax[1].set_ylabel('$u$'); ax[1].set_title('(b) control input')
    pos = np.arange(len(names)); ax[2].boxplot([[m['iae'] for m in res[n]] for n in names], positions=pos, widths=.6)
    ax[2].set_xticks(pos); ax[2].set_xticklabels(names); ax[2].set_ylabel('IAE of $\\delta-\\delta_{ref}$'); ax[2].set_title(f'(c) {R} realisations')
    fig.tight_layout(); save(fig, 'fig_scalar_closedloop')

    if SEV != 1.0:
        json.dump(dict(cl=summary['cl'], pred_ratio=summary['pred_ratio'], rms_ex=rms_ex, rms_err=rms_err), open(RES / f'scalar_summary{TAG}.json', 'w'), indent=1, default=float)
        return
    # ------------------------------------------------------------------ E3b disturbance sweep
    rows = []
    for amp in [0.0, 0.04, 0.08, 0.12, 0.22]:
        rng = np.random.default_rng(51); vals = {n: [] for n in ['A0', 'A3']}
        for i in range(8):
            om, ph, d0 = rng.uniform(1, 3), rng.uniform(0, 6.28), rng.uniform(0.15, 0.3)
            for n in vals:
                log, _ = run_closed_loop(make_mpc(n, coef, E, eps), (d0, 0.0), 12.0, (amp, om, ph)); vals[n].append(run_metrics(log))
        rows.append((amp,) + tuple(np.mean([m[k] for m in vals[n]]) for n in ['A0', 'A3'] for k in ['iae', 'last', 'viol']))
    csv('scalar_disturbance_sweep.csv', ['amp', 'A0_iae', 'A0_last', 'A0_viol', 'A3_iae', 'A3_last', 'A3_viol'], rows)

    # ------------------------------------------------------------------ E4 learning curves
    rows = []
    for noise in [0.0, 1e-3, 3e-3, 1e-2]:
        for ntr in [2, 4, 8, 16, 32]:
            errs_ = []
            for seed in range(4):
                tr = make_identification_data(ntr, 100 + seed, noise); va = make_identification_data(6, 200 + seed, noise)
                ft = fit_residual(tr, va, degrees=(3, 4), ridges=(1e-6, 1e-4), threshes=(0.005, 0.02))
                h_ = np.array([res_eval(p[0], p[1], p[2], ft['coef'], ft['E']) for p in test_pts])
                errs_.append(np.sqrt(np.mean((ex - h_) ** 2)))
            rows.append((noise, ntr, float(np.mean(errs_)), float(np.std(errs_))))
    csv('scalar_learning_curve.csv', ['noise_std', 'n_train', 'rmse_mean', 'rmse_std'], rows)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for noise in [0.0, 1e-3, 3e-3, 1e-2]:
        r_ = [r for r in rows if r[0] == noise]; ax.errorbar([r[1] for r in r_], [r[2] for r in r_], yerr=[r[3] for r in r_], marker='o', ms=3, capsize=2, label=f'$\\sigma$={noise:g}')
    ax.axhline(rms_ex, color='k', ls=':', lw=.8); ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('training trajectories'); ax.set_ylabel('held-out residual RMSE'); ax.legend(fontsize=6); fig.tight_layout(); save(fig, 'fig_scalar_learning_curve')

    # ------------------------------------------------------------------ E4b library complexity (degree-3 library)
    E3 = monomial_exponents(3, 3)
    Trn = [weak_form_targets(t, E3) for t in make_identification_data(24, 11, 1e-3)]
    ThA = np.vstack([a_ for a_, _ in Trn]); yA = np.concatenate([b_ for _, b_ in Trn])
    PH = monomial_features(test_pts, E3); rows = []
    for thr in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        c3 = stlsq(ThA, yA, ridge=1e-6, thresh=thr)
        rows.append((thr, int(np.sum(c3 != 0)), float(np.sqrt(np.mean((ex - PH @ c3) ** 2)))))
    csv('scalar_complexity.csv', ['threshold', 'active_terms', 'heldout_residual_rmse'], rows)

    # ------------------------------------------------------------------ E5 nonlinear analysis
    ps = np.linspace(-0.6, 0.9, 151)
    br_fb = equilibria_branch(ps, UFF, D_REF, True); br_ol = equilibria_branch(ps, UFF, D_REF, False)
    rows = [('feedback', r[0], r[1], r[2]) for r in br_fb] + [('open_loop', r[0], r[1], r[2]) for r in br_ol]
    csv('scalar_equilibria.csv', ['loop', 'bias_force', 'delta_eq', 'max_real_eig'], rows)
    nom = [r for r in br_fb if abs(r[0]) < 1e-9]
    ev_nom = nom[0][3] if nom else np.array([np.nan, np.nan])
    summary['eig_nom'] = [complex(x) for x in ev_nom]; summary['delta_eq_nom'] = float(nom[0][1]) if nom else None
    dg = np.linspace(-0.6, 2.6, 64); wg = np.linspace(-2.2, 2.2, 64)
    bs_fb = basin_grid(dg, wg, UFF, D_REF, 20.0, 0.02, True, 0.0)
    bs_ol = basin_grid(dg, wg, UFF, D_REF, 20.0, 0.02, False, 0.0)
    deq = summary['delta_eq_nom']
    def ok(b): return (np.abs(b[..., 0] - deq) < 0.05) & (np.abs(b[..., 1]) < 0.05)
    frac_fb, frac_ol = float(np.mean(ok(bs_fb))), float(np.mean(ok(bs_ol)))
    csv('scalar_basin_fraction.csv', ['loop', 'fraction_converged_to_nominal_equilibrium'], [('feedback', frac_fb), ('open_loop', frac_ol)])
    summary['basin'] = (frac_fb, frac_ol)
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for rows_, lab, c_, ls_ in [(br_ol, 'open loop', '#7f7f7f', '-'), (br_fb, 'feedback', COL['A3'], '-')]:
        pp = np.array([r[0] for r in rows_]); xx = np.array([r[1] for r in rows_]); st_ = np.array([r[2] < 0 for r in rows_])
        ax[0].plot(pp[st_], xx[st_], '.', ms=2, color=c_, label=lab + ' (stable)'); ax[0].plot(pp[~st_], xx[~st_], 'x', ms=2, color=c_, alpha=.5)
    ax[0].set_xlabel('constant force bias $p$'); ax[0].set_ylabel('equilibrium $\\delta^*$'); ax[0].set_title('(a) equilibrium continuation'); ax[0].legend(fontsize=6)
    ax[1].imshow(ok(bs_fb).T, origin='lower', extent=[dg[0], dg[-1], wg[0], wg[-1]], aspect='auto', cmap='Greens', alpha=.8, vmin=0, vmax=1)
    ax[1].plot(deq, 0, 'k*', ms=6); ax[1].set_xlabel('$\\delta_0$'); ax[1].set_ylabel('$\\omega_0$'); ax[1].set_title(f'(b) basin, feedback ({100*frac_fb:.0f}%)')
    ax[2].imshow(ok(bs_ol).T, origin='lower', extent=[dg[0], dg[-1], wg[0], wg[-1]], aspect='auto', cmap='Greens', alpha=.8, vmin=0, vmax=1)
    ax[2].set_xlabel('$\\delta_0$'); ax[2].set_title(f'(c) basin, open loop ({100*frac_ol:.0f}%)')
    fig.tight_layout(); save(fig, 'fig_scalar_nonlinear')
    json.dump({k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in summary.items() if k != 'eig_nom'} | {'eig_nom': [[c.real, c.imag] for c in summary['eig_nom']]},
              open(RES / 'scalar_summary.json', 'w'), indent=1, default=float)
    print('scalar total time', time.time() - t_all)


if __name__ == '__main__':
    main()
