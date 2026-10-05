"""Network-benchmark experiments N1-N5 for MATPOWER case9 and case39 (same pipeline as the scalar study)."""
import sys, time, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common import *
from network_benchmark import *
from network_gradient import mpc_cost_net_analytic

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / 'results', ROOT / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 120,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
COL = {'C0': '#9467bd', 'A0': '#7f7f7f', 'A1': '#1f77b4', 'A2': '#2ca02c', 'A3': '#d62728'}
TS, NSUB = 0.1, 2
STRESS = {'case9': 2.75, 'case39': 2.25}
GAMMA = np.radians(30.0)
MODES = {'A0': (0, 0), 'A1': (0, 1), 'A2': (1, 1), 'A3': (1, 1)}


def csv(name, header, rows):
    with open(RES / name, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'{x:.8g}' if isinstance(x, (float, np.floating)) else str(x) for x in r) + '\n')


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def args_of(c):
    return (c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'])


def plant_kwargs(c):
    return dict(lf=c['lf'], lt=c['lt'], lb=c['lb'], gbus=c['gbus'], bg=c['bg'], M=c['M'], Dm=c['D'], Pm=c['Pm'], kappa=c['kappa'], fric=c['fric'])


def simulate_plant(c, th0, de0, w0, controller, T, load, hp=0.0125, Ts=TS, seed=0, record_every=1):
    """Closed-loop true plant. controller(t, delta, w, theta, u_prev, Pd_now) -> u. load=(amp, omega, phase)."""
    lamp, lom, lph = load
    ng = c['ng']; nsteps = int(round(T / Ts)); nsub = int(round(Ts / hp))
    de, w, th = de0.copy(), w0.copy(), th0.copy()
    u_prev = np.zeros(ng); t = 0.0
    logs = dict(t=[], de=[], w=[], th=[], u=[], solve=[])
    for k in range(nsteps):
        Pd_now = c['Pd'] * (1 + lamp * np.sin(lom * t + lph))
        t0 = time.perf_counter()
        u = controller(t, de, w, th, u_prev, Pd_now)
        logs['solve'].append(time.perf_counter() - t0)
        for key, val in (('t', t), ('de', de.copy()), ('w', w.copy()), ('th', th.copy()), ('u', u.copy())):
            logs[key].append(val)
        for _ in range(nsub):
            de, w, th = plant_step(de, w, th, u, hp, t, c['Pd'], lamp, lom, lph, c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['D'], c['Pm'], c['kappa'], c['fric'])
            t += hp
        u_prev = u
    for key, val in (('t', t), ('de', de.copy()), ('w', w.copy()), ('th', th.copy()), ('u', u_prev.copy())):
        logs[key].append(val)
    return {k: (np.array(v) if k != 'solve' else float(np.mean(v))) for k, v in logs.items()}


def static_ctrl(c):
    kgain = 1.0 * c['D']
    return lambda t, de, w, th, up, Pd: np.clip(-kgain * w, -c['umax'], c['umax'])


# ----------------------------------------------------------------------------- identification
def make_training(c, th_eq, de_eq, n_traj, seed, noise, h=0.025, T=3.0, ic=0.20, wic=0.8, lamp=0.04):
    rng = np.random.default_rng(seed)
    out = []
    kg = static_ctrl(c)
    for _ in range(n_traj):
        de0 = de_eq + rng.normal(0, ic, c['ng']); w0 = rng.normal(0, wic, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        load = (rng.uniform(0, lamp), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        dither_state = {'u': np.zeros(c['ng']), 'k': 0}
        def ctrl(t, de, w, th, up, Pd):
            return np.clip(-c['D'] * w + rng.normal(0, 0.35, c['ng']) * c['umax'], -c['umax'], c['umax'])
        lg = simulate_plant(c, th0, de0, w0, ctrl, T, load, hp=h / 2, Ts=h)
        de_m = lg['de'] + rng.normal(0, noise, lg['de'].shape); w_m = lg['w'] + rng.normal(0, noise, lg['w'].shape)
        Pd_series = np.array([c['Pd'] * (1 + load[0] * np.sin(load[1] * tt + load[2])) for tt in lg['t']])
        out.append(dict(de=de_m, w=w_m, U=lg['u'][:-1], Pd=Pd_series, h=h, de_true=lg['de'], w_true=lg['w'], th_true=lg['th']))
    return out


def weak_form_net(c, tr, th_eq):
    de, w, U, Pd, h = tr['de'], tr['w'], tr['U'], tr['Pd'], tr['h']
    K = U.shape[0]
    th = np.zeros((K + 1, c['n'])); guess = th_eq
    for k in range(K + 1):
        guess, _ = solve_theta(guess, de[k], Pd[k], *args_of(c), 1e-11, 30); th[k] = guess
    ap = np.array([swing_acc(de[k], w[k], U[min(k, K - 1)], th[k], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric']) for k in range(K + 1)])
    Y = np.zeros((K, c['ng'])); TH = np.zeros((K, c['ng'], NF))
    for k in range(K):
        ap_k = swing_acc(de[k], w[k], U[k], th[k], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
        ap_k1 = swing_acc(de[k + 1], w[k + 1], U[k], th[k + 1], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
        Y[k] = (w[k + 1] - w[k]) / h - 0.5 * (ap_k + ap_k1)
        for i in range(c['ng']):
            TH[k, i] = 0.5 * (feats(de[k, i] - th[k, c['gbus'][i]], w[k, i], U[k, i]) + feats(de[k + 1, i] - th[k + 1, c['gbus'][i]], w[k + 1, i], U[k, i]))
    return TH, Y


def fit_net(c, train, val, th_eq, ridges=(1e-6, 1e-4, 1e-2), threshes=(0.002, 0.01, 0.03, 0.1, 0.3)):
    Tr = [weak_form_net(c, t, th_eq) for t in train]; Va = [weak_form_net(c, t, th_eq) for t in val]
    TH = np.concatenate([a for a, _ in Tr]); Y = np.concatenate([b for _, b in Tr])
    THv = np.concatenate([a for a, _ in Va]); Yv = np.concatenate([b for _, b in Va])
    coef = np.zeros((c['ng'], NF + 1)); info = []
    for i in range(c['ng']):
        cands = []
        for rg in ridges:
            for tq in threshes:
                cc = stlsq(TH[:, i, :], Y[:, i], ridge=rg, thresh=tq)
                e = np.sqrt(np.mean((Yv[:, i] - THv[:, i, :] @ cc) ** 2))
                cands.append((e, int(np.sum(cc != 0)), cc))
        ebest = min(x[0] for x in cands)
        e, _, cc = min([x for x in cands if x[0] <= 1.05 * ebest], key=lambda x: (x[1], x[0]))      # sparsest within 5% of best
        coef[i, :NF] = cc; coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i])); info.append(e)
    cal = np.concatenate([Yv[:, i] - THv[:, i, :] @ coef[i, :NF] for i in range(c['ng'])])
    return coef, float(np.mean(info)), cal


def exact_resid_stats(c, trs, coef):
    ex, er = [], []
    for tr in trs:
        for k in range(tr['U'].shape[0]):
            de, w, th = tr['de_true'][k], tr['w_true'][k], tr['th_true'][k]
            u = tr['U'][k]
            at = swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['D'], c['Pm'], True, c['kappa'], c['fric'])
            ap = swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
            rh = res_acc(de, w, u, th, c['gbus'], coef)
            ex.append(at - ap); er.append(at - ap - rh)
    ex = np.concatenate(ex); er = np.concatenate(er)
    return float(np.sqrt(np.mean(ex ** 2))), float(np.sqrt(np.mean(er ** 2))), float(np.max(np.abs(er)))


# ----------------------------------------------------------------------------- prediction ablation
def make_test_runs(c, th_eq, de_eq, n, seed, lamp_hi, T=6.0, lamp_lo=0.0):
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        de0 = de_eq + rng.normal(0, 0.10, c['ng']); w0 = rng.normal(0, 0.3, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        load = (rng.uniform(lamp_lo, lamp_hi), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        lg = simulate_plant(c, th0, de0, w0, static_ctrl(c), T, load)
        runs.append((load, lg))
    return runs


def predict_windows(c, runs, name, coef, N=10, stride=4, Pd_fn=None):
    mode, use_res = MODES[name]
    err_state = [[] for _ in range(N)]; gmax = []; angerr = [[] for _ in range(N)]
    for load, lg in runs:
        K = len(lg['t']) - 1
        for k in range(0, K - N, stride):
            Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
            gm = 0.0
            for j in range(N):
                u = lg['u'][k + j]
                for _ in range(NSUB):
                    de, w, th = pred_step(de, w, th, u, TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
                coi_p = np.sum(c['M'] * de) / np.sum(c['M']); de_t = lg['de'][k + j + 1]; coi_t = np.sum(c['M'] * de_t) / np.sum(c['M'])
                e = np.r_[(de - coi_p) - (de_t - coi_t), w - lg['w'][k + j + 1]]
                err_state[j].append(e)
                g, _ = alg_res_jac(th, de, Pd_k, *args_of(c)[:3], c['gbus'], c['bg'], False)
                gm = max(gm, float(np.max(np.abs(g))))
                angerr[j].append(np.max(np.abs(th[c['lf']] - th[c['lt']])) - np.max(np.abs(lg['th'][k + j + 1][c['lf']] - lg['th'][k + j + 1][c['lt']])))
            gmax.append(gm)
    return [np.array(e) for e in err_state], np.array(gmax), [np.array(a) for a in angerr]


def envelope_traj(c, runs, coef, N=5):
    """Trajectory-level envelope input: for every run, max over windows of |max line-angle prediction error| at step k."""
    out = []
    for r in runs:
        _, _, ang = predict_windows(c, [r], 'A2', coef, N=N)
        out.append([np.max(np.abs(a)) for a in ang[:N]])
    return np.array(out)


def envelope_from(pm):
    return np.r_[0.0, [conformal_quantile(pm[:, j], 0.05) for j in range(pm.shape[1])]]


def mpc_log_runs(c, th_eq, de_eq, coef, rel_ref, n, seed, T, lamp_hi, gam=GAMMA, lamp_lo=0.0):
    """Closed-loop A2 (no tightening) trajectories, used as policy-matched calibration/test data."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        load = (rng.uniform(lamp_lo, lamp_hi), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        de0 = de_eq + rng.normal(0, 0.05, c['ng']); w0 = rng.normal(0, 0.15, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        ctrl = make_net_mpc(c, 'A2', coef, rel_ref, np.zeros(6), gam=gam)
        out.append((load, simulate_plant(c, th0, de0, w0, ctrl, T, load)))
    return out


# ----------------------------------------------------------------------------- MPC
@njit
def mpc_cost_net(Uf, de0, w0, th0, uprev, N, Ts, nsub, mode, use_res, coef, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric,
                 rel_ref, umax, qd, qw, r, rd, gam, eps, pen):
    ng = de0.shape[0]
    de, w, th = de0.copy(), w0.copy(), th0.copy()
    J = 0.0
    h = Ts / nsub
    Msum = np.sum(M)
    for k in range(N):
        u = np.empty(ng); up = np.empty(ng)
        for i in range(ng):
            u[i] = Uf[k * ng + i]
            up[i] = Uf[(k - 1) * ng + i] if k > 0 else uprev[i]
        for s in range(nsub):
            de, w, th = pred_step(de, w, th, u, h, Pd, mode, use_res, coef, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric)
        coi = np.sum(M * de) / Msum
        for i in range(ng):
            rel = (de[i] - coi) - rel_ref[i]
            J += qd * rel * rel + qw * w[i] * w[i] + r * (u[i] / umax[i]) ** 2 + rd * ((u[i] - up[i]) / umax[i]) ** 2
        lim = gam - eps[k + 1]
        for l in range(lf.shape[0]):
            v = abs(th[lf[l]] - th[lt[l]]) - lim
            if v > 0:
                J += pen * v * v
        for i in range(ng):
            v = abs(de[i] - th[gbus[i]]) - (lim + 0.35)
            if v > 0:
                J += pen * v * v
    return J


@njit
def mpc_cost_net_grad(Uf, de0, w0, th0, uprev, N, Ts, nsub, mode, use_res, coef, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric,
                      rel_ref, umax, qd, qw, r, rd, gam, eps, pen):
    J0 = mpc_cost_net(Uf, de0, w0, th0, uprev, N, Ts, nsub, mode, use_res, coef, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric,
                      rel_ref, umax, qd, qw, r, rd, gam, eps, pen)
    g = np.zeros(Uf.shape[0])
    for j in range(Uf.shape[0]):
        Up = Uf.copy(); Up[j] += 1e-5
        g[j] = (mpc_cost_net(Up, de0, w0, th0, uprev, N, Ts, nsub, mode, use_res, coef, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric,
                             rel_ref, umax, qd, qw, r, rd, gam, eps, pen) - J0) / 1e-5
    return J0, g


def make_net_mpc(c, name, coef, rel_ref, eps, N=5, maxiter=12, gam=GAMMA, qd=0.5, qw=20.0, r=0.02, rd=0.2, pen=3e3, analytic=False):
    mode, use_res = MODES[name]
    cf = coef if use_res else np.zeros((c['ng'], NF + 1))
    e = eps if name == 'A3' else np.zeros(N + 1)
    gm = gam if name == 'A3' else gam      # same physical limit for all; only A3 tightens it by the calibrated envelope
    st = {'U': np.zeros(N * c['ng'])}
    umax = c['umax']; ng = c['ng']
    bounds = [(-umax[i], umax[i]) for _ in range(N) for i in range(ng)]

    def ctrl(t, de, w, th, up, Pd):
        U0 = np.r_[st['U'][ng:], st['U'][-ng:]]
        a = (de, w, th, up, N, TS, NSUB, mode, use_res, cf, Pd, c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'],
             rel_ref, umax, qd, qw, r, rd, gm, e, pen)
        cost_grad = mpc_cost_net_analytic if (analytic and mode == 1) else mpc_cost_net_grad
        res = minimize(lambda U: cost_grad(U, *a), U0, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=maxiter))
        st['U'] = res.x
        return res.x[:ng].copy()
    return ctrl


def net_metrics(c, lg):
    th, de, w = lg['th'], lg['de'], lg['w']
    ang = np.array([np.max(np.abs(t[c['lf']] - t[c['lt']])) for t in th])
    lam = np.array([regularity(c, th[k], de[k], Pd=c['Pd'])[0] for k in range(0, len(th), 2)])
    w_rms = float(np.sqrt(np.mean(w[len(w) // 3:] ** 2)))
    return dict(angle_peak_deg=float(np.degrees(ang.max())), viol_integral=float(np.sum(np.maximum(0, ang - GAMMA)) * TS), w_rms=w_rms,
                u_energy=float(np.sum((lg['u'] / c['umax']) ** 2) * TS), lam_min=float(lam.min()), solve_ms=lg['solve'] * 1e3)


def run_case(name, quick=False):
    t_case = time.time(); out = {}
    c1 = build_case(name); th1, de1, res1 = equilibrium(c1)
    # ---- N1: regularity margin, spectrum, loadability continuation
    ev = swing_jacobian_eigs(c1, th1, de1); ev = ev[np.abs(ev) > 1e-6]
    zeta = -ev.real / np.abs(ev); osc = ev[ev.imag > 1e-9]
    mus = np.round(np.arange(1.0, 4.51, 0.125), 3)
    cont = loadability_continuation(name, mus)
    csv(f'{name}_loadability.csv', ['load_scale', 'lambda_min_J', 'cond_J', 'max_line_angle_deg', 'max_gen_angle_deg', 'residual'], cont)
    lam0, cond0, amax0, gmax0 = regularity(c1, th1, de1)
    mu_lim = max([r[0] for r in cont if r[1] == r[1]])
    # Prop. (certificate): lambda_min(J) >= cos(gamma)*lambda_min(L_b + G) for all angle differences <= gamma
    _, Jnom = alg_res_jac(np.zeros(c1['n']), np.zeros(c1['ng']), c1['Pd'], *args_of(c1)[:3], c1['gbus'], c1['bg'], True)
    lam_cert = float(np.linalg.eigvalsh(Jnom).min())
    csv(f'{name}_spectrum.csv', ['quantity', 'value'], [
        ('buses', c1['n']), ('generators', c1['ng']), ('equilibrium_residual', res1), ('lambda_min_J_nominal', lam0), ('cond_J_nominal', cond0),
        ('max_line_angle_deg_nominal', amax0), ('largest_continued_load_scale', mu_lim), ('lambda_min_Lb_plus_G', lam_cert),
        ('cert_lower_bound_at_30deg', np.cos(GAMMA) * lam_cert), ('slowest_damping_ratio', float(np.min(zeta[ev.imag > 1e-9])) if len(osc) else np.nan),
        ('max_real_part_nonneutral', float(ev.real.max())), ('lowest_mode_freq_hz', float(osc.imag.min() / 2 / np.pi)), ('highest_mode_freq_hz', float(osc.imag.max() / 2 / np.pi))])
    out['spectrum'] = ev; out['cont'] = cont
    # ---- identification at nominal loading and at the stress loading used for control
    result = {}
    for tag, mu in (('nom', 1.0), ('stress', STRESS[name])):
        c = build_case(name, load_scale=mu); th_eq, de_eq, _ = equilibrium(c)
        ntr = (14, 4, 8) if quick else (30, 8, 16)
        train = make_training(c, th_eq, de_eq, ntr[0], 1001, 1e-3); val = make_training(c, th_eq, de_eq, ntr[1], 1002, 1e-3); test = make_training(c, th_eq, de_eq, ntr[2], 1003, 0.0)
        coef, val_rmse, _ = fit_net(c, train, val, th_eq)
        calib = make_training(c, th_eq, de_eq, 8, 1004, 1e-3)           # independent calibration set
        cal = np.concatenate([(lambda TH, Y: np.concatenate([Y[:, i] - TH[:, i, :] @ coef[i, :NF] for i in range(c['ng'])]))(*weak_form_net(c, t, th_eq)) for t in calib])
        rms_ex, rms_err, max_err = exact_resid_stats(c, test, coef)
        rho = conformal_quantile(np.abs(cal), 0.05)
        csv(f'{name}_{tag}_residual_fit.csv', ['quantity', 'value'], [
            ('load_scale', mu), ('train_traj', ntr[0]), ('residual_rms_true_test', rms_ex), ('residual_rms_error_after_fit', rms_err),
            ('error_reduction_percent', 100 * (1 - rms_err / rms_ex)), ('max_abs_error_test', max_err), ('rho_pointwise_95_measured', rho),
            ('active_terms_total', int(np.sum(coef[:, :NF] != 0))), ('terms_available', int(coef[:, :NF].size))])
        result[tag] = dict(c=c, th=th_eq, de=de_eq, coef=coef, rms_ex=rms_ex, rms_err=rms_err, rho=rho)
        print(name, tag, 'resid rms', rms_ex, '-> err', rms_err, 'active', int(np.sum(coef[:, :NF] != 0)), '/', coef[:, :NF].size, flush=True)
    # ---- N3: prediction ablation (nominal loading, load fluctuation up to 0.08, i.e. 2x the training maximum)
    R = result['nom']; c = R['c']
    nrun = 6 if quick else 12
    runs = make_test_runs(c, R['th'], R['de'], nrun, 2001, 0.08)
    cal_runs = make_test_runs(c, R['th'], R['de'], 6 if quick else 10, 2002, 0.08)
    abl = {}
    for nm in ('A0', 'A1', 'A2'):
        abl[nm] = predict_windows(c, runs, nm, R['coef'])
    rows = []
    for nm in abl:
        for j in range(10):
            e = abl[nm][0][j]
            rows.append((nm, (j + 1) * TS, float(np.sqrt(np.mean(np.sum(e ** 2, axis=1) / c['ng']))), float(np.mean(np.abs(e[:, :c['ng']]))), float(np.mean(np.abs(e[:, c['ng']:])))))
    csv(f'{name}_prediction_ablation.csv', ['model', 'horizon_s', 'state_rmse', 'mean_abs_delta_err', 'mean_abs_omega_err'], rows)
    csv(f'{name}_algebraic_residual.csv', ['model', 'mean_window_max_abs_g', 'max_abs_g'], [(nm, float(np.mean(abl[nm][1])), float(np.max(abl[nm][1]))) for nm in abl])
    # paired per-run ratio
    ratios = []
    for ri in range(len(runs)):
        e0 = predict_windows(c, [runs[ri]], 'A0', R['coef'])[0]; e2 = predict_windows(c, [runs[ri]], 'A2', R['coef'])[0]
        ratios.append(np.sqrt(np.mean(np.vstack(e2) ** 2) / np.mean(np.vstack(e0) ** 2)))
    ratios = np.array(ratios)
    ci = bootstrap_ci(ratios); out['pred_ratio'] = ci
    csv(f'{name}_prediction_paired.csv', ['run', 'rmse_ratio_A2_over_A0'], [(i, float(r)) for i, r in enumerate(ratios)])
    # calibrated k-step envelope for the maximum line-angle (used by A3)
    eps_net = envelope_from(envelope_traj(c, cal_runs, R['coef']))          # trajectory-level, static-law calibration (nominal loading)
    csv(f'{name}_tube_envelope.csv', ['k', 'eps_line_angle_rad'], [(k, float(eps_net[k])) for k in range(6)])
    print(name, 'N3 ratio', ci, 'g', [(nm, float(np.mean(abl[nm][1]))) for nm in abl], 'eps', eps_net, flush=True)
    out['abl'] = abl; out['eps'] = eps_net
    # ---- N5: sensitivity (plant parameters change, predictors keep the nominal parameters) and Monte Carlo
    sens_rows = []
    scales = (0.8, 1.0, 1.25) if quick else (0.6, 0.8, 1.0, 1.25, 1.5)
    for pname in ('load_scale', 'line_scale', 'inertia_scale', 'damping_scale'):
        for sc in scales:
            try:
                cp = build_case(name, **{pname: sc})
                if pname == 'load_scale':
                    thp, dep, rp = equilibrium(cp)
                    if rp > 1e-8: continue
                else:
                    thp, dep, rp = equilibrium(cp)
                runs_p = make_test_runs(cp, thp, dep, 3 if quick else 5, 3001, 0.06)
                # predictor sees nominal-parameter model: swap plant-side arrays only
                def pw(nm):
                    m, ur = MODES[nm]; E_ = []
                    for load, lg in runs_p:
                        K = len(lg['t']) - 1
                        for k in range(0, K - 10, 6):
                            Pd_k = c['Pd'] * (cp['Pd'].sum() / c['Pd'].sum()) * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2])) if False else cp['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
                            de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
                            for j in range(10):
                                for _ in range(NSUB):
                                    de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, m, ur, R['coef'], *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
                            coi_p = np.sum(c['M'] * de) / np.sum(c['M']); de_t = lg['de'][k + 10]; coi_t = np.sum(c['M'] * de_t) / np.sum(c['M'])
                            E_.append(np.sum(((de - coi_p) - (de_t - coi_t)) ** 2) + np.sum((w - lg['w'][k + 10]) ** 2))
                    return float(np.sqrt(np.mean(E_) / c['ng']))
                e0, e2 = pw('A0'), pw('A2')
                if not (np.isfinite(e0) and np.isfinite(e2)): raise ValueError
                sens_rows.append((pname, sc, e0, e2, e2 / e0))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                sens_rows.append((pname, sc, float('nan'), float('nan'), float('nan')))   # plant leaves the regular region
    csv(f'{name}_sensitivity.csv', ['parameter', 'scale', 'rmse_A0_1s', 'rmse_A2_1s', 'ratio'], sens_rows)
    out['sens'] = sens_rows
    # Monte Carlo ensemble (parameters drawn jointly; 24 realisations unless quick)
    rng = np.random.default_rng(4001); mc = []
    for i in range(8 if quick else 24):
        kw = dict(load_scale=rng.uniform(0.8, 1.2), line_scale=rng.uniform(0.85, 1.15), inertia_scale=rng.uniform(0.8, 1.2), damping_scale=rng.uniform(0.8, 1.2))
        cp = build_case(name, **kw); thp, dep, rp = equilibrium(cp)
        if rp > 1e-8: continue
        try:
            run_p = make_test_runs(cp, thp, dep, 1, 5000 + i, rng.uniform(0.0, 0.08))
            r0 = predict_windows(cp, run_p, 'A0', R['coef'], stride=6)[0]; r2 = predict_windows(cp, run_p, 'A2', R['coef'], stride=6)[0]
            v0, v2 = np.sqrt(np.mean(np.vstack(r0) ** 2)), np.sqrt(np.mean(np.vstack(r2) ** 2))
            if np.isfinite(v0) and np.isfinite(v2): mc.append((v0, v2))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
    mc = np.array(mc)
    csv(f'{name}_montecarlo.csv', ['realisation', 'rmse_A0', 'rmse_A2', 'ratio'], [(i, a, b, b / a) for i, (a, b) in enumerate(mc)])
    out['mc'] = (bootstrap_ci(mc[:, 1] / mc[:, 0]), bootstrap_ci(mc[:, 0]), bootstrap_ci(mc[:, 1]), len(mc))
    print(name, 'MC', out['mc'], flush=True)
    # ---- N4: closed loop under stress loading
    S_ = result['stress']; cs = S_['c']
    rel_ref = (S_['de'] - np.sum(cs['M'] * S_['de']) / np.sum(cs['M']))
    nreal = 2 if quick else (6 if name == 'case9' else 4)
    Tcl = 4.0 if quick else (6.0 if name == 'case9' else 5.0)
    # policy-matched, trajectory-level envelope at the elevated loading: static-law and MPC calibration trajectories
    lamp_c = 0.11
    cal_static = make_test_runs(cs, S_['th'], S_['de'], 4, 7000, lamp_c, T=Tcl, lamp_lo=0.07)
    cal_mpc = mpc_log_runs(cs, S_['th'], S_['de'], S_['coef'], rel_ref, 3, 7010, Tcl, lamp_c, lamp_lo=0.07)
    eps_s = envelope_from(envelope_traj(cs, cal_static + cal_mpc, S_['coef']))
    csv(f'{name}_tube_envelope_stress.csv', ['k', 'eps_line_angle_rad'], [(k, float(eps_s[k])) for k in range(6)])
    test_static = make_test_runs(cs, S_['th'], S_['de'], 4, 7020, lamp_c, T=Tcl, lamp_lo=0.07)
    test_mpc = mpc_log_runs(cs, S_['th'], S_['de'], S_['coef'], rel_ref, 3, 7030, Tcl, lamp_c, lamp_lo=0.07)
    cov_rows = []
    for src, rr in (('static law', test_static), ('MPC (A2)', test_mpc)):
        pm_ = envelope_traj(cs, rr, S_['coef']); cov_rows.append((src, len(rr), float(np.mean(pm_[:, 4] <= eps_s[5])), float(np.mean(np.all(pm_ <= eps_s[1:][None, :], axis=1)))))
    rng = np.random.default_rng(6001); cl = {n: [] for n in ('C0', 'A0', 'A2', 'A3')}; cl_logs = {}; a2_logs = []
    for i in range(nreal):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
        de0 = S_['de'] + rng.normal(0, 0.03, cs['ng']); w0 = rng.normal(0, 0.1, cs['ng'])
        th0, _ = solve_theta(S_['th'], de0, cs['Pd'], *args_of(cs), 1e-12, 30)
        for nm in cl:
            ctrl = static_ctrl(cs) if nm == 'C0' else make_net_mpc(cs, nm, S_['coef'], rel_ref, eps_s)
            lg = simulate_plant(cs, th0, de0, w0, ctrl, Tcl, load)
            cl[nm].append(net_metrics(cs, lg))
            if i == 0: cl_logs[nm] = lg
            if nm == 'A2': a2_logs.append((load, lg))
        print(name, 'CL run', i, {n: (round(cl[n][-1]['angle_peak_deg'], 2), round(cl[n][-1]['w_rms'], 4), round(cl[n][-1]['solve_ms'], 1)) for n in cl}, flush=True)
    pm_ = envelope_traj(cs, a2_logs, S_['coef']); cov_rows.append(('closed-loop A2 realizations', len(a2_logs), float(np.mean(pm_[:, 4] <= eps_s[5])), float(np.mean(np.all(pm_ <= eps_s[1:][None, :], axis=1)))))
    csv(f'{name}_envelope_coverage.csv', ['test_source', 'n_traj', 'cover_k5', 'cover_joint_k1_to_5'], cov_rows)
    print(name, 'envelope coverage', cov_rows, 'eps', eps_s, flush=True)
    keys = ['angle_peak_deg', 'viol_integral', 'w_rms', 'u_energy', 'lam_min', 'solve_ms']
    csv(f'{name}_closed_loop.csv', ['controller'] + [f'{k}_mean' for k in keys], [(n,) + tuple(float(np.mean([m[k] for m in cl[n]])) for k in keys) for n in cl])
    csv(f'{name}_closed_loop_runs.csv', ['run', 'controller'] + keys, [(i, n) + tuple(cl[n][i][k] for k in keys) for n in cl for i in range(nreal)])
    out['cl'] = cl; out['cl_logs'] = cl_logs; out['cs'] = cs
    print(name, 'time', time.time() - t_case, flush=True)
    return out


def figures(res):
    # Fig: regularity / spectrum / loadability
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for nm, col in (('case9', '#1f77b4'), ('case39', '#d62728')):
        cont = res[nm]['cont']; mu = [r[0] for r in cont if r[1] == r[1]]; lam = [r[1] for r in cont if r[1] == r[1]]; ang = [r[3] for r in cont if r[1] == r[1]]
        ax[0].plot(mu, np.array(lam) / lam[0], '-o', ms=2, color=col, label=nm); ax[1].plot(mu, ang, '-o', ms=2, color=col, label=nm)
        ev = res[nm]['spectrum']; ax[2].plot(ev.real, ev.imag / 2 / np.pi, '.', ms=4, color=col, label=nm)
    ax[0].set_xlabel('load scale $\\mu$'); ax[0].set_ylabel('$\\lambda_{\\min}(J_g)/\\lambda_{\\min}^{(\\mu=1)}$'); ax[0].set_title('(a) regularity margin'); ax[0].legend()
    ax[1].axhline(30, color='k', ls='--', lw=.7); ax[1].set_xlabel('load scale $\\mu$'); ax[1].set_ylabel('max line angle [deg]'); ax[1].set_title('(b) line-angle growth')
    ax[2].set_xlabel('Re $\\lambda$ [1/s]'); ax[2].set_ylabel('Im $\\lambda/2\\pi$ [Hz]'); ax[2].set_title('(c) linearised spectrum, $\\mu=1$'); ax[2].legend()
    fig.tight_layout(); save(fig, 'fig_network_regularity')
    # Fig: prediction ablation
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for k, nm in enumerate(('case9', 'case39')):
        abl = res[nm]['abl']; hz = (np.arange(10) + 1) * TS
        for mname in ('A0', 'A1', 'A2'):
            ax[k].plot(hz, [np.sqrt(np.mean(np.sum(abl[mname][0][j] ** 2, axis=1) / (3 if nm == 'case9' else 10))) for j in range(10)], '-o', ms=2.5, color=COL[mname], label=mname)
        ax[k].set_yscale('log'); ax[k].set_xlabel('horizon [s]'); ax[k].set_ylabel('state RMSE'); ax[k].set_title(f'({"ab"[k]}) {nm}'); ax[k].legend(ncol=3, fontsize=6)
    mcs = [(nm, res[nm]['mc']) for nm in ('case9', 'case39')]
    ax[2].bar([0, 1], [m[1][0][0] for m in mcs], color=['#1f77b4', '#d62728'], yerr=[[m[1][0][0] - m[1][0][1] for m in mcs], [m[1][0][2] - m[1][0][0] for m in mcs]], capsize=3)
    ax[2].axhline(1, color='k', ls=':', lw=.8); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(['case9', 'case39']); ax[2].set_ylabel('RMSE ratio A2/A0'); ax[2].set_title('(c) Monte Carlo, 95% CI')
    fig.tight_layout(); save(fig, 'fig_network_ablation')
    # Fig: closed loop traces
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.4))
    for k, nm in enumerate(('case9', 'case39')):
        cs = res[nm]['cs']
        for n, lg in res[nm]['cl_logs'].items():
            ang = np.degrees([np.max(np.abs(t[cs['lf']] - t[cs['lt']])) for t in lg['th']])
            ax[0, k].plot(lg['t'], ang, color=COL[n if n != 'C0' else 'C0'], lw=1.0, label=n); ax[1, k].plot(lg['t'], np.max(np.abs(lg['w']), axis=1), color=COL[n], lw=1.0)
        ax[0, k].axhline(30, color='k', ls='--', lw=.7); ax[0, k].set_ylabel('max line angle [deg]'); ax[0, k].set_title(f'({"ab"[k]}) {nm}, $\\mu$={STRESS[nm]}'); ax[0, k].legend(ncol=4, fontsize=6)
        ax[1, k].set_ylabel('max $|\\omega_i|$ [rad/s]'); ax[1, k].set_xlabel('time [s]')
    fig.tight_layout(); save(fig, 'fig_network_closedloop')


if __name__ == '__main__':
    quick = len(sys.argv) > 1 and sys.argv[1] == 'quick'
    results = {}
    for nm in (['case9'] if (len(sys.argv) > 2 and sys.argv[2] == 'case9only') else ['case9', 'case39']):
        results[nm] = run_case(nm, quick)
    if len(results) == 2:
        figures(results)
    summ = {nm: dict(pred_ratio=list(map(float, r['pred_ratio'])), mc_ratio=list(map(float, r['mc'][0])), mc_n=int(r['mc'][3])) for nm, r in results.items()}
    json.dump(summ, open(RES / ('network_summary_quick.json' if quick else 'network_summary.json'), 'w'), indent=1)
    print('done')
