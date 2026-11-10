"""Structure-preserving nonlinear descriptor realisation of MATPOWER case9 / case39.

Differential variables: generator internal angle delta_i and speed deviation omega_i (classical swing model).
Algebraic variables: all bus voltage angles theta_j (lossless P-theta network, unit voltages):
    g_j = sum_{i at j} bg_i sin(delta_i - theta_j) - Pd_j - sum_{l~j} b_l sin(theta_j - theta_k) = 0
    M_i omega_i' = Pm_i + u_i - D_i omega_i - Pe_i,    Pe_i = bg_i sin(delta_i - theta_{bus(i)})
Topology, loads and dispatch come from the official MATPOWER files in ../data.  Inertia, damping and the
generator reactance are NOT in MATPOWER and are declared assumptions (see build_case).
The unmodelled plant effects are generator saliency, a saturating friction term and a load fluctuation.
"""
import re
from pathlib import Path
import numpy as np
from scipy.optimize import root
from common import njit, conformal_quantile

DATA = Path(__file__).resolve().parents[1] / 'data'
FS = 2 * np.pi * 60.0


def parse_matpower(name):
    s = (DATA / f'{name}.m').read_text()
    out = {}
    m = re.search(r'mpc\.baseMVA\s*=\s*([\d.]+);', s); out['baseMVA'] = float(m.group(1))
    for key in ('bus', 'gen', 'branch'):
        m = re.search(r'mpc\.%s\s*=\s*\[(.*?)\];' % key, s, re.S)
        txt = re.sub(r'%.*', '', m.group(1)).replace(';', '\n')
        out[key] = np.array([[float(x) for x in r.split()] for r in txt.split('\n') if r.strip()])
    return out


def build_case(name, load_scale=1.0, line_scale=1.0, inertia_scale=1.0, damping_scale=1.0, H=4.0, xd_pu=0.3, zeta_rate=2.0,
               saliency=0.08, fric=0.15, phys_damp=0.8, ool=0.0):
    mp = parse_matpower(name)
    base = mp['baseMVA']; bus, gen, br = mp['bus'], mp['gen'], mp['branch']
    n = len(bus); ng = len(gen)
    idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    Pd = bus[:, 2] / base * load_scale
    gbus = np.array([idx[int(b)] for b in gen[:, 0]], dtype=np.int64)
    Pmax = gen[:, 8] / base
    Pg = gen[:, 1] / base
    Pm = Pg * Pd.sum() / Pg.sum()                                  # lossless re-dispatch so that sum Pm = sum Pd
    lf = np.array([idx[int(b)] for b in br[:, 0]], dtype=np.int64); lt = np.array([idx[int(b)] for b in br[:, 1]], dtype=np.int64)
    lb = 1.0 / br[:, 3] / line_scale
    Sg = Pmax                                                       # rating proxy [pu on system base]
    bg = Sg / xd_pu                                                 # 1/x'd on system base, x'd = xd_pu on unit base
    M = 2 * H * Sg / FS * inertia_scale
    D = zeta_rate * 2 * M * damping_scale                           # decay rate ~ zeta_rate [1/s]
    umax = 0.10 * Sg
    return dict(name=name, n=n, ng=ng, lf=lf, lt=lt, lb=lb, gbus=gbus, bg=bg, Pd=Pd, Pm=Pm, M=M, D=D, umax=umax,
                kappa=np.array([saliency, ool]), fric=fric * D, Dphys=phys_damp * D, base=base, Sg=Sg)


# ----------------------------------------------------------------------------- algebraic part
@njit
def alg_res_jac(theta, delta, Pd, lf, lt, lb, gbus, bg, with_jac):
    n = theta.shape[0]
    g = -Pd.copy()
    J = np.zeros((n, n))
    for i in range(gbus.shape[0]):
        j = gbus[i]
        a = delta[i] - theta[j]
        g[j] += bg[i] * np.sin(a)
        J[j, j] += bg[i] * np.cos(a)
    for l in range(lf.shape[0]):
        j, k = lf[l], lt[l]
        s = lb[l] * np.sin(theta[j] - theta[k]); c = lb[l] * np.cos(theta[j] - theta[k])
        g[j] -= s; g[k] += s
        J[j, j] += c; J[k, k] += c; J[j, k] -= c; J[k, j] -= c
    return g, J


@njit
def solve_theta(theta0, delta, Pd, lf, lt, lb, gbus, bg, tol, maxit):
    th = theta0.copy()
    nit = 0
    for it in range(maxit):
        g, J = alg_res_jac(th, delta, Pd, lf, lt, lb, gbus, bg, True)
        nr = np.max(np.abs(g))
        if nr < tol:
            break
        th = th + np.linalg.solve(J, g)
        nit += 1
    return th, nit


@njit
def tangent_theta(theta0, Jinv, delta0, delta, gbus, bg):
    rhs = np.zeros(theta0.shape[0])
    for i in range(gbus.shape[0]):
        j = gbus[i]
        rhs[j] += bg[i] * np.cos(delta0[i] - theta0[j]) * (delta[i] - delta0[i])
    return theta0 + Jinv @ rhs


# ----------------------------------------------------------------------------- generator residual features
NF = 13


@njit
def feats(psi, w, u):
    f = np.empty(NF)
    f[0] = np.sin(psi); f[1] = np.cos(psi); f[2] = np.sin(2 * psi); f[3] = np.cos(2 * psi)
    f[4] = np.sin(3 * psi); f[5] = np.cos(3 * psi); f[6] = w; f[7] = w * w; f[8] = w ** 3
    f[9] = u; f[10] = u * w; f[11] = 1.0; f[12] = psi
    return f


@njit
def res_acc(delta, w, u, theta, gbus, coef):
    out = np.zeros(delta.shape[0])
    for i in range(delta.shape[0]):
        f = feats(delta[i] - theta[gbus[i]], w[i], u[i])
        s = 0.0
        for k in range(NF):
            s += coef[i, k] * f[k]
        cb = coef[i, NF]                      # saturation bound = 1.5 x largest training residual (bounded-residual assumption)
        if cb > 0.0:
            s = min(max(s, -cb), cb)
        out[i] = s
    return out


@njit
def swing_acc(delta, w, u, theta, gbus, bg, M, Dm, Pm, extra, kappa, fric):
    """Acceleration.  extra=1: true plant (saliency + saturating friction); extra=0: model with damping Dm only."""
    ng = delta.shape[0]
    a = np.zeros(ng)
    for i in range(ng):
        psi = delta[i] - theta[gbus[i]]
        pe = bg[i] * np.sin(psi)
        fr = Dm[i] * w[i]
        if extra:
            pe += kappa[0] * bg[i] * np.sin(2 * psi) + kappa[1] * bg[i] * psi * abs(psi)
            fr += fric[i] * np.tanh(2.0 * w[i])
        a[i] = (Pm[i] + u[i] - fr - pe) / M[i]
    return a


@njit
def plant_step(delta, w, theta, u, h, t, Pd0, lamp, lom, lph, lf, lt, lb, gbus, bg, M, Dm, Pm, kappa, fric):
    ng = delta.shape[0]
    def pd(tt):
        return Pd0 * (1.0 + lamp * np.sin(lom * tt + lph))
    th1, _ = solve_theta(theta, delta, pd(t), lf, lt, lb, gbus, bg, 1e-11, 20)
    k1d = w; k1w = swing_acc(delta, w, u, th1, gbus, bg, M, Dm, Pm, True, kappa, fric)
    d2 = delta + 0.5 * h * k1d; w2 = w + 0.5 * h * k1w
    th2, _ = solve_theta(th1, d2, pd(t + 0.5 * h), lf, lt, lb, gbus, bg, 1e-11, 20)
    k2d = w2; k2w = swing_acc(d2, w2, u, th2, gbus, bg, M, Dm, Pm, True, kappa, fric)
    d3 = delta + 0.5 * h * k2d; w3 = w + 0.5 * h * k2w
    th3, _ = solve_theta(th2, d3, pd(t + 0.5 * h), lf, lt, lb, gbus, bg, 1e-11, 20)
    k3d = w3; k3w = swing_acc(d3, w3, u, th3, gbus, bg, M, Dm, Pm, True, kappa, fric)
    d4 = delta + h * k3d; w4 = w + h * k3w
    th4, _ = solve_theta(th3, d4, pd(t + h), lf, lt, lb, gbus, bg, 1e-11, 20)
    k4d = w4; k4w = swing_acc(d4, w4, u, th4, gbus, bg, M, Dm, Pm, True, kappa, fric)
    dn = delta + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
    wn = w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
    thn, _ = solve_theta(th4, dn, pd(t + h), lf, lt, lb, gbus, bg, 1e-11, 20)
    return dn, wn, thn


@njit
def pred_step(delta, w, theta, u, h, Pd, mode, use_res, coef, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric):
    """Predictor RK4 step.  mode 0: first-order tangent update of the algebraic variables (no corrector);
    mode 1: Newton corrector (projection onto g=0) at every stage and at the end."""
    ng = delta.shape[0]
    g0, J0 = alg_res_jac(theta, delta, Pd, lf, lt, lb, gbus, bg, True)
    if mode == 0:
        Jinv = np.linalg.inv(J0)
    else:
        Jinv = J0
    zero_ng = np.zeros(ng)
    # stage 1
    th1 = theta
    a1 = swing_acc(delta, w, u, th1, gbus, bg, M, Dphys, Pm, False, kappa, fric)
    if use_res:
        a1 = a1 + res_acc(delta, w, u, th1, gbus, coef)
    d2 = delta + 0.5 * h * w; w2 = w + 0.5 * h * a1
    th2 = tangent_theta(theta, Jinv, delta, d2, gbus, bg) if mode == 0 else solve_theta(theta, d2, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)[0]
    a2 = swing_acc(d2, w2, u, th2, gbus, bg, M, Dphys, Pm, False, kappa, fric)
    if use_res:
        a2 = a2 + res_acc(d2, w2, u, th2, gbus, coef)
    d3 = delta + 0.5 * h * w2; w3 = w + 0.5 * h * a2
    th3 = tangent_theta(theta, Jinv, delta, d3, gbus, bg) if mode == 0 else solve_theta(th2, d3, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)[0]
    a3 = swing_acc(d3, w3, u, th3, gbus, bg, M, Dphys, Pm, False, kappa, fric)
    if use_res:
        a3 = a3 + res_acc(d3, w3, u, th3, gbus, coef)
    d4 = delta + h * w3; w4 = w + h * a3
    th4 = tangent_theta(theta, Jinv, delta, d4, gbus, bg) if mode == 0 else solve_theta(th3, d4, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)[0]
    a4 = swing_acc(d4, w4, u, th4, gbus, bg, M, Dphys, Pm, False, kappa, fric)
    if use_res:
        a4 = a4 + res_acc(d4, w4, u, th4, gbus, coef)
    dn = delta + h * (w + 2 * w2 + 2 * w3 + w4) / 6.0
    wn = w + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
    thn = tangent_theta(theta, Jinv, delta, dn, gbus, bg) if mode == 0 else solve_theta(th4, dn, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)[0]
    return dn, wn, thn


# ----------------------------------------------------------------------------- equilibrium, regularity, spectrum
def equilibrium(case, Pd=None, Pm=None, guess=None):
    n, ng = case['n'], case['ng']
    Pd = case['Pd'] if Pd is None else Pd; Pm = case['Pm'] if Pm is None else Pm
    lf, lt, lb, gbus, bg = case['lf'], case['lt'], case['lb'], case['gbus'], case['bg']

    def F(z):
        th, de = z[:n], z[n:]
        g, _ = alg_res_jac(th, de, Pd, lf, lt, lb, gbus, bg, False)
        pe = bg * np.sin(de - th[gbus]) * 1.0
        return np.r_[g, Pm - pe]
    z0 = np.zeros(n + ng) if guess is None else guess
    if guess is None:
        z0[n:] = 0.1
        z0[:n] = 0.05
    sol = root(F, z0, method='lm', options=dict(xtol=1e-14, ftol=1e-14, maxiter=4000))
    z = sol.x
    res = np.max(np.abs(F(z)))
    th, de = z[:n].copy(), z[n:].copy()
    coi = np.sum(case['M'] * de) / np.sum(case['M'])
    return th - coi, de - coi, res


def regularity(case, th, de, Pd=None):
    Pd = case['Pd'] if Pd is None else Pd
    _, J = alg_res_jac(th, de, Pd, case['lf'], case['lt'], case['lb'], case['gbus'], case['bg'], True)
    ev = np.linalg.eigvalsh(J)
    ang = np.abs(th[case['lf']] - th[case['lt']])
    return float(ev.min()), float(ev.max() / ev.min()), float(np.degrees(ang.max())), float(np.degrees(np.max(np.abs(de - th[case['gbus']]))))


def swing_jacobian_eigs(case, th, de):
    """Eigenvalues of the reduced (algebraic-eliminated) linearisation of the model (true plant, u=0)."""
    ng, n = case['ng'], case['n']
    Pd = case['Pd']
    def f(x):
        d, w = x[:ng], x[ng:]
        t, _ = solve_theta(th, d, Pd, case['lf'], case['lt'], case['lb'], case['gbus'], case['bg'], 1e-13, 30)
        a = swing_acc(d, w, np.zeros(ng), t, case['gbus'], case['bg'], case['M'], case['D'], case['Pm'], True, case['kappa'], case['fric'])
        return np.r_[w, a]
    x0 = np.r_[de, np.zeros(ng)]
    A = np.zeros((2 * ng, 2 * ng)); e = 1e-6
    for j in range(2 * ng):
        dx = np.zeros(2 * ng); dx[j] = e
        A[:, j] = (f(x0 + dx) - f(x0 - dx)) / (2 * e)
    return np.linalg.eigvals(A)


def loadability_continuation(case_name, mus, **kw):
    rows = []; guess = None
    for mu in mus:
        c = build_case(case_name, load_scale=mu, **kw)
        th, de, res = equilibrium(c, guess=guess)
        if res > 1e-8:
            rows.append((mu, np.nan, np.nan, np.nan, np.nan, res)); break
        lam, cond, amax, gmax = regularity(c, th, de)
        guess = np.r_[th, de]
        rows.append((mu, lam, cond, amax, gmax, res))
    return rows
