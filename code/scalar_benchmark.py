"""Scalar nonlinear descriptor benchmark (Section 3 of the manuscript).

True plant (index-1 semi-explicit DAE, one angle/frequency pair, one algebraic voltage-like variable):
    d' = w
    M w' = Pm - kv*v*sin(d) - D*w + u - c3*d^3 + wd(t)
    0   = g(d,v,u) = v + a*sin(d) + cb*v^3 - (1 + ku*u)

Physics-only model: omits the cubic term and mis-specifies kv and D (parametric + structural error).
The algebraic balance g is treated as known (Kirchhoff-type law) in every predictor variant.

Predictor variants (ablation A0-A2 at prediction level, A0-A3 at control level):
    A0 physics-only, algebraic variable advanced by a first-order tangent update (no corrector)
    A1 physics + learned residual, tangent update
    A2 physics + learned residual + Newton projection onto g = 0 at every stage
    A3 = A2 inside a residual-envelope-tightened receding-horizon controller
"""
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, expm, sqrtm
from common import njit, monomial_exponents, monomial_features, stlsq, conformal_quantile

M = 0.55
A_G, CB, KU = 0.12, 0.08, 0.04
TRUE_PM, TRUE_KV, TRUE_D, TRUE_C3 = 1.10, 0.85, 0.18, 0.10
SEVERITY = float(os.environ.get('PHYS_SEVERITY', '1.0'))      # 1.0: reference benchmark; scales every model mismatch
PHYS_PM, PHYS_KV, PHYS_D = 1.10, 0.85 - 0.05 * SEVERITY, 0.18 - 0.04 * SEVERITY
PHYS_C3 = TRUE_C3 * (1.0 - SEVERITY)                           # SEVERITY=1: cubic stiffness omitted
U_MAX = 0.8
D_REF = 0.55
D_MAX = 0.66          # angle limit used by the receding-horizon controllers
W_MAX = 0.60          # frequency-like limit
W_BAR = 0.22          # disturbance amplitude bound (force units)


# --------------------------------------------------------------------------- DAE pieces
@njit
def g_res(d, v, u):
    return v + A_G * np.sin(d) + CB * v ** 3 - (1.0 + KU * u)


@njit
def solve_v(d, u):
    v = 1.0 - A_G * np.sin(d) - KU * u
    for _ in range(8):
        v -= g_res(d, v, u) / (1.0 + 3.0 * CB * v * v)
    return v


@njit
def acc_true(d, w, u, wd):
    v = solve_v(d, u)
    return (TRUE_PM - TRUE_KV * v * np.sin(d) - TRUE_D * w + u - TRUE_C3 * d ** 3 + wd) / M


@njit
def acc_phys(d, w, u, v):
    return (PHYS_PM - PHYS_KV * v * np.sin(d) - PHYS_D * w + u - PHYS_C3 * d ** 3) / M


@njit
def res_eval(d, w, u, coef, E):
    s = 0.0
    for j in range(E.shape[0]):
        if coef[j] != 0.0:
            s += coef[j] * d ** E[j, 0] * w ** E[j, 1] * u ** E[j, 2]
    return s


@njit
def plant_step(d, w, u, t, h, amp, om, ph):
    """RK4 for the true DAE with exact algebraic solve at every stage; ZOH input."""
    k1d = w
    k1w = acc_true(d, w, u, amp * np.sin(om * t + ph))
    k2d = w + 0.5 * h * k1w
    k2w = acc_true(d + 0.5 * h * k1d, w + 0.5 * h * k1w, u, amp * np.sin(om * (t + 0.5 * h) + ph))
    k3d = w + 0.5 * h * k2w
    k3w = acc_true(d + 0.5 * h * k2d, w + 0.5 * h * k2w, u, amp * np.sin(om * (t + 0.5 * h) + ph))
    k4d = w + h * k3w
    k4w = acc_true(d + h * k3d, w + h * k3w, u, amp * np.sin(om * (t + h) + ph))
    return d + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0, w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0


@njit
def _F(d, w, u, v, use_res, coef, E):
    a = acc_phys(d, w, u, v)
    if use_res:
        a += res_eval(d, w, u, coef, E)
    return w, a


@njit
def pred_step(d, w, v, u, u_prev, h, mode, use_res, coef, E):
    """One RK4 step of a predictor.  mode 0: tangent (first-order) algebraic update;
    mode 1: Newton projection g = 0 at every stage and at the end of the step."""
    d0 = d
    gv0 = 1.0 + 3.0 * CB * v * v
    if mode == 0:
        v = v + KU * (u - u_prev) / gv0            # input jump, first order
        gv0 = 1.0 + 3.0 * CB * v * v
    gd0 = A_G * np.cos(d0)
    v1 = solve_v(d, u) if mode == 1 else v
    k1d, k1w = _F(d, w, u, v1, use_res, coef, E)
    d2 = d + 0.5 * h * k1d
    v2 = solve_v(d2, u) if mode == 1 else v - gd0 * (d2 - d0) / gv0
    k2d, k2w = _F(d2, w + 0.5 * h * k1w, u, v2, use_res, coef, E)
    d3 = d + 0.5 * h * k2d
    v3 = solve_v(d3, u) if mode == 1 else v - gd0 * (d3 - d0) / gv0
    k3d, k3w = _F(d3, w + 0.5 * h * k2w, u, v3, use_res, coef, E)
    d4 = d + h * k3d
    v4 = solve_v(d4, u) if mode == 1 else v - gd0 * (d4 - d0) / gv0
    k4d, k4w = _F(d4, w + h * k3w, u, v4, use_res, coef, E)
    dn = d + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
    wn = w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
    vn = solve_v(dn, u) if mode == 1 else v - gd0 * (dn - d0) / gv0
    return dn, wn, vn


@njit
def rollout(d0, w0, U, uprev, Ts, nsub, mode, use_res, coef, E):
    N = U.shape[0]
    out = np.zeros((N + 1, 3))
    d, w = d0, w0
    v = solve_v(d0, uprev)
    out[0, 0], out[0, 1], out[0, 2] = d, w, v
    u_p = uprev
    h = Ts / nsub
    for k in range(N):
        for s in range(nsub):
            d, w, v = pred_step(d, w, v, U[k], u_p, h, mode, use_res, coef, E)
            u_p = U[k]
        out[k + 1, 0], out[k + 1, 1], out[k + 1, 2] = d, w, v
    return out


# --------------------------------------------------------------------------- controllers
def static_law(d, w, u_ff, gain_robust=0.06):
    u = u_ff - 1.25 * (d - D_REF) - 0.52 * w - gain_robust * np.sign(w) * min(abs(w), 1.0)
    return float(np.clip(u, -U_MAX, U_MAX))


def equilibrium_input(dref, pm, kv, c3, extra=0.0):
    v = float(solve_v(dref, 0.0))
    for _ in range(30):                              # u enters g weakly: fixed-point
        u = -(pm - kv * v * np.sin(dref) - c3 * dref ** 3 + extra)
        v = float(solve_v(dref, u))
    return float(-(pm - kv * v * np.sin(dref) - c3 * dref ** 3 + extra))


@njit
def mpc_cost(U, d0, w0, uprev, N, Ts, nsub, mode, use_res, coef, E, dref,
             qd, qw, r, rd, p11, p12, p22, dmax, wmax, eps, pen):
    d, w = d0, w0
    v = solve_v(d0, uprev)
    u_p = uprev
    h = Ts / nsub
    J = 0.0
    for k in range(N):
        u = U[k]
        for s in range(nsub):
            d, w, v = pred_step(d, w, v, u, u_p, h, mode, use_res, coef, E)
            u_p = u
        ed = d - dref
        up = U[k - 1] if k > 0 else uprev
        J += qd * ed * ed + qw * w * w + r * u * u + rd * (u - up) ** 2
        vd = max(0.0, d - (dmax - eps[k + 1, 0]))
        vw = max(0.0, abs(w) - (wmax - eps[k + 1, 1]))
        J += pen * (vd * vd + vw * vw)
    ed = d - dref
    J += p11 * ed * ed + 2.0 * p12 * ed * w + p22 * w * w
    return J


@njit
def mpc_cost_grad(U, d0, w0, uprev, N, Ts, nsub, mode, use_res, coef, E, dref,
                  qd, qw, r, rd, p11, p12, p22, dmax, wmax, eps, pen):
    J0 = mpc_cost(U, d0, w0, uprev, N, Ts, nsub, mode, use_res, coef, E, dref,
                  qd, qw, r, rd, p11, p12, p22, dmax, wmax, eps, pen)
    g = np.zeros(N)
    h = 1e-6
    for k in range(N):
        Up = U.copy(); Up[k] += h
        Um = U.copy(); Um[k] -= h
        g[k] = (mpc_cost(Up, d0, w0, uprev, N, Ts, nsub, mode, use_res, coef, E, dref,
                         qd, qw, r, rd, p11, p12, p22, dmax, wmax, eps, pen)
                - mpc_cost(Um, d0, w0, uprev, N, Ts, nsub, mode, use_res, coef, E, dref,
                           qd, qw, r, rd, p11, p12, p22, dmax, wmax, eps, pen)) / (2 * h)
    return J0, g


class MPCSettings:
    N = 12
    Ts = 0.1
    nsub = 2
    qd, qw, r, rd = 40.0, 2.0, 0.05, 0.5
    pen = 2.0e3


def terminal_weight(coef, E, use_res, dref, Ts, qd, qw, r):
    """DARE terminal cost from the discrete linearisation of the *model used* at (dref, 0)."""
    ueq = equilibrium_input(dref, PHYS_PM, PHYS_KV, PHYS_C3)
    eps_ = 1e-6
    def fl(x, u):
        out = rollout(x[0], x[1], np.array([u]), u, Ts, 2, 1, use_res, coef, E)
        return out[1, :2]
    x0 = np.array([dref, 0.0])
    A = np.zeros((2, 2)); B = np.zeros((2, 1))
    for j in range(2):
        dx = np.zeros(2); dx[j] = eps_
        A[:, j] = (fl(x0 + dx, ueq) - fl(x0 - dx, ueq)) / (2 * eps_)
    B[:, 0] = (fl(x0, ueq + eps_) - fl(x0, ueq - eps_)) / (2 * eps_)
    Pm = solve_discrete_are(A, B, np.diag([qd, qw]), np.array([[r]]))
    return Pm, A, B


def contraction_factor(Ts, dref):
    """Contraction c < 1 of the ancillary static feedback in a Lyapunov norm (used for tube growth)."""
    v = float(solve_v(dref, 0.0))
    kk = TRUE_KV * (v * np.cos(dref)) + 3 * TRUE_C3 * dref ** 2
    A = np.array([[0.0, 1.0], [-kk / M, -TRUE_D / M]])
    B = np.array([[0.0], [1.0 / M]])
    K = np.array([[1.25, 0.52]])
    Phi = expm((A - B @ K) * Ts)
    P = solve_discrete_lyapunov(Phi.T, np.eye(2))
    Ps = np.real(sqrtm(P)); Pis = np.linalg.inv(Ps)
    return float(np.linalg.norm(Ps @ Phi @ Pis, 2))


def run_closed_loop(controller, x0, T, dist, Ts=0.1, hp=0.01, rng=None):
    """controller(t, d, w, u_prev) -> u ; returns arrays on the sampling grid."""
    amp, om, ph = dist
    nsteps = int(round(T / Ts)); nsub = int(round(Ts / hp))
    d, w = x0
    u_prev = 0.0
    log = np.zeros((nsteps + 1, 5))                     # t, d, w, v, u
    t = 0.0
    solve_times = []
    for k in range(nsteps):
        t0 = time.perf_counter()
        u = controller(t, d, w, u_prev)
        solve_times.append(time.perf_counter() - t0)
        log[k] = (t, d, w, solve_v(d, u), u)
        for _ in range(nsub):
            d, w = plant_step(d, w, u, t, hp, amp, om, ph)
            t += hp
        u_prev = u
    log[nsteps] = (t, d, w, solve_v(d, u_prev), u_prev)
    return log, float(np.mean(solve_times))


# --------------------------------------------------------------------------- identification data
def make_identification_data(n_traj, seed, noise, h=0.05, T=6.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_traj):
        d, w = rng.uniform(0.0, 0.9), rng.uniform(-0.4, 0.4)
        amp, om, ph = rng.uniform(0, 0.06), rng.uniform(0.5, 3.0), rng.uniform(0, 2 * np.pi)
        n = int(T / h)
        X = np.zeros((n + 1, 2)); U = np.zeros(n)
        ref = rng.uniform(0.15, 0.85); dither = 0.0
        t = 0.0
        for k in range(n):
            if k % 30 == 0:
                ref = rng.uniform(0.15, 0.85)
            if k % 5 == 0:
                dither = rng.normal(0, 0.18)
            u = float(np.clip(equilibrium_input(ref, TRUE_PM, TRUE_KV, TRUE_C3) - 1.25 * (d - ref) - 0.52 * w + dither, -U_MAX, U_MAX))
            X[k] = (d, w); U[k] = u
            d, w = plant_step(d, w, u, t, h, amp, om, ph); t += h
        X[n] = (d, w)
        Xm = X + rng.normal(0, noise, X.shape)
        out.append(dict(X=X, Xm=Xm, U=U, h=h))
    return out


def weak_form_targets(traj, E):
    """Integral-form regression data (robust to the input jumps of a ZOH signal):
    y_k = [w_{k+1}-w_k]/h - 0.5*(a_phys(x_k,u_k)+a_phys(x_{k+1},u_k));  Theta_k = 0.5*(phi(x_k,u_k)+phi(x_{k+1},u_k))."""
    Xm, U, h = traj['Xm'], traj['U'], traj['h']
    d0, w0, d1, w1 = Xm[:-1, 0], Xm[:-1, 1], Xm[1:, 0], Xm[1:, 1]
    v0 = np.array([solve_v(a, u) for a, u in zip(d0, U)]); v1 = np.array([solve_v(a, u) for a, u in zip(d1, U)])
    ap0 = np.array([acc_phys(a, b, u, v) for a, b, u, v in zip(d0, w0, U, v0)])
    ap1 = np.array([acc_phys(a, b, u, v) for a, b, u, v in zip(d1, w1, U, v1)])
    y = (w1 - w0) / h - 0.5 * (ap0 + ap1)
    Th = 0.5 * (monomial_features(np.c_[d0, w0, U], E) + monomial_features(np.c_[d1, w1, U], E))
    return Th, y


def exact_residual(d, w, u, wd=0.0):
    v = solve_v(d, u)
    return acc_true(d, w, u, wd) - acc_phys(d, w, u, v)


def fit_residual(train, val, degrees=(3, 4), ridges=(1e-8, 1e-6, 1e-4), threshes=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2), tol=1.05):
    """Weak-form thresholded regression.  Among all (degree, ridge, threshold) candidates, the sparsest model whose
    validation RMSE is within `tol` of the best one is returned (one-standard-error-type parsimony rule)."""
    cands = []
    for deg in degrees:
        E = monomial_exponents(3, deg)
        Tr = [weak_form_targets(t, E) for t in train]; Va = [weak_form_targets(t, E) for t in val]
        Th = np.vstack([a for a, _ in Tr]); y = np.concatenate([b for _, b in Tr])
        Thv = np.vstack([a for a, _ in Va]); yv = np.concatenate([b for _, b in Va])
        for rg in ridges:
            for th in threshes:
                c = stlsq(Th, y, ridge=rg, thresh=th)
                e = np.sqrt(np.mean((yv - Thv @ c) ** 2))
                cands.append((e, int(np.sum(c != 0)), deg, rg, th, c, E))
    ebest = min(x[0] for x in cands)
    ok = [x for x in cands if x[0] <= tol * ebest]
    e, nact, deg, rg, th, c, E = min(ok, key=lambda x: (x[1], x[0]))
    return dict(val_rmse=e, degree=deg, ridge=rg, thresh=th, coef=c, E=E, best_val_rmse=ebest)


def collect_test_points(n_traj, seed):
    data = make_identification_data(n_traj, seed, 0.0)
    pts = []
    for tr in data:
        X, U = tr['X'][:-1], tr['U']
        pts.append(np.c_[X, U])
    return np.vstack(pts), data


# --------------------------------------------------------------------------- nonlinear analysis helpers
@njit
def static_law_nb(d, w, u_ff, dref, grob):
    s = 1.0 if w > 0 else (-1.0 if w < 0 else 0.0)
    u = u_ff - 1.25 * (d - dref) - 0.52 * w - grob * s * min(abs(w), 1.0)
    return min(max(u, -0.8), 0.8)


@njit
def basin_grid(d_grid, w_grid, u_ff, dref, T, h, feedback, p_bias):
    nd, nw = d_grid.shape[0], w_grid.shape[0]
    out = np.zeros((nd, nw, 2))
    for i in range(nd):
        for j in range(nw):
            d, w = d_grid[i], w_grid[j]
            t = 0.0
            ok = True
            for _ in range(int(T / h)):
                u = static_law_nb(d, w, u_ff, dref, 0.06) if feedback else u_ff
                d, w = plant_step(d, w, u, t, h, p_bias, 0.0, 1.5707963267948966)  # constant force p_bias
                t += h
                if abs(d) > 6.0 or abs(w) > 20.0:
                    ok = False
                    break
            out[i, j, 0] = d if ok else np.nan
            out[i, j, 1] = w if ok else np.nan
    return out


def closed_loop_field(d, w, u_ff, dref, p, feedback):
    u = static_law(d, w, u_ff) if feedback else u_ff
    return np.array([w, acc_true(d, w, u, p)])


def equilibria_branch(p_values, u_ff, dref, feedback):
    """Equilibria (w=0) of the true closed loop with constant extra force p; stability by FD Jacobian."""
    ds = np.linspace(-0.6, 3.2, 3801)
    rows = []
    for p in p_values:
        F = np.array([closed_loop_field(d, 0.0, u_ff, dref, p, feedback)[1] for d in ds])
        idx = np.where(np.sign(F[:-1]) * np.sign(F[1:]) < 0)[0]
        for i in idx:
            a, b = ds[i], ds[i + 1]
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = closed_loop_field(m, 0.0, u_ff, dref, p, feedback)[1]
                if np.sign(fm) == np.sign(closed_loop_field(a, 0.0, u_ff, dref, p, feedback)[1]):
                    a = m
                else:
                    b = m
            x = 0.5 * (a + b)
            J = np.zeros((2, 2)); e = 1e-6
            for k, dx in enumerate([(e, 0), (0, e)]):
                fp = closed_loop_field(x + dx[0], dx[1], u_ff, dref, p, feedback)
                fm_ = closed_loop_field(x - dx[0], -dx[1], u_ff, dref, p, feedback)
                J[:, k] = (fp - fm_) / (2 * e)
            ev = np.linalg.eigvals(J)
            rows.append((p, x, np.max(ev.real), ev))
    return rows
