"""Analytic (discrete-adjoint) gradient of the projected-predictor cost used by the network controllers A2 and A3.

With the Newton projection the algebraic variables are an exact function theta(delta) of the rotor angles, and
d theta / d delta = J^{-1} E C with C = diag(b_i cos(psi_i)) (Proposition 3).  The predictor is therefore an ordinary
RK4 integration of the reduced model x = (delta, omega), and the gradient of the quadratic-penalty cost with respect to the
input sequence follows from one forward pass that stores the step sensitivities and one backward pass.  Cost: roughly
two predictor rollouts instead of N*n_g + 1 rollouts for finite differences.
"""
import numpy as np
from common import njit
from network_benchmark import alg_res_jac, solve_theta, NF


@njit
def _stage(delta, w, u, theta, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, coef):
    """Acceleration a and the Jacobian blocks of F=(w, a) at a state whose algebraic variables theta are consistent."""
    ng = delta.shape[0]; n = theta.shape[0]
    g, J = alg_res_jac(theta, delta, Pd, lf, lt, lb, gbus, bg, True)
    B = np.zeros((n, ng))
    psi = np.empty(ng)
    for i in range(ng):
        psi[i] = delta[i] - theta[gbus[i]]
        B[gbus[i], i] = bg[i] * np.cos(psi[i])
    Th = np.linalg.solve(J, B)                                   # d theta / d delta
    a = np.empty(ng); sp = np.empty(ng); sw = np.empty(ng); su = np.empty(ng)
    for i in range(ng):
        pe = bg[i] * np.sin(psi[i])
        a[i] = (Pm[i] + u[i] - Dphys[i] * w[i] - pe) / M[i]
        sp[i] = -bg[i] * np.cos(psi[i]) / M[i]
        sw[i] = -Dphys[i] / M[i]
        su[i] = 1.0 / M[i]
        f = np.empty(NF); df_p = np.zeros(NF); df_w = np.zeros(NF); df_u = np.zeros(NF)
        p, wi, ui = psi[i], w[i], u[i]
        f[0] = np.sin(p); f[1] = np.cos(p); f[2] = np.sin(2 * p); f[3] = np.cos(2 * p); f[4] = np.sin(3 * p); f[5] = np.cos(3 * p)
        f[6] = wi; f[7] = wi * wi; f[8] = wi ** 3; f[9] = ui; f[10] = ui * wi; f[11] = 1.0; f[12] = p
        df_p[0] = np.cos(p); df_p[1] = -np.sin(p); df_p[2] = 2 * np.cos(2 * p); df_p[3] = -2 * np.sin(2 * p)
        df_p[4] = 3 * np.cos(3 * p); df_p[5] = -3 * np.sin(3 * p); df_p[12] = 1.0
        df_w[6] = 1.0; df_w[7] = 2 * wi; df_w[8] = 3 * wi * wi; df_w[10] = ui
        df_u[9] = 1.0; df_u[10] = wi
        s = 0.0; rp = 0.0; rw = 0.0; ru = 0.0
        for k in range(NF):
            s += coef[i, k] * f[k]; rp += coef[i, k] * df_p[k]; rw += coef[i, k] * df_w[k]; ru += coef[i, k] * df_u[k]
        cb = coef[i, NF]
        if cb > 0.0:
            if s > cb or s < -cb:
                rp = 0.0; rw = 0.0; ru = 0.0; s = cb if s > 0 else -cb
        a[i] += s
        sp[i] += rp; sw[i] += rw; su[i] += ru
    Fx = np.zeros((2 * ng, 2 * ng)); Fu = np.zeros((2 * ng, ng))
    for i in range(ng):
        Fx[i, ng + i] = 1.0
        for j in range(ng):
            Pij = (1.0 if i == j else 0.0) - Th[gbus[i], j]
            Fx[ng + i, j] = sp[i] * Pij
        Fx[ng + i, ng + i] += sw[i]
        Fu[ng + i, i] = su[i]
    return a, Fx, Fu, Th


@njit
def mpc_cost_net_analytic(Uf, de0, w0, th0, uprev, N, Ts, nsub, mode, use_res, coef, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, kappa, fric,
                          rel_ref, umax, qd, qw, r, rd, gam, eps, pen):
    """Same cost as mpc_cost_net (projected predictor, mode=1) together with its exact gradient.  Returns (J, grad)."""
    ng = de0.shape[0]; n = th0.shape[0]; nx = 2 * ng
    h = Ts / nsub
    Msum = np.sum(M)
    de = de0.copy(); w = w0.copy(); th = th0.copy()
    J = 0.0
    Tx = np.zeros((N, nx, nx)); Tu = np.zeros((N, nx, ng)); gx = np.zeros((N, nx)); gu = np.zeros((N, ng))
    I = np.eye(nx)
    for k in range(N):
        u = np.empty(ng); up = np.empty(ng)
        for i in range(ng):
            u[i] = Uf[k * ng + i]
            up[i] = Uf[(k - 1) * ng + i] if k > 0 else uprev[i]
        TX = np.eye(nx); TU = np.zeros((nx, ng))
        Th = np.zeros((n, ng))
        for s in range(nsub):
            a1, F1x, F1u, Th1 = _stage(de, w, u, th, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, coef)
            d2 = de + 0.5 * h * w; w2 = w + 0.5 * h * a1
            th2, _ = solve_theta(th, d2, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)
            a2, F2x, F2u, Th2 = _stage(d2, w2, u, th2, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, coef)
            d3 = de + 0.5 * h * w2; w3 = w + 0.5 * h * a2
            th3, _ = solve_theta(th2, d3, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)
            a3, F3x, F3u, Th3 = _stage(d3, w3, u, th3, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, coef)
            d4 = de + h * w3; w4 = w + h * a3
            th4, _ = solve_theta(th3, d4, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)
            a4, F4x, F4u, Th4 = _stage(d4, w4, u, th4, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, coef)
            K1x = F1x; K1u = F1u
            K2x = F2x @ (I + 0.5 * h * K1x); K2u = F2x @ (0.5 * h * K1u) + F2u
            K3x = F3x @ (I + 0.5 * h * K2x); K3u = F3x @ (0.5 * h * K2u) + F3u
            K4x = F4x @ (I + h * K3x); K4u = F4x @ (h * K3u) + F4u
            Sx = I + h / 6.0 * (K1x + 2 * K2x + 2 * K3x + K4x)
            Su = h / 6.0 * (K1u + 2 * K2u + 2 * K3u + K4u)
            TU = Sx @ TU + Su; TX = Sx @ TX
            dn = de + h * (w + 2 * w2 + 2 * w3 + w4) / 6.0
            wn = w + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
            th, _ = solve_theta(th4, dn, Pd, lf, lt, lb, gbus, bg, 1e-10, 12)
            de = dn; w = wn
        Tx[k] = TX; Tu[k] = TU
        # stage cost (identical to mpc_cost_net) and its partial derivatives
        coi = np.sum(M * de) / Msum
        rel = np.empty(ng); srel = 0.0
        for i in range(ng):
            rel[i] = (de[i] - coi) - rel_ref[i]; srel += rel[i]
        gd = np.zeros(ng); gw = np.zeros(ng); gth = np.zeros(n)
        for i in range(ng):
            J += qd * rel[i] * rel[i] + qw * w[i] * w[i] + r * (u[i] / umax[i]) ** 2 + rd * ((u[i] - up[i]) / umax[i]) ** 2
            gd[i] += 2 * qd * (rel[i] - srel * M[i] / Msum)
            gw[i] += 2 * qw * w[i]
            gu[k, i] += 2 * r * u[i] / umax[i] ** 2 + 2 * rd * (u[i] - up[i]) / umax[i] ** 2
        lim = gam - eps[k + 1]
        for l in range(lf.shape[0]):
            d_ = th[lf[l]] - th[lt[l]]
            v = abs(d_) - lim
            if v > 0:
                J += pen * v * v
                sg = 1.0 if d_ > 0 else -1.0
                gth[lf[l]] += 2 * pen * v * sg; gth[lt[l]] -= 2 * pen * v * sg
        for i in range(ng):
            d_ = de[i] - th[gbus[i]]
            v = abs(d_) - (lim + 0.35)
            if v > 0:
                J += pen * v * v
                sg = 1.0 if d_ > 0 else -1.0
                gd[i] += 2 * pen * v * sg; gth[gbus[i]] -= 2 * pen * v * sg
        # d theta / d delta at the end state
        _, _, _, ThN = _stage(de, w, u, th, Pd, lf, lt, lb, gbus, bg, M, Dphys, Pm, coef)
        for j in range(ng):
            acc = 0.0
            for m in range(n):
                acc += ThN[m, j] * gth[m]
            gd[j] += acc
        for i in range(ng):
            gx[k, i] = gd[i]; gx[k, ng + i] = gw[i]
    grad = np.zeros(N * ng)
    lam_prev = np.zeros(nx)
    for k in range(N - 1, -1, -1):
        lam = gx[k].copy()
        if k < N - 1:
            lam = lam + Tx[k + 1].T @ lam_prev
        gk = gu[k] + Tu[k].T @ lam
        if k < N - 1:
            for i in range(ng):
                gk[i] -= 2 * rd * (Uf[(k + 1) * ng + i] - Uf[k * ng + i]) / umax[i] ** 2
        for i in range(ng):
            grad[k * ng + i] = gk[i]
        lam_prev = lam
    return J, grad
