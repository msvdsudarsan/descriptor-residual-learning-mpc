"""AC descriptor variant of MATPOWER case9 / case39 (agent_ac, minimal version).

Differential variables: rotor angle delta_i, speed w_i (classical swing, constant |E'_i| behind x'd_i), identical in structure to
network_benchmark.py.  Algebraic variables: z = (theta_1..theta_n, V_1..V_n)  (2n unknowns, bus angles AND voltage magnitudes):
    g_P,j = P_gen,j - Pd_j (V_j/V0_j)^np - V_j sum_k V_k (G_jk cos th_jk + B_jk sin th_jk) = 0
    g_Q,j = Q_gen,j - Qd_j (V_j/V0_j)^nq - V_j sum_k V_k (G_jk sin th_jk - B_jk cos th_jk) = 0
    P_gen = bgE_i V_j sin(psi),  Q_gen = bgE_i V_j cos(psi) - V_j^2 / x_i,  psi = delta_i - theta_j,  bgE_i = E_i / x_i
Ybus from MATPOWER r, x, b (line charging), tap ratio and bus shunts (Gs, Bs).  Default loads: constant PQ (np = nq = 0).
Inertia, damping, x'd are the same declared assumptions as the lossless model (Sg = Pmax, x'd = 0.3 pu on Sg, H = 4 s).
E'_i is computed from the solved MATPOWER power flow (slack generator absorbs the losses), so the nominal point is a true
equilibrium of the swing equations with Pm_i = Pg_i.
"""
import numpy as np
from scipy.optimize import root
from common import njit
from network_benchmark import parse_matpower, FS


def make_ybus(mp):
    base = mp['baseMVA']; bus, br = mp['bus'], mp['branch']
    n = len(bus); idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    Y = np.zeros((n, n), complex)
    for r in br:
        if r.shape[0] > 10 and r[10] == 0:
            continue
        f, t = idx[int(r[0])], idx[int(r[1])]
        ys = 1.0 / (r[2] + 1j * r[3]); bc = r[4]; tap = r[8] if r[8] != 0 else 1.0
        Ytt = ys + 1j * bc / 2; Yff = Ytt / tap ** 2; Yft = -ys / tap; Ytf = -ys / tap
        Y[f, f] += Yff; Y[t, t] += Ytt; Y[f, t] += Yft; Y[t, f] += Ytf
    Y += np.diag((bus[:, 4] + 1j * bus[:, 5]) / base)
    return Y


def build_ac(name, load_scale=1.0, xd_pu=0.3, H=4.0, zeta_rate=2.0, saliency=0.08, fric=0.15, phys_damp=0.8, ool=0.0,
             npw=0.0, nqw=0.0, line_out=None, charging=True):
    mp = parse_matpower(name)
    base = mp['baseMVA']; bus, gen = mp['bus'], mp['gen']
    if not charging:
        mp['branch'] = mp['branch'].copy(); mp['branch'][:, 4] = 0.0
    if line_out is not None:
        mp['branch'] = mp['branch'].copy(); mp['branch'][line_out, 10] = 0.0
    Y = make_ybus(mp)
    n, ng = len(bus), len(gen)
    idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    Pd = bus[:, 2] / base * load_scale; Qd = bus[:, 3] / base * load_scale
    gbus = np.array([idx[int(b)] for b in gen[:, 0]], dtype=np.int64)
    assert len(set(gbus)) == ng
    Sg = gen[:, 8] / base
    xinv = Sg / xd_pu
    M = 2 * H * Sg / FS; D = zeta_rate * 2 * M
    return dict(name=name, n=n, ng=ng, Y=Y, G=Y.real.copy(), B=Y.imag.copy(), gbus=gbus, Pd=Pd, Qd=Qd, Sg=Sg, xinv=xinv,
                M=M, D=D, umax=0.10 * Sg, kappa=np.array([saliency, ool]), fric=fric * D, Dphys=phys_damp * D, base=base,
                npw=npw, nqw=nqw, mp=mp, bustype=bus[:, 1].astype(int), Vg=gen[:, 5], Pg0=gen[:, 1] / base * load_scale,
                lf=np.array([idx[int(b)] for b in mp['branch'][:, 0]]), lt=np.array([idx[int(b)] for b in mp['branch'][:, 1]]))


def power_flow(c):
    """MATPOWER-style Newton power flow with the case's voltage set points; slack generator absorbs losses."""
    n = c['n']; Y = c['Y']; bt = c['bustype']; gbus = c['gbus']
    slack = int(np.where(bt == 3)[0][0]); pv = [j for j in range(n) if bt[j] == 2]; pq = [j for j in range(n) if bt[j] == 1]
    Vset = np.ones(n)
    for i, j in enumerate(gbus):
        Vset[j] = c['Vg'][i]
    Pg = np.zeros(n); Pg[gbus] = c['Pg0']
    Psp = Pg - c['Pd']; Qsp = -c['Qd']
    nons = [j for j in range(n) if j != slack]
    def F(x):
        th = np.zeros(n); V = Vset.copy()
        th[nons] = x[:len(nons)]; V[pq] = x[len(nons):]
        S = (V * np.exp(1j * th)) * np.conj(Y @ (V * np.exp(1j * th)))
        return np.r_[S.real[nons] - Psp[nons], S.imag[pq] - Qsp[pq]]
    x0 = np.r_[np.zeros(len(nons)), np.ones(len(pq))]
    sol = root(F, x0, method='hybr', options=dict(xtol=1e-14))
    x = sol.x; th = np.zeros(n); V = Vset.copy(); th[nons] = x[:len(nons)]; V[pq] = x[len(nons):]
    Vc = V * np.exp(1j * th); S = Vc * np.conj(Y @ Vc)
    res = np.max(np.abs(F(x)))
    return th, V, S, res


def nominal_point(c):
    """Equilibrium: (z0, delta0, Pm, bgE, V0) with COI-referenced angles.  Fills c['bgE'], c['Pm'], c['V0']."""
    th, V, S, res = power_flow(c); gbus = c['gbus']
    Sgen = S[gbus] + c['Pd'][gbus] + 1j * c['Qd'][gbus]                # generator terminal complex power
    Vc = V[gbus] * np.exp(1j * th[gbus]); I = np.conj(Sgen / Vc)
    x = 1.0 / c['xinv']; Eph = Vc + 1j * x * I
    E = np.abs(Eph); de = np.angle(Eph)
    c['bgE'] = E * c['xinv']; c['Pm'] = Sgen.real.copy(); c['V0'] = V.copy(); c['E'] = E
    coi = np.sum(c['M'] * de) / np.sum(c['M'])
    return np.r_[th - coi, V], de - coi, res


def net_of(c):
    return (c['G'], c['B'], c['gbus'], c['bgE'], c['xinv'], c['V0'], c['Pd'], c['Qd'], float(c['npw']), float(c['nqw']))


def dyn_of(c, phys=False):
    return (c['M'], c['Dphys'] if phys else c['D'], c['Pm'], c['kappa'], c['fric'])


# ----------------------------------------------------------------------------- algebraic part
@njit
def ac_res_jac(z, delta, fac, net, with_jac):
    G, B, gbus, bgE, xinv, V0, Pd, Qd, npw, nqw = net
    n = G.shape[0]
    th = z[:n]; V = z[n:]
    g = np.zeros(2 * n)
    J = np.zeros((2 * n, 2 * n)) if with_jac else np.zeros((1, 1))
    for j in range(n):
        Pn = 0.0; Qn = 0.0
        dPdthj = 0.0; dQdthj = 0.0; dPdVj = 0.0; dQdVj = 0.0
        for k in range(n):
            t = th[j] - th[k]; c = np.cos(t); s = np.sin(t)
            gc = G[j, k] * c + B[j, k] * s; gs = G[j, k] * s - B[j, k] * c
            Pn += V[k] * gc; Qn += V[k] * gs
            if with_jac and k != j:
                dPdth = V[j] * V[k] * gs;  dQdth = -V[j] * V[k] * gc
                J[j, k] = -dPdth; J[n + j, k] = -dQdth
                J[j, n + k] = -V[j] * gc; J[n + j, n + k] = -V[j] * gs
                dPdthj -= dPdth; dQdthj -= dQdth
                dPdVj += V[k] * gc; dQdVj += V[k] * gs
        g[j] = -V[j] * Pn; g[n + j] = -V[j] * Qn
        if with_jac:
            J[j, j] = -dPdthj; J[n + j, j] = -dQdthj
            J[j, n + j] = -(dPdVj + 2.0 * G[j, j] * V[j]); J[n + j, n + j] = -(dQdVj - 2.0 * B[j, j] * V[j])
    for j in range(n):                                                   # loads
        pl = fac * Pd[j] * (V[j] / V0[j]) ** npw; ql = fac * Qd[j] * (V[j] / V0[j]) ** nqw
        g[j] -= pl; g[n + j] -= ql
        if with_jac:
            J[j, n + j] -= npw * pl / V[j]; J[n + j, n + j] -= nqw * ql / V[j]
    for i in range(gbus.shape[0]):                                       # generators
        j = gbus[i]; psi = delta[i] - th[j]; s = np.sin(psi); c = np.cos(psi)
        g[j] += bgE[i] * V[j] * s; g[n + j] += bgE[i] * V[j] * c - xinv[i] * V[j] ** 2
        if with_jac:
            J[j, j] += -bgE[i] * V[j] * c; J[j, n + j] += bgE[i] * s
            J[n + j, j] += bgE[i] * V[j] * s; J[n + j, n + j] += bgE[i] * c - 2.0 * xinv[i] * V[j]
    return g, J


@njit
def dg_ddelta(z, delta, net):
    """d g / d delta  (2n x ng)."""
    G, B, gbus, bgE, xinv, V0, Pd, Qd, npw, nqw = net
    n = G.shape[0]; ng = gbus.shape[0]
    out = np.zeros((2 * n, ng))
    for i in range(ng):
        j = gbus[i]; psi = delta[i] - z[j]
        out[j, i] = bgE[i] * z[n + j] * np.cos(psi); out[n + j, i] = -bgE[i] * z[n + j] * np.sin(psi)
    return out


@njit
def solve_z(z0, delta, fac, net, tol, maxit):
    z = z0.copy(); nit = 0
    for it in range(maxit):
        g, J = ac_res_jac(z, delta, fac, net, True)
        if np.max(np.abs(g)) < tol:
            break
        dz = np.linalg.solve(J, g)
        z = z - dz
        nit += 1
    g, _ = ac_res_jac(z, delta, fac, net, False)
    return z, nit, np.max(np.abs(g))


@njit
def tangent_z(z0, Jinv, delta0, delta, net):
    d = dg_ddelta(z0, delta0, net)
    return z0 - Jinv @ (d @ (delta - delta0))


# ----------------------------------------------------------------------------- generator residual features (same library as the repo)
NF = 13


@njit
def feats(psi, w, u):
    f = np.empty(NF)
    f[0] = np.sin(psi); f[1] = np.cos(psi); f[2] = np.sin(2 * psi); f[3] = np.cos(2 * psi)
    f[4] = np.sin(3 * psi); f[5] = np.cos(3 * psi); f[6] = w; f[7] = w * w; f[8] = w ** 3
    f[9] = u; f[10] = u * w; f[11] = 1.0; f[12] = psi
    return f


@njit
def res_acc(delta, w, u, z, net, coef):
    gbus = net[2]; n = net[0].shape[0]
    out = np.zeros(delta.shape[0])
    for i in range(delta.shape[0]):
        f = feats(delta[i] - z[gbus[i]], w[i], u[i])
        s = 0.0
        for k in range(NF):
            s += coef[i, k] * f[k]
        cb = coef[i, NF]
        if cb > 0.0:
            s = min(max(s, -cb), cb)
        out[i] = s
    return out


@njit
def swing_acc(delta, w, u, z, net, dyn, extra):
    """extra=1: true plant (saliency x V + saturating friction); extra=0: model with damping Dm only."""
    gbus = net[2]; bgE = net[3]; n = net[0].shape[0]
    M, Dm, Pm, kappa, fric = dyn
    ng = delta.shape[0]
    a = np.zeros(ng)
    for i in range(ng):
        j = gbus[i]; psi = delta[i] - z[j]; V = z[n + j]
        pe = bgE[i] * V * np.sin(psi)
        fr = Dm[i] * w[i]
        if extra:
            pe += kappa[0] * bgE[i] * V * np.sin(2 * psi) + kappa[1] * bgE[i] * V * psi * abs(psi)
            fr += fric[i] * np.tanh(2.0 * w[i])
        a[i] = (Pm[i] + u[i] - fr - pe) / M[i]
    return a


@njit
def plant_step(delta, w, z, u, h, t, net, dyn, lamp, lom, lph):
    def fac(tt):
        return 1.0 + lamp * np.sin(lom * tt + lph)
    z1, _, _ = solve_z(z, delta, fac(t), net, 1e-11, 25)
    k1d = w; k1w = swing_acc(delta, w, u, z1, net, dyn, True)
    d2 = delta + 0.5 * h * k1d; w2 = w + 0.5 * h * k1w
    z2, _, _ = solve_z(z1, d2, fac(t + 0.5 * h), net, 1e-11, 25)
    k2d = w2; k2w = swing_acc(d2, w2, u, z2, net, dyn, True)
    d3 = delta + 0.5 * h * k2d; w3 = w + 0.5 * h * k2w
    z3, _, _ = solve_z(z2, d3, fac(t + 0.5 * h), net, 1e-11, 25)
    k3d = w3; k3w = swing_acc(d3, w3, u, z3, net, dyn, True)
    d4 = delta + h * k3d; w4 = w + h * k3w
    z4, _, _ = solve_z(z3, d4, fac(t + h), net, 1e-11, 25)
    k4d = w4; k4w = swing_acc(d4, w4, u, z4, net, dyn, True)
    dn = delta + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
    wn = w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
    zn, _, _ = solve_z(z4, dn, fac(t + h), net, 1e-11, 25)
    return dn, wn, zn


@njit
def pred_step(delta, w, z, u, h, fac, mode, use_res, coef, net, dynp):
    """Predictor RK4 step. mode 0: tangent update (no corrector); mode 1: Newton projection at every stage; mode 2: learned
    algebraic map is supplied externally (not used here)."""
    g0, J0 = ac_res_jac(z, delta, fac, net, True)
    Jinv = np.linalg.inv(J0) if mode == 0 else J0
    a1 = swing_acc(delta, w, u, z, net, dynp, False)
    if use_res:
        a1 = a1 + res_acc(delta, w, u, z, net, coef)
    d2 = delta + 0.5 * h * w; w2 = w + 0.5 * h * a1
    z2 = tangent_z(z, Jinv, delta, d2, net) if mode == 0 else solve_z(z, d2, fac, net, 1e-10, 15)[0]
    a2 = swing_acc(d2, w2, u, z2, net, dynp, False)
    if use_res:
        a2 = a2 + res_acc(d2, w2, u, z2, net, coef)
    d3 = delta + 0.5 * h * w2; w3 = w + 0.5 * h * a2
    z3 = tangent_z(z, Jinv, delta, d3, net) if mode == 0 else solve_z(z2, d3, fac, net, 1e-10, 15)[0]
    a3 = swing_acc(d3, w3, u, z3, net, dynp, False)
    if use_res:
        a3 = a3 + res_acc(d3, w3, u, z3, net, coef)
    d4 = delta + h * w3; w4 = w + h * a3
    z4 = tangent_z(z, Jinv, delta, d4, net) if mode == 0 else solve_z(z3, d4, fac, net, 1e-10, 15)[0]
    a4 = swing_acc(d4, w4, u, z4, net, dynp, False)
    if use_res:
        a4 = a4 + res_acc(d4, w4, u, z4, net, coef)
    dn = delta + h * (w + 2 * w2 + 2 * w3 + w4) / 6.0
    wn = w + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
    zn = tangent_z(z, Jinv, delta, dn, net) if mode == 0 else solve_z(z4, dn, fac, net, 1e-10, 15)[0]
    return dn, wn, zn
