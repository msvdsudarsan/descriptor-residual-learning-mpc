"""Helpers for the tube theory: reduced stiffness K, constants, angle maps. Uses the repo's model (network_benchmark)."""
import numpy as np
from network_benchmark import *

def args_of(c):
    return (c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'])

def Cmat(c):
    C = np.zeros((c['n'], c['ng'])); C[c['gbus'], np.arange(c['ng'])] = 1.0
    return C

def J_of(c, th, de, Pd=None):
    Pd = c['Pd'] if Pd is None else Pd
    return alg_res_jac(th, de, Pd, *args_of(c), True)[1]

def J1_of(c):
    z = np.zeros(c['n']); zg = np.zeros(c['ng'])
    return alg_res_jac(z, zg, c['Pd'], *args_of(c), True)[1]

def stiffness(c, th, de):
    """K = d f/d delta = Kd - Kd C^T J^{-1} C Kd (ng x ng), S = d theta/d delta (n x ng)."""
    C = Cmat(c); J = J_of(c, th, de)
    kd = c['bg'] * np.cos(de - th[c['gbus']])
    Kd = np.diag(kd)
    S = np.linalg.solve(J, C @ Kd)
    K = Kd - Kd @ C.T @ S
    return 0.5 * (K + K.T), S, Kd

def angles(c, th, de):
    """all constrained angles: line angles then rotor-to-bus angles"""
    return np.r_[th[c['lf']] - th[c['lt']], de - th[c['gbus']]]

def solve_alg(c, th0, de, Pd):
    return solve_theta(th0, de, Pd, *args_of(c), 1e-13, 60)[0]

def edge_Z(c, S):
    """Z[e,:] = d A_e / d delta for all constrained edges (lines then generator edges)"""
    n, ng = S.shape
    Zl = S[c['lf']] - S[c['lt']]
    Zg = np.eye(ng) - S[c['gbus']]
    return np.vstack([Zl, Zg])


# ---------------------------------------------------------------- constants of the tube theorem
def residual_lipschitz(coef, c, gbar, wmax, umax_scale=1.0, ngrid=61):
    """a_psi_i = sup |d rhat_i / d psi|, a_om_i = sup |d rhat_i / d omega| over |psi|<=gbar, |omega|<=wmax, |u|<=umax_i (analytic derivative of the library;
    the saturation clamp can only reduce the Lipschitz constant)."""
    ng = c['ng']; psi = np.linspace(-gbar, gbar, ngrid); w = np.linspace(-wmax, wmax, ngrid)
    P, W = np.meshgrid(psi, w, indexing='ij'); a_psi = np.zeros(ng); a_om = np.zeros(ng)
    for i in range(ng):
        cf = coef[i]; um = c['umax'][i] * umax_scale
        dpsi = (cf[0] * np.cos(P) - cf[1] * np.sin(P) + 2 * cf[2] * np.cos(2 * P) - 2 * cf[3] * np.sin(2 * P) + 3 * cf[4] * np.cos(3 * P) - 3 * cf[5] * np.sin(3 * P) + cf[12])
        a_psi[i] = np.abs(dpsi).max()
        dom = cf[6] + 2 * cf[7] * W + 3 * cf[8] * W ** 2
        a_om[i] = (np.abs(dom) + abs(cf[10]) * um).max()
    return a_psi, a_om

def constants(c, coef, gbar_deg, wmax, lam_rel_pd=None):
    gbar = np.radians(gbar_deg); c0 = np.cos(gbar)
    C = Cmat(c); J1 = J1_of(c); Kd1 = np.diag(c['bg'])
    K1 = Kd1 - Kd1 @ C.T @ np.linalg.solve(J1, C @ Kd1); K1 = .5 * (K1 + K1.T)
    lam2 = np.linalg.eigvalsh(K1)[1]
    a_psi, a_om = residual_lipschitz(coef, c, gbar, wmax)
    d = np.min(c['Dphys'] / c['M'] - a_om)
    theta_bar = np.sqrt(np.max(c['M'] * a_psi ** 2 / (c['bg'] * c0)))
    J1inv = np.linalg.inv(J1)
    pd_nrm = np.sqrt(c['Pd'] @ J1inv @ c['Pd'])              # ||Pd0||_{J1^-1}
    # load -> angle sensitivity (Cauchy-Schwarz in J1^-1 inner product): sup over edges of sqrt(a_e^T J1^-1 a_e)
    lines = [np.sqrt((np.eye(c['n'])[j] - np.eye(c['n'])[k]) @ J1inv @ (np.eye(c['n'])[j] - np.eye(c['n'])[k])) for j, k in zip(c['lf'], c['lt'])]
    gens = [np.sqrt(J1inv[b, b]) for b in c['gbus']]
    Lam = max(lines + gens) / c0 * pd_nrm                       # angle change per unit relative load deviation  [rad]
    Wl1 = c['Pd'].sum() / np.sqrt(c['M'].min())                 # l1-route constant for ||M^1/2 m_l||
    Wl2 = np.sqrt(np.max(c['bg'] / c['M']) / c0) * pd_nrm        # J-route constant
    return dict(gbar=gbar, c0=c0, lam2=lam2, a_psi=a_psi, a_om=a_om, d=d, theta_bar=theta_bar, Lam=Lam, Wl=min(Wl1, Wl2), Wl1=Wl1, Wl2=Wl2,
                sqrtM=np.sqrt(c['M'].sum()), Mmin=c['M'].min(), tan=np.tan(gbar))

def tube(consts, rho, lamp, lom_max, Omega_pred, ts, eta_grid=(0.1, 0.25, 0.5, 1, 2, 4, 8), V0=0.0, fixed_omega=None):
    """Tube bound at times ts.  Two valid routes for the energy V (both use the damping d on the stiffness-perturbation term):
      route E (Young on the forcing too):  V' <= nu_E V + c_eta W^2,  nu_E = mu + (1+eta) th^2/(2d)
      route Y (forcing kept linear):       y=sqrt(V):  y' <= (mu/2 + th^2/(4d)) y + W/sqrt(2)
    W(t) nondecreasing => W<=W(t_j) on step j.  Pointwise minimum over routes/eta is valid.  Omega: self-consistent plant speed span."""
    k = consts; Mmin = k['Mmin']; ts = np.asarray(ts, float); dts = np.diff(np.r_[0.0, ts])
    Wt = rho * k['sqrtM'] + k['Wl'] * np.minimum(2 * lamp, lamp * lom_max * ts)
    def run(Om):
        mu = k['tan'] * Om; best = None
        for eta in eta_grid:
            nu = mu + (1 + eta) * k['theta_bar'] ** 2 / (2 * k['d']); cc = (1 + 1 / eta) / (4 * k['d'])
            V = V0; Vs = []
            for dt, W in zip(dts, Wt):
                V = V * np.exp(nu * dt) + cc * W * W * (np.exp(nu * dt) - 1) / nu; Vs.append(V)
            Vs = np.array(Vs); best = Vs if best is None else np.minimum(best, Vs)
        r = mu / 2 + k['theta_bar'] ** 2 / (4 * k['d']); y = np.sqrt(V0); Ys = []
        for dt, W in zip(dts, Wt):
            y = y * np.exp(r * dt) + W / np.sqrt(2) * (np.exp(r * dt) - 1) / r; Ys.append(y)
        return np.minimum(best, np.array(Ys) ** 2)
    Om = Omega_pred if fixed_omega is None else fixed_omega
    for _ in range(60 if fixed_omega is None else 1):
        Vs = run(Om); Om_new = Omega_pred + 2 * np.sqrt(2 * Vs.max() / Mmin) if fixed_omega is None else Om
        if abs(Om_new - Om) < 1e-7 * max(1, Om): break
        Om = Om_new
        if Om > 1e3: break
    span_pot = 2 * np.sqrt(Vs / (k['c0'] * k['lam2']))
    sp_vel = np.cumsum(dts * 2 * np.sqrt(2 * Vs / Mmin))
    span = np.minimum(span_pot, sp_vel)
    load_ang = k['Lam'] * np.minimum(2 * lamp, lamp * lom_max * ts)
    return dict(V=Vs, span=span, span_pot=span_pot, span_vel=sp_vel, load_ang=load_ang, ang=span + load_ang, Omega=Om)


# ================================================================== Theorem 3: centre-of-inertia (COI) reduced energy, Metzler comparison system
from scipy.linalg import expm
def phi_load(c, states, Pd_nom=None):
    """sup over given (theta,delta) states of ||P_perp M^-1/2 Kd C^T J^-1 Pd_nom||  (per unit lambda) and of max_e |a_e^T J^-1 Pd_nom| (per unit lambda)."""
    Pd_nom = c['Pd'] if Pd_nom is None else Pd_nom; C = Cmat(c); q1 = np.sqrt(c['M']); ph = 0.0; lm = 0.0
    for th, de in states:
        J = J_of(c, th, de); u = np.linalg.solve(J, Pd_nom); kd = c['bg'] * np.cos(de - th[c['gbus']])
        v = (kd * u[c['gbus']]) / q1; v = v - q1 * (q1 @ v) / (q1 @ q1); ph = max(ph, np.linalg.norm(v))
        A = np.r_[u[c['lf']] - u[c['lt']], u[c['gbus']]]; lm = max(lm, np.abs(A).max())
    return ph, lm

def constants_coi(c, coef, gbar_deg, wmax):
    K = constants(c, coef, gbar_deg, wmax)
    r = c['Dphys'] / c['M']; assert np.ptp(r) < 1e-9 * r.max(), 'COI theorem needs D_phys = d0 M'
    K['d0'] = r[0]; K['aom'] = K['a_om'].max(); K['Mmax'] = c['M'].max(); K['sumM'] = c['M'].sum(); K['sumPd'] = c['Pd'].sum()
    K['phi_rig'] = np.sqrt(max(K['Wl'] ** 2 - K['sumPd'] ** 2 / K['sumM'], 0.0))
    return K

def omega_slope_spread(c, coef, wmax, Omega):
    """delta_om >= sup |slope_i - slope_j| of d rhat/d omega across generators (states with |omega|<=wmax, speed spread <= Omega, |u_i|<=umax_i); used for the COI-removed coupling."""
    ng = c['ng']; best = 0.0
    for i in range(ng):
        for j in range(ng):
            a, b = coef[i], coef[j]
            v = (abs(a[6] - b[6]) + 2 * abs(a[7] - b[7]) * wmax + 2 * abs(b[7]) * Omega + 3 * abs(a[8] - b[8]) * wmax ** 2 + 3 * abs(b[8]) * (2 * wmax * Omega + Omega ** 2)
                 + abs(a[10]) * c['umax'][i] + abs(b[10]) * c['umax'][j])
            best = max(best, v)
    return best

def tube_coi(K, rho, phi, lamp, lom_max, Omega, ts, eps_hat0=0.0):
    """Theorem 3.  State x=(y,beta): y=sqrt(Vtilde) (COI-removed energy), beta=sqrt(sumM)|eps_c| (COI speed error).
    x' <= A x + g(t), A Metzler.  Returns span bound 2 sqrt(Vt/(c0 lam2)) and angle-bound pieces."""
    mu = K['tan'] * Omega; th = K['theta_bar']; ao = K['aom']; d0 = K['d0']; d = d0 - ao
    Mq = np.array([[-d, th / 2], [th / 2, mu / 2]]); lmax = np.linalg.eigvalsh(Mq)[-1]
    dsp = K.get('dom_spread', ao); A = np.array([[lmax, dsp / np.sqrt(2)], [np.sqrt(2) * (th + ao), -d]])
    ts = np.asarray(ts, float); x = np.zeros(2); out = []; tprev = 0.0
    for t in ts:
        lb = min(2 * lamp, lamp * lom_max * t); g = np.array([(rho * K['sqrtM'] + phi * lb) / np.sqrt(2), lb * K['sumPd'] / K['sqrtM'] + rho * K['sqrtM']])
        dt = t - tprev; Aug = np.zeros((3, 3)); Aug[:2, :2] = A; Aug[:2, 2] = g; E = expm(Aug * dt)
        x = E[:2, :2] @ x + E[:2, 2]; out.append(x.copy()); tprev = t
    out = np.array(out); Vt = out[:, 0] ** 2
    span = 2 * np.sqrt(Vt / (K['c0'] * K['lam2']))
    return dict(span=span, Vt=Vt, beta=out[:, 1], lmax=lmax, A=A)


def tube_cross(K, rho, phi, lamp, lom_max, Omega, ts, kappas=np.linspace(0.05, 6, 60)):
    """Theorem 4 (strict Lyapunov with cross term kappa*e~^T M eps~), COI-removed, D_phys = d0 M, requires a_om = 0 (otherwise the COI speed couples in).
    V_c = |z~|^2/2 + s^2/2 + kappa e~^T M eps~ ;  Q(kappa) = [[-d+kappa, (th+kappa d0 ke + kappa ke a_om)/2],[., mu/2-kappa+kappa ke th]] ;
    V_c' <= -a V_c + b sqrt(V_c) W,  a = -2 lambda_max(Q)/(1+kappa ke),  b = sqrt(2) sqrt(1+kappa^2 ke^2)/sqrt(1-kappa ke).  Pointwise minimum over kappa is valid."""
    th = K['theta_bar']; ao = K['aom']; d0 = K['d0']; d = d0 - ao; mu = K['tan'] * Omega
    ke = np.sqrt(K['Mmax'] / (K['c0'] * K['lam2'])); ts = np.asarray(ts, float); best = np.full(len(ts), np.inf); info = None
    for kap in kappas:
        if kap * ke >= 0.95: continue
        Q = np.array([[-d + kap, (th + kap * d0 * ke + kap * ke * ao) / 2], [(th + kap * d0 * ke + kap * ke * ao) / 2, mu / 2 - kap + kap * ke * th]])
        lm = np.linalg.eigvalsh(Q)[-1]; a = -2 * lm / ((1 + kap * ke) if lm < 0 else (1 - kap * ke)); b = np.sqrt(2) * np.sqrt(1 + (kap * ke) ** 2) / np.sqrt(1 - kap * ke)
        y = 0.0; tprev = 0.0; out = []
        for t in ts:
            W = rho * K['sqrtM'] + phi * min(2 * lamp, lamp * lom_max * t); dt = t - tprev
            r = -a / 2
            y = y * np.exp(r * dt) + (b * W / 2) * ((np.exp(r * dt) - 1) / r if abs(r) > 1e-12 else dt); out.append(y); tprev = t
        out = np.array(out); sp = 2 * out / np.sqrt((1 - kap * ke) * K['c0'] * K['lam2'])
        better = sp < best
        if better.any() and info is None: info = (kap, lm)
        best = np.minimum(best, sp)
    return dict(span=best, info=info)
