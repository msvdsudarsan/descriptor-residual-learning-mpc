"""Structure-preserving residual trust gate (SP-RTG) and the plant with an abrupt, unannounced operating-point shift.

The gate watches how well the full corrected predictor (physics + learned residual + Newton projection) explains the last few
measured steps, and attenuates the learned residual when it does not:  d_eff = tau * d_hat,  tau in [0, 1].  The Newton
projection stays active for every tau, so the algebraic constraint is kept even when the residual is switched off.  The gate is a
transparent heuristic and carries no guarantee.
"""
import numpy as np
from network_experiments import *

SIG_MEAS = 1e-3          # measurement noise on the signals the gate sees (same level as in the identification data)
M_SMOOTH = 3             # innovation scores are averaged over the last three samples


# ----------------------------------------------------------------------------- plant with an abrupt shift
def simulate_shift(c0, c1, t_shift, th0, de0, w0, controller, T, load, hp=0.0125, Ts=TS, t_restore=None):
    """True plant; at t_shift the plant switches from case c0 to case c1 (load and dispatch scale together, as in Table 5), and
    back to c0 at t_restore if given.  controller(t, delta, w, theta, u_prev, Pd_now) -> u.  The algebraic variables are re-solved at each switch."""
    lamp, lom, lph = load
    ng = c0['ng']; nsteps = int(round(T / Ts)); nsub = int(round(Ts / hp))
    de, w, th = de0.copy(), w0.copy(), th0.copy()
    u_prev = np.zeros(ng); t = 0.0; cur = c0
    logs = dict(t=[], de=[], w=[], th=[], u=[], Pd=[], solve=[])
    for k in range(nsteps):
        cp = c1 if (t >= t_shift - 1e-9 and (t_restore is None or t < t_restore - 1e-9)) else c0
        Pd_now = cp['Pd'] * (1 + lamp * np.sin(lom * t + lph))
        if cp is not cur:
            th, _ = solve_theta(th, de, Pd_now, *args_of(cp), 1e-12, 30); cur = cp
        t0 = time.perf_counter()
        u = controller(t, de, w, th, u_prev, Pd_now)
        logs['solve'].append(time.perf_counter() - t0)
        for key, val in (('t', t), ('de', de.copy()), ('w', w.copy()), ('th', th.copy()), ('u', u.copy()), ('Pd', Pd_now.copy())):
            logs[key].append(val)
        for _ in range(nsub):
            de, w, th = plant_step(de, w, th, u, hp, t, cp['Pd'], lamp, lom, lph, cp['lf'], cp['lt'], cp['lb'], cp['gbus'], cp['bg'], cp['M'], cp['D'], cp['Pm'], cp['kappa'], cp['fric'])
            t += hp
        u_prev = u
    for key, val in (('t', t), ('de', de.copy()), ('w', w.copy()), ('th', th.copy()), ('u', u_prev.copy()), ('Pd', Pd_now.copy())):
        logs[key].append(val)
    return {k: (np.array(v) if k != 'solve' else float(np.mean(v))) for k, v in logs.items()}


# ----------------------------------------------------------------------------- gate
def innovation_vec(c, coef, Pm_pred, y_prev, u_prev, Pd_prev, y_now):
    """One-step innovation of the full corrected predictor between two measured samples; y = (delta, omega).
    Angles are taken relative to the centre of inertia, since the neutral angle shift is irrelevant."""
    de_p, w_p = y_prev; de_n, w_n = y_now
    de_p = de_p - np.sum(c['M'] * de_p) / np.sum(c['M'])
    th_p, _ = solve_theta(np.zeros(c['n']), de_p, Pd_prev, *args_of(c), 1e-10, 40)
    de, w, th = de_p.copy(), w_p.copy(), th_p
    for _ in range(NSUB):
        de, w, th = pred_step(de, w, th, u_prev, TS / NSUB, Pd_prev, 1, 1, coef, *args_of(c), c['M'], c['Dphys'], Pm_pred, c['kappa'], c['fric'])
    coi_p = np.sum(c['M'] * de) / np.sum(c['M']); coi_n = np.sum(c['M'] * de_n) / np.sum(c['M'])
    return np.r_[(de - coi_p) - (de_n - coi_n), w - w_n]


def innovation_score(c, coef, Pm_pred, y_prev, u_prev, Pd_prev, y_now):
    return float(np.sqrt(np.mean(innovation_vec(c, coef, Pm_pred, y_prev, u_prev, Pd_prev, y_now) ** 2)))


class Reconciler:
    """Trust gate with optional dispatch reconciliation.  The predictor's dispatch is rho * Pm_nominal.  After a detection (q > 1) the scale rho is
    re-estimated by one Gauss-Newton step on the flagged one-step innovations, so that the innovation of the corrected predictor shrinks again and
    the trust returns.  With reconcile=False rho stays at one and the object is the plain trust gate."""

    def __init__(self, c, coef, q0, a, reconcile, drho=0.05, rho_lim=(0.4, 2.5)):
        self.c, self.coef, self.q0, self.a, self.reconcile = c, coef, q0, a, reconcile
        self.rho = 1.0; self.s = []; self.buf = []; self.drho, self.lim = drho, rho_lim

    def step(self, y_prev, y_now, u_prev, Pd_prev):
        c = self.c
        e = innovation_vec(c, self.coef, self.rho * c['Pm'], y_prev, u_prev, Pd_prev, y_now)
        self.s.append(float(np.sqrt(np.mean(e ** 2)))); self.buf.append((y_prev, y_now, u_prev, Pd_prev))
        q = float(np.mean(self.s[-M_SMOOTH:]) / self.q0)
        tau = trust(q, 0.0, self.a, 0.0)
        if self.reconcile and q > 1.0:
            num = den = 0.0
            for j in range(max(0, len(self.buf) - M_SMOOTH), len(self.buf)):
                yp, yn, uu, pd_ = self.buf[j]
                e0 = innovation_vec(c, self.coef, self.rho * c['Pm'], yp, uu, pd_, yn)
                if np.sqrt(np.mean(e0 ** 2)) <= self.q0: continue
                e1 = innovation_vec(c, self.coef, (self.rho + self.drho) * c['Pm'], yp, uu, pd_, yn)
                g = (e1 - e0) / self.drho; num += float(g @ e0); den += float(g @ g)
            if den > 0: self.rho = float(np.clip(self.rho - num / den, *self.lim))
        return tau, q, self.rho


def reconcile_series(c, coef, lg, q0, a, reconcile, rng, sig=SIG_MEAS):
    """tau_k, q_k and the dispatch scale rho_k for every sample of a logged run (noisy measurements, data up to sample k only)."""
    K = len(lg['t']); de_m = lg['de'] + rng.normal(0, sig, lg['de'].shape); w_m = lg['w'] + rng.normal(0, sig, lg['w'].shape)
    rec = Reconciler(c, coef, q0, a, reconcile); tau = np.ones(K); q = np.zeros(K); rho = np.ones(K)
    for k in range(1, K):
        tau[k], q[k], rho[k] = rec.step((de_m[k - 1], w_m[k - 1]), (de_m[k], w_m[k]), lg['u'][k - 1], lg['Pd'][k - 1])
    return tau, q, rho


def regularity_score(c, de_meas, lam_ref):
    de_meas = de_meas - np.sum(c['M'] * de_meas) / np.sum(c['M'])
    th, _ = solve_theta(np.zeros(c['n']), de_meas, c['Pd'], *args_of(c), 1e-10, 40)
    lam = regularity(c, th, de_meas, Pd=c['Pd'])[0]
    return float(max(0.0, 1.0 - lam / lam_ref))


def trust(q, r, a, b):
    return float(np.clip(np.exp(-a * max(q - 1.0, 0.0) - b * r), 0.0, 1.0))


def gate_series(c, coef, Pm_pred_fn, lg, q0, a, b, lam_ref, rng, sig=SIG_MEAS):
    """tau_k, q_k, r_k for every sample of a logged run, using noisy measurements and only data up to sample k."""
    K = len(lg['t']); de_m = lg['de'] + rng.normal(0, sig, lg['de'].shape); w_m = lg['w'] + rng.normal(0, sig, lg['w'].shape)
    s = np.full(K, np.nan); q = np.zeros(K); r = np.zeros(K); tau = np.ones(K)
    for k in range(1, K):
        cc = dict(c); cc['Pd'] = lg['Pd'][k - 1]
        s[k] = innovation_score(c, coef, Pm_pred_fn(lg['t'][k - 1]), (de_m[k - 1], w_m[k - 1]), lg['u'][k - 1], lg['Pd'][k - 1], (de_m[k], w_m[k]))
        q[k] = np.mean(s[max(1, k - M_SMOOTH + 1):k + 1]) / q0
        cc = dict(c); cc['Pd'] = lg['Pd'][k]
        r[k] = regularity_score(cc, de_m[k], lam_ref)
        tau[k] = trust(q[k], r[k], a, b)
    return tau, q, r, s
