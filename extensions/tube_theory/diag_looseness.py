"""Step 1: where does the ~70x (0.5 s) looseness of Theorem 1 come from?  Sequential relaxation ladder on case9 (and case39 ideal).
For each held-out window we record the ACTUAL energy V_act, span, forcing norm ||M^-1/2 F||, load lambda(t), residual error |w_s(t)|, speeds,
then recompute the bound with ingredients replaced by their window-actual running maxima (still >= the actual signal, i.e. same rigorous form of bound)."""
import sys; sys.path.insert(0, 'code')
import numpy as np
from env import *
name = sys.argv[1]; ideal = len(sys.argv) > 2 and sys.argv[2] == 'ideal'
LAMP, LOM = 0.08, 6.0; GB = 45.0; OMB = 1.5; N = 10; ts = TS * np.arange(1, N + 1)
c, th_eq, de_eq, coef = make_env(name, ideal=ideal)
mode, use_res = MODES['A2']; use_res = 0 if ideal else use_res
K = constants(c, coef, GB, 2.4); test = make_test_runs(c, th_eq, de_eq, 10, 6001, LAMP); cal = make_test_runs(c, th_eq, de_eq, 6, 5001, LAMP)
def ws_state(lg, k):
    de, w, th, u = lg['de'][k], lg['w'][k], lg['th'][k], lg['u'][k]
    ms = swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['D'], c['Pm'], True, c['kappa'], c['fric']) - swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
    return res_acc(de, w, u, th, c['gbus'], coef) - ms
rho = max(max(np.abs(ws_state(lg, k)).max() for _, lg in cal + test for k in range(len(lg['t']) - 1)), 1e-12) if not ideal else 0.0
lf, lt, gb = c['lf'], c['lt'], c['gbus']
def Kbar_energy(de_p, de_h, Pd0):
    # Simpson average of K along the segment (rotor angles), algebraic solved at Pd0 (predictor's load)
    Ks = []
    for s in (0, .25, .5, .75, 1):
        d = de_p + s * (de_h - de_p); t = solve_alg(c, th_eq, d, Pd0); Ks.append(stiffness(c, t, d)[0])
    return (7 * Ks[0] + 32 * Ks[1] + 12 * Ks[2] + 32 * Ks[3] + 7 * Ks[4]) / 90
rows = []
for load, lg in test:
    for k in range(0, len(lg['t']) - 1 - N, 4):
        Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
        de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
        vrel, fperp, pot, eperp, sp, V, Fn, lam, wsn, om_p, om_h, angerr = [[] for _ in range(12)]
        for j in range(N):
            u = lg['u'][k + j]
            for _ in range(NSUB):
                de, w, th = pred_step(de, w, th, u, TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
            dp, wp, tp = lg['de'][k + j + 1], lg['w'][k + j + 1], lg['th'][k + j + 1]
            e = de - dp; eps = w - wp; Kb = Kbar_energy(dp, de, Pd_k)
            pot.append(e @ Kb @ e); Mw = c['M']; ept = eps - (Mw @ eps) / Mw.sum(); vrel.append(0.5 * np.sum(Mw * ept ** 2) + 0.5 * e @ Kb @ e); eperp.append(np.linalg.norm(e - e.mean())); V.append(0.5 * np.sum(c['M'] * eps ** 2) + 0.5 * e @ Kb @ e); sp.append(np.ptp(e))
            tt = lg['t'][k + j + 1]; lam.append(abs(load[0] * (np.sin(load[1] * tt + load[2]) - np.sin(load[1] * lg['t'][k] + load[2]))))
            t0 = solve_alg(c, tp, dp, Pd_k); dfl = c['bg'] * np.sin(dp - tp[gb]) - c['bg'] * np.sin(dp - t0[gb])
            ws = ws_state(lg, k + j + 1) if k + j + 1 < len(lg['t']) - 1 else np.zeros(c['ng'])
            Fv = (dfl + c['M'] * ws) / np.sqrt(c['M']); Fn.append(np.linalg.norm(Fv)); q1 = np.sqrt(c['M']); Fp = Fv - q1 * (q1 @ Fv) / (q1 @ q1); fperp.append(np.linalg.norm(Fp)); wsn.append(np.abs(ws).max())
            om_p.append(np.ptp(wp)); om_h.append(np.ptp(w))
            angerr.append(np.abs(angles(c, th, de) - angles(c, tp, dp)).max())
        rows.append(dict(vrel=np.array(vrel), fperp=np.array(fperp), pot=np.array(pot), eperp=np.array(eperp), sp=np.array(sp), V=np.array(V), F=np.array(Fn), lam=np.array(lam), ws=np.array(wsn), Om=max(max(om_p), max(om_h)), ang=np.array(angerr)))
def bound(rho_, lamfun, Om, Wl=None, Fact=None, nopert=False, ts_=ts):
    k = dict(K); 
    if nopert: k['theta_bar'] = 0.0; k['d'] = np.min(c['Dphys'] / c['M'])
    kk = k; Mmin = kk['Mmin']; dts = np.diff(np.r_[0.0, ts_])
    Wt = (rho_ * kk['sqrtM'] + kk['Wl'] * lamfun) if Fact is None else Fact
    mu = kk['tan'] * Om; best = None
    for eta in (0.1, .25, .5, 1, 2, 4, 8):
        nu = mu + (1 + eta) * kk['theta_bar'] ** 2 / (2 * kk['d']); cc = (1 + 1 / eta) / (4 * kk['d']); V = 0; Vs = []
        for dt, W in zip(dts, Wt):
            V = V * np.exp(nu * dt) + cc * W * W * (np.exp(nu * dt) - 1) / max(nu, 1e-12); Vs.append(V)
        best = np.array(Vs) if best is None else np.minimum(best, Vs)
    r = mu / 2 + kk['theta_bar'] ** 2 / (4 * kk['d']); y = 0.0; Ys = []
    for dt, W in zip(dts, Wt):
        y = y * np.exp(r * dt) + W / np.sqrt(2) * (np.exp(r * dt) - 1) / max(r, 1e-12); Ys.append(y)
    Vs = np.minimum(best, np.array(Ys) ** 2)
    return Vs, 2 * np.sqrt(Vs / (kk['c0'] * kk['lam2']))
J5 = 4  # index of 0.5 s
worst = np.minimum(2 * LAMP, LAMP * LOM * ts)
lab = ['B0 full bound (sup rho, worst-case load, Omega_bar=1.5, theta_bar, d)']
res = {lab[0]: [], 'span_obs': []}
lad = {}
def add(key, v): lad.setdefault(key, []).append(v)
for R in rows:
    add('span_obs', R['sp'][J5]); add('ang_obs', R['ang'][J5])
    Vb, sb = bound(rho, worst, OMB); add('B0 full bound', sb[J5])
    runlam = np.maximum.accumulate(R['lam'] * 1.0)
    Vb, sb = bound(rho, runlam, OMB); add('B1 + load -> window-actual running max |lambda|', sb[J5])
    runws = np.maximum.accumulate(R['ws']); 
    Wt = runws * K['sqrtM'] + K['Wl'] * runlam
    Vb, sb = bound(None, None, OMB, Fact=Wt); add('B2 + rho -> window-actual running max |w_s|', sb[J5])
    Fr = np.maximum.accumulate(R['F'])
    Vb, sb = bound(None, None, OMB, Fact=Fr); add('B3 + forcing norm -> actual running max ||M^-1/2 F|| (removes Cauchy-Schwarz/uniform Wl)', sb[J5])
    Vb, sb = bound(None, None, R['Om'], Fact=Fr); add('B4 + Omega_bar -> window-actual speed span', sb[J5])
    Vb, sb = bound(None, None, R['Om'], Fact=Fr, nopert=True); add('B5 + residual-Lipschitz perturbation theta_bar -> 0, d=Dphys/M', sb[J5])
    add('B6 energy->span conversion only: span_pot(V_act)', 2 * np.sqrt(R['V'][J5] / (K['c0'] * K['lam2'])))
print(f'{name} ideal={ideal} rho_sup={rho:.4g}  windows={len(rows)}   (all numbers at t=0.5 s, medians over windows, ratio to the window actual span error)')
so = np.array(lad['span_obs']); prev = None
print('%-100s %12s %10s %12s' % ('ingredient ladder (cumulative replacement)', 'median span', 'x obs', 'step factor'))
for key in lad:
    if key in ('span_obs', 'ang_obs'): continue
    v = np.array(lad[key]); m = np.median(v / so); st = (m / prev) if prev and key.startswith('B') and key[1] != '6' else float('nan')
    print('%-100s %12.4g %10.1f %12.2f' % (key, np.median(v), m, st))
    if key.startswith('B') and key[1] != '6': prev = m
print('median observed span %.3g, median observed angle error %.3g, median actual forcing norm %.3g vs worst-case W %.3g, median actual energy V %.3g' % (np.median(so), np.median(lad['ang_obs']), np.median([R['F'][J5] for R in rows]), rho * K['sqrtM'] + K['Wl'] * worst[J5], np.median([R['V'][J5] for R in rows])))
print('actual |lambda| at 0.5 s median %.4f vs worst-case bound %.4f ; actual max|w_s| median %.4g vs rho %.4g' % (np.median([R['lam'][J5] for R in rows]), worst[J5], np.median([R['ws'].max() for R in rows]), rho))

Fa = np.array([np.sqrt(2 * R['V'][J5] / R['pot'][J5]) for R in rows])           # (kinetic+potential)/potential : s^2 <= 2V loses this
Fb = np.array([np.sqrt(R['pot'][J5] / (K['c0'] * K['lam2'] * R['eperp'][J5] ** 2)) for R in rows])   # e is in a stiffer mode than the weakest one (lambda2)
Fc = np.array([2 / (R['sp'][J5] / R['eperp'][J5]) * 0.5 * 1.0 for R in rows])   # span <= sqrt2 |e_perp| (x sqrt2 from s^2<=2V accounted in Fa)
print('conversion split at 0.5 s (medians): sqrt(2V/s^2)=%.2f (kinetic+COI energy counted as potential), sqrt(s^2/(c0*lam2*|e_perp|^2))=%.2f (weakest-mode assumption), sqrt2|e_perp|/span=%.2f' % (np.median(Fa), np.median(Fb), np.median([np.sqrt(2) * R['eperp'][J5] / R['sp'][J5] for R in rows])))
print('kinetic share of V_act: median %.2f' % np.median([1 - R['pot'][J5] / 2 / R['V'][J5] for R in rows]))

# COI-removed prototype (proportional damping): relative energy and projected forcing
Vr = np.array([R['vrel'][J5] for R in rows]); Va = np.array([R['V'][J5] for R in rows])
print('COI removal: median V_rel/V_act = %.4f ; span_pot(V_rel)/span_obs = %.1f (vs %.1f with V_act); projected actual forcing / full forcing (median) = %.3f' % (np.median(Vr / Va), np.median(2 * np.sqrt(Vr / (K['c0'] * K['lam2'])) / so), np.median(2 * np.sqrt(Va / (K['c0'] * K['lam2'])) / so), np.median([R['fperp'][J5] / R['F'][J5] for R in rows])))
v = []
for R in rows:
    Fp = np.maximum.accumulate(R['fperp']); Vb, sb = bound(None, None, R['Om'], Fact=Fp, nopert=True); v.append(sb[J5])
print('best case (COI-removed, forcing = ACTUAL projected running max, Omega actual, no perturbation): span bound / observed span = %.1f' % np.median(np.array(v) / so))
