"""Step 2 (c),(d): COI-reduced energy bound (Theorem 3).  Variants: phi/Lam rigorous (uniform, Cauchy-Schwarz) and phi/Lam calibrated (state-sup over recorded states, hypothesis H8).
Usage: python3 check_tube2.py case9|case39 [stress] [gbar] [Omega_bar]"""
import sys; sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from env import *
import os
IDEAL = os.environ.get('IDEAL') == '1'
name = sys.argv[1]; stress = len(sys.argv) > 2 and sys.argv[2] == 'stress'; gbar = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0; OMB = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
c, th_eq, de_eq, coef = make_env(name, stress=stress, ideal=IDEAL); LAMP = 0.11 if stress else 0.08; LOM = 6.0; N = 10; ts = TS * np.arange(1, N + 1)
mode, use_res = MODES['A2']
if IDEAL: use_res = 0
cal = make_test_runs(c, th_eq, de_eq, 6, 5001, LAMP); test = make_test_runs(c, th_eq, de_eq, 10, 6001, LAMP)
def ws_state(lg, k):
    de, w, th, u = lg['de'][k], lg['w'][k], lg['th'][k], lg['u'][k]
    ms = swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['D'], c['Pm'], True, c['kappa'], c['fric']) - swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
    return np.abs(res_acc(de, w, u, th, c['gbus'], coef) - ms).max()
rho_cal = max(ws_state(lg, k) for _, lg in cal for k in range(len(lg['t']) - 1)); rho_test = max(ws_state(lg, k) for _, lg in test for k in range(len(lg['t']) - 1))
rho = max(rho_cal, rho_test)    # the stated hypothesis must hold on the tested windows
states_cal = [(lg['th'][k], lg['de'][k]) for _, lg in cal for k in range(len(lg['t']))]
states_test = [(lg['th'][k], lg['de'][k]) for _, lg in test for k in range(len(lg['t']))]
phi_c, lam_c = phi_load(c, states_cal); phi_t, lam_t = phi_load(c, states_test)
wobs = max(np.abs(lg['w']).max() for _, lg in cal + test)
K = constants_coi(c, coef, gbar, 2.0 * max(1.0, wobs)); K['dom_spread'] = min(K['aom'], omega_slope_spread(c, coef, 2.0 * max(1.0, wobs), OMB)) if os.environ.get('LIP') != 'cal' else 0.0
if os.environ.get('LIP') == 'cal':
    psi_obs = max(np.abs(de - th[c['gbus']]).max() for _, lg in cal + test for th, de in zip(lg['th'], lg['de']))
    a_psi, a_om = residual_lipschitz(coef, c, 1.2 * psi_obs, 1.2 * wobs)
    K['a_psi'], K['a_om'], K['aom'] = a_psi, a_om, a_om.max(); K['d'] = np.min(c['Dphys'] / c['M'] - a_om); K['theta_bar'] = np.sqrt(np.max(c['M'] * a_psi ** 2 / (c['bg'] * K['c0'])))
    sl = 0.0
    for _, lg in cal + test:
        for k in range(len(lg['t']) - 1):
            w_, u_ = lg['w'][k], lg['u'][k]; s_ = np.array([coef[i, 6] + 2 * coef[i, 7] * w_[i] + 3 * coef[i, 8] * w_[i] ** 2 + coef[i, 10] * u_[i] for i in range(c['ng'])]); sl = max(sl, np.ptp(s_))
    K['dom_spread_obs'] = 1.2 * sl; K['dom_spread'] = 1.2 * sl
    print('CALIBRATED Lipschitz box (psi<=%.3f rad, |omega|<=%.2f): a_psi_max=%.2f a_om_max=%.2f theta_bar=%.3f dom_spread=%.3f' % (1.2 * psi_obs, 1.2 * wobs, a_psi.max(), a_om.max(), K['theta_bar'], K['dom_spread']))
print('omega-slope spread across generators (differential part of the COI-speed feedback): %.3f vs a_om %.3f' % (K['dom_spread'], K['aom']))
print(f'{name} stress={stress} gbar={gbar} Omega_bar={OMB} lamp={LAMP}: rho_cal={rho_cal:.4g} rho_test={rho_test:.4g}; phi: rigorous={K["phi_rig"]:.3f} calibrated(cal states)={phi_c:.3f} test-state max={phi_t:.3f}; Lam: rigorous={K["Lam"]:.3f} calibrated={lam_c:.3f} test={lam_t:.3f}')
print(f'd0={K["d0"]:.3f} a_om_max={K["aom"]:.3f} (d=d0-a_om={K["d0"]-K["aom"]:.3f}, no sign requirement in Thm 3) theta_bar={K["theta_bar"]:.3f}')
phi_use = max(phi_c, phi_t); lam_use = max(lam_c, lam_t)   # calibrated variants use the larger of cal/test (hypothesis H8 must hold on tested windows)
rows = []; nwin = 0; hyp_bad = 0
for load, lg in test:
    for k in range(0, len(lg['t']) - 1 - N, 4):
        Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2])); de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
        obs = []; omx = [np.ptp(w)]; amax = np.abs(angles(c, th, de)).max()
        for j in range(N):
            for _ in range(NSUB):
                de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
            dp, wp, tp = lg['de'][k + j + 1], lg['w'][k + j + 1], lg['th'][k + j + 1]
            obs.append((np.ptp(de - dp), np.abs(angles(c, th, de) - angles(c, tp, dp)).max())); omx += [np.ptp(w), np.ptp(wp)]; amax = max(amax, np.abs(angles(c, th, de)).max(), np.abs(angles(c, tp, dp)).max())
        ok = (max(omx) <= OMB) and (amax + max(o[0] for o in obs) <= np.radians(gbar)); hyp_bad += (not ok); nwin += 1
        obs = np.array(obs); Tr = tube_coi(K, rho, K['phi_rig'], LAMP, LOM, OMB, ts); Tc = tube_coi(K, rho, phi_use, LAMP, LOM, OMB, ts)
        if IDEAL: Tx = tube_cross(K, rho, phi_use, LAMP, LOM, OMB, ts); Txr = tube_cross(K, rho, K['phi_rig'], LAMP, LOM, OMB, ts)
        lb = np.minimum(2 * LAMP, LAMP * LOM * ts)
        for j in range(N):
            rows.append((j + 1, obs[j, 0], obs[j, 1], Tr['span'][j], Tr['span'][j] + K['Lam'] * lb[j], Tc['span'][j], Tc['span'][j] + lam_use * lb[j], ok, (Txr['span'][j] + K['Lam'] * lb[j]) if IDEAL else np.nan, (Tx['span'][j] + lam_use * lb[j]) if IDEAL else np.nan))
df = pd.DataFrame(rows, columns=['j', 'span_obs', 'ang_obs', 'span_rig', 'ang_rig', 'span_cal', 'ang_cal', 'hyp_ok', 'ang_cross_rig', 'ang_cross_cal']); df.to_csv(f'results/tube2_{"ideal_" if IDEAL else ""}{"lipcal_" if os.environ.get("LIP")=="cal" else ""}{name}{"_stress" if stress else ""}.csv', index=False)
ok = df[df.hyp_ok]; print(f'windows {nwin}, hypotheses (Omega_bar, regularity level) hold on {nwin - hyp_bad}')
for v in ('rig', 'cal'): print(f'violations ({v}): span {(ok.span_obs > ok["span_" + v]).sum()}, angle {(ok.ang_obs > ok["ang_" + v]).sum()} of {len(ok)}')
env = pd.read_csv(f'results/{name}_tube_envelope{"_stress" if stress else ""}.csv').set_index('k')['eps_line_angle_rad']
print(' j | max obs span | max obs ang | span bnd rig | ang bnd rig (x max obs) | span bnd cal | ang bnd cal (x max obs) | paper conformal env')
for j in range(1, N + 1):
    s = ok[ok.j == j]; print('%2d | %.3e | %.3e | %.3e | %.3e (%5.1fx) | %.3e | %.3e (%5.1fx) | %s' % (j, s.span_obs.max(), s.ang_obs.max(), s.span_rig.median(), s.ang_rig.median(), s.ang_rig.median() / s.ang_obs.max(), s.span_cal.median(), s.ang_cal.median(), s.ang_cal.median() / s.ang_obs.max(), '%.3e' % env[j] if j in env.index else '-'))

if IDEAL:
    for v in ('cross_rig', 'cross_cal'): print(f'violations ({v}): angle {(ok.ang_obs > ok["ang_" + v]).sum()} of {len(ok)}')
    for j in (1, 3, 5, 7, 10):
        s = ok[ok.j == j]; print('cross-term bound j=%d: rig %.3e (%.1fx max obs) | cal %.3e (%.1fx max obs)' % (j, s.ang_cross_rig.median(), s.ang_cross_rig.median() / s.ang_obs.max(), s.ang_cross_cal.median(), s.ang_cross_cal.median() / s.ang_obs.max()))
