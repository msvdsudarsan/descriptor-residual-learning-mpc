"""T-A numerical test of the tube theorem (THEORY.md Thm 2).
Usage: python3 check_tube.py case9|case39 [stress] [gbar_deg]
Calibration runs (seed A) fix rho (sup over recorded plant states of max_i |rhat_i(x_p) - m_s,i|).  Held-out runs (seed B) are the test windows.
Reports: assumption checks, violations (must be 0), tightness ratios, comparison with the repo's conformal envelope."""
import sys; sys.path.insert(0, 'code')
import numpy as np
from tubelib import *
from network_experiments import (make_training, fit_net, make_test_runs, TS, NSUB, MODES, STRESS, static_ctrl, mpc_log_runs, simulate_plant)
import pandas as pd
name = sys.argv[1]; stress = len(sys.argv) > 2 and sys.argv[2] == 'stress'; gbar_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
mu_load = STRESS[name] if stress else 1.0
import os
IDEAL = os.environ.get('IDEAL') == '1'   # plant == physics model except for the load fluctuation (no saliency, no friction, exact damping); predictor = physics only
c = build_case(name, load_scale=mu_load, **(dict(saliency=0.0, fric=0.0, phys_damp=1.0) if IDEAL else {})); th_eq, de_eq, res = equilibrium(c); ng = c['ng']
train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
coef, _, _ = fit_net(c, train, val, th_eq)
if IDEAL: coef = np.zeros_like(coef)
LAMP = 0.11 if stress else 0.08; LOM_MAX = 6.0
mode, use_res = MODES['A2']
if IDEAL: use_res = 0
def rho_states(runs):
    out = []
    for load, lg in runs:
        for k in range(len(lg['t']) - 1):
            de, w, th, u = lg['de'][k], lg['w'][k], lg['th'][k], lg['u'][k]
            ms = swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['D'], c['Pm'], True, c['kappa'], c['fric']) - swing_acc(de, w, u, th, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])
            rh = res_acc(de, w, u, th, c['gbus'], coef)
            out.append(np.abs(rh - ms).max())
    return np.array(out)
cal = make_test_runs(c, th_eq, de_eq, 6, 5001, LAMP)
test = make_test_runs(c, th_eq, de_eq, 10, 6001, LAMP)
rho_cal = rho_states(cal).max(); rho_test = rho_states(test)
wmax_obs = max(np.abs(lg['w']).max() for _, lg in cal + test)
amax_obs = max(np.abs(angles(c, t, d)).max() for _, lg in cal + test for t, d in zip(lg['th'], lg['de']))
print('IDEAL (no model mismatch, only load fluctuation):', IDEAL)
print(f'{name} stress={stress} load_scale={mu_load} gbar={gbar_deg} deg   ng={ng}')
print(f'rho_cal(max over calibration states)={rho_cal:.4g} rad/s^2 ; test-state max={rho_test.max():.4g}  (assumption rho >= test max: {"HOLDS" if rho_test.max()<=rho_cal else "FAILS on test data, using test max for theorem check"})')
rho = max(rho_cal, rho_test.max())     # use the larger so that the assumption holds on every tested window; reported separately
print(f'observed max |omega| = {wmax_obs:.3f} rad/s ; observed max constrained angle = {np.degrees(amax_obs):.2f} deg')
K = constants(c, coef, gbar_deg, wmax=2.0 * max(1.0, wmax_obs))
print('constants: c0=%.3f lam2(K1)=%.3f d=%.3f theta_bar=%.3f Lam(load->angle per unit lambda)=%.4f rad  Wl=%.4g (l1 %.3g, J %.3g) sqrtM=%.3f Mmin=%.4g' % (K['c0'], K['lam2'], K['d'], K['theta_bar'], K['Lam'], K['Wl'], K['Wl1'], K['Wl2'], K['sqrtM'], K['Mmin']))
print('residual Lipschitz: a_psi=', np.round(K['a_psi'], 3), ' a_om=', np.round(K['a_om'], 3), ' Dphys/M=', np.round(c['Dphys'] / c['M'], 3))
OMEGA_BAR = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
N = 10; ts = TS * np.arange(1, N + 1); rows = []; bad_assump = 0; nwin = 0
lf, lt, gb = c['lf'], c['lt'], c['gbus']
for load, lg in test:
    Kk = len(lg['t']) - 1
    for k in range(0, Kk - N, 4):
        Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
        de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
        obs = []; om_pred = [np.ptp(w)]; omp_plant = [np.ptp(w)]; ang_max = np.abs(angles(c, th, de)).max()
        for j in range(N):
            u = lg['u'][k + j]
            for _ in range(NSUB):
                de, w, th = pred_step(de, w, th, u, TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
            de_t, w_t, th_t = lg['de'][k + j + 1], lg['w'][k + j + 1], lg['th'][k + j + 1]
            dd = de - de_t; span = np.ptp(dd); einf = np.abs(dd - dd.mean()).max()
            ang_err = np.abs(angles(c, th, de) - angles(c, th_t, de_t)).max()
            obs.append((span, ang_err)); om_pred.append(np.ptp(w)); omp_plant.append(np.ptp(w_t))
            ang_max = max(ang_max, np.abs(angles(c, th, de)).max(), np.abs(angles(c, th_t, de_t)).max())
        lam_bar_ok = True
        T = tube(K, rho, LAMP, LOM_MAX, max(om_pred), ts, fixed_omega=OMEGA_BAR)
        T_boot = tube(K, rho, LAMP, LOM_MAX, max(om_pred), ts)
        obs = np.array(obs)
        # assumption checks (post hoc): plant speed span <= Omega, |omega|<=wmax, angles + bound <= gbar
        a_ok = (max(max(omp_plant), max(om_pred)) <= OMEGA_BAR) and (ang_max + max(o[0] for o in obs) <= np.radians(gbar_deg))
        bad_assump += (not a_ok); nwin += 1
        rows.append(np.c_[np.arange(1, N + 1), obs[:, 0], obs[:, 1], T['span'], T['ang'], T['span_pot'], T['span_vel'], T['load_ang'], np.full(N, a_ok), np.full(N, T_boot['span'][-1])])
R = np.vstack(rows); cols = ['j', 'span_obs', 'ang_obs', 'span_bnd', 'ang_bnd', 'span_pot', 'span_vel', 'load_ang', 'assump_ok', 'span_bootstrap_final']
df = pd.DataFrame(R, columns=cols); df.to_csv(f'results/tube_check_{"ideal_" if IDEAL else ""}{name}{"_stress" if stress else ""}.csv', index=False)
ok = df[df.assump_ok == 1]
print(f'Omega_bar (hypothesis: plant and predicted speed span <= Omega_bar) = {OMEGA_BAR}')
print(f'windows {nwin}; windows where the hypotheses (speed span <= Omega_bar, angle level + span(e) <= gbar) hold on the observed trajectories: {nwin - bad_assump}')
print('self-consistent (bootstrapped) Omega bound finite? median final-step span bound with bootstrap =', df.span_bootstrap_final.median())
print('violations among assumption-satisfying windows:  span_obs > span_bnd: %d ; ang_obs > ang_bnd: %d  (of %d samples)' % ((ok.span_obs > ok.span_bnd).sum(), (ok.ang_obs > ok.ang_bnd).sum(), len(ok)))
print('violations over ALL windows:  span %d ; angle %d' % ((df.span_obs > df.span_bnd).sum(), (df.ang_obs > df.ang_bnd).sum()))
env = None
try:
    env = pd.read_csv(f'results/{name}_tube_envelope{"_stress" if stress else ""}.csv').set_index('k')['eps_line_angle_rad']
except Exception as e: print('no envelope', e)
print(' j | t[s] | med obs span | med span bound | ratio(max obs vs bound) | med obs ang | max obs ang | ang bound (med) | bound/maxobs | conformal envelope (paper)')
for j in range(1, N + 1):
    s = ok[ok.j == j]
    print('%2d | %.1f | %.3e | %.3e | %7.1f | %.3e | %.3e | %.3e | %7.1f | %s' % (j, j * TS, s.span_obs.median(), s.span_bnd.median(), s.span_bnd.median() / s.span_obs.max(), s.ang_obs.median(), s.ang_obs.max(), s.ang_bnd.median(), s.ang_bnd.median() / s.ang_obs.max(), ('%.3e' % env[j]) if env is not None and j in env.index else '-'))
print('component split at j=5: pot-route span %.3e, velocity-route span %.3e, load-angle term %.3e' % (ok[ok.j == 5].span_pot.median(), ok[ok.j == 5].span_vel.median(), ok[ok.j == 5].load_ang.median()))
