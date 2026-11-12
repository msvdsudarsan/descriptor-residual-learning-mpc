"""Worked numerical example on MATPOWER case39 (Section 'Application' of the manuscript), step by step; same six steps as worked_example_case9.py.

Run:  python code/worked_example_case39.py          (a few minutes in pure Python, faster with numba)
Output: printed log, results/worked_example_case39.json (every number quoted in the manuscript) and
        results/worked_example_case39_prediction.csv (state error of A0, A1, A2 along the one-second window).

Steps
  1  system data and nominal equilibrium on the constraint manifold g = 0
  2  regularity of the algebraic Jacobian J and the eigenvalue bound of Proposition 2
  3  one weak-form regression equation (generator 1) and the stored residual model
  4  Newton projection: algebraic residual after 0, 1, 2, 3 iterations
  5  one-second prediction window: physics only (A0), residual without projection (A1), residual with projection (A2)
  6  trust gate and dispatch reconciliation after an unannounced 25% step of load and dispatch
All random numbers use fixed seeds.  The residual coefficients are those stored in results/coef_nominal_case39.npy
(fitted on noisy nominal-loading trajectories by network_experiments.py); nothing is fitted here.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_experiments as ne            # noqa: E402
from network_benchmark import (build_case, equilibrium, regularity, alg_res_jac, solve_theta, tangent_theta,
                               swing_acc, res_acc, feats, NF)       # noqa: E402
import network_trust as nt                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
Q0_CASE39 = 0.03706069563617833              # calibrated gate level, results/validity_gate_case39_log.txt
A_GATE = 0.25                               # gate constant chosen on validation scenarios (manuscript, Section 'Trust gate')


def coi(c, de):
    return np.sum(c['M'] * de) / np.sum(c['M'])


def compute(verbose=True):
    out = {}
    say = print if verbose else (lambda *a, **k: None)
    c = build_case('case39')
    args = ne.args_of(c)
    coef = np.load(RES / 'coef_nominal_case39.npy')

    # ---- step 1: system data and equilibrium
    th, de, res = equilibrium(c)
    out['n_bus'], out['n_gen'] = int(c['n']), int(c['ng'])
    out['gen_buses'] = [int(b) + 1 for b in c['gbus']]
    out['Pm_pu'] = [round(float(x), 4) for x in c['Pm']]
    out['Pd_total_pu'] = round(float(c['Pd'].sum()), 4)
    out['delta_eq_deg'] = [round(float(x), 2) for x in np.degrees(de)]
    out['rotor_angle_deg'] = [round(float(x), 2) for x in np.degrees(de - th[c['gbus']])]
    out['eq_residual'] = float(res)
    say('1  case39: n=%d buses, %d generators at buses %s; sum Pm = sum Pd = %.4f pu' % (c['n'], c['ng'], out['gen_buses'], c['Pd'].sum()))
    say('   equilibrium residual %.1e; rotor angles psi_i [deg] = %s' % (res, out['rotor_angle_deg']))

    # ---- step 2: regularity
    lam_J, cond_J, amax, gmax = regularity(c, th, de)
    _, B = alg_res_jac(np.zeros(c['n']), np.zeros(c['ng']), c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], True)
    lam_B = float(np.linalg.eigvalsh(B).min())
    gam = np.radians(30.0)
    bound = float(np.cos(gam) * lam_B)
    out.update(lam_min_J=round(lam_J, 3), cond_J=round(cond_J, 1), lam_min_B=round(lam_B, 3), lam_bound=round(bound, 3),
               Jinv_norm_bound=round(1.0 / bound, 3), Jinv_norm=round(1.0 / lam_J, 3),
               max_line_angle_deg=round(amax, 2), max_rotor_angle_deg=round(gmax, 2))
    say('2  lambda_min(J)=%.3f, bound cos(30deg)*lambda_min(B)=%.3f, so ||J^-1||_2=%.3f <= %.3f; largest line angle %.1f deg, rotor %.1f deg'
        % (lam_J, bound, 1 / lam_J, 1 / bound, amax, gmax))

    # ---- step 3: one weak-form equation for generator 1 and the stored model
    tr = ne.make_training(c, th, de, 1, seed=11, noise=1e-3)[0]
    TH, Y = ne.weak_form_net(c, tr, th)                       # (K, ng, NF), (K, ng)
    i, k = 0, 12
    pred = float(TH[k, i] @ coef[i, :NF])
    y_all, p_all = Y[:, i], TH[:, i, :] @ coef[i, :NF]
    act = [(j, float(coef[i, j])) for j in range(NF) if coef[i, j] != 0.0]
    out.update(wf_h=float(tr['h']), wf_k=k, wf_target=round(float(Y[k, i]), 4), wf_model=round(pred, 4),
               wf_rms_target=round(float(np.sqrt(np.mean(y_all ** 2))), 3), wf_rms_resid=round(float(np.sqrt(np.mean((y_all - p_all) ** 2))), 3),
               wf_n_samples=int(len(y_all)), wf_active_terms_gen1=len(act), wf_clip=round(float(coef[i, NF]), 3))
    say('3  weak-form equation (generator 1, sample %d): target %.4f, model %.4f; over %d samples RMS %.3f -> %.3f; %d active terms'
        % (k, Y[k, i], pred, len(y_all), out['wf_rms_target'], out['wf_rms_resid'], len(act)))

    # ---- step 4: Newton projection trace from the tangent update
    rng = np.random.default_rng(5)
    d0 = de + rng.normal(0, 0.10, c['ng'])
    th0, _ = solve_theta(th, de, c['Pd'], *args, 1e-13, 30)
    d1 = d0 + rng.normal(0, 0.10, c['ng'])
    g0, J0 = alg_res_jac(th0, de, c['Pd'], *args[:3], c['gbus'], c['bg'], True)
    t_tan = tangent_theta(th0, np.linalg.inv(J0), de, d1, c['gbus'], c['bg'])
    trace = []
    t_k = t_tan.copy()
    for it in range(4):
        g, J = alg_res_jac(t_k, d1, c['Pd'], *args[:3], c['gbus'], c['bg'], True)
        trace.append(float(np.max(np.abs(g))))
        t_k = t_k + np.linalg.solve(J, g)
    out['newton_trace'] = [float('%.2e' % x) for x in trace]
    out['dpsi_deg'] = round(float(np.degrees(np.max(np.abs(d1 - de)))), 2)
    say('4  Newton projection after a rotor-angle change of up to %.1f deg: max|g| = %s' % (out['dpsi_deg'], ['%.1e' % x for x in trace]))

    # ---- step 5: one-second prediction window
    load, lg = ne.make_test_runs(c, th, de, 1, seed=21, lamp_hi=0.08)[0]
    N = 10
    rows = []
    errs = {}
    gfin = {}
    for name in ('A0', 'A1', 'A2'):
        mode, use_res = ne.MODES[name]
        d, w, t = lg['de'][0].copy(), lg['w'][0].copy(), lg['th'][0].copy()
        Pd0 = c['Pd'] * (1 + load[0] * np.sin(load[2]))
        e_k = []
        for j in range(N):
            for _ in range(ne.NSUB):
                d, w, t = ne.pred_step(d, w, t, lg['u'][j], ne.TS / ne.NSUB, Pd0, mode, use_res, coef, *args, c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
            dt, wt = lg['de'][j + 1], lg['w'][j + 1]
            e = np.r_[(d - coi(c, d)) - (dt - coi(c, dt)), w - wt]
            e_k.append(float(np.sqrt(np.mean(e ** 2))))
        g, _ = alg_res_jac(t, d, Pd0, *args[:3], c['gbus'], c['bg'], False)
        errs[name] = e_k
        gfin[name] = float(np.max(np.abs(g)))
    for j in range(N):
        rows.append((round((j + 1) * ne.TS, 1), errs['A0'][j], errs['A1'][j], errs['A2'][j]))
    with open(RES / 'worked_example_case39_prediction.csv', 'w') as f:
        f.write('t_s,rmse_A0,rmse_A1,rmse_A2\n')
        for r in rows:
            f.write('%.1f,%.5f,%.5f,%.5f\n' % r)
    out.update(load_amp=round(float(load[0]), 3), pred_A0_1s=round(errs['A0'][-1], 3), pred_A1_1s=round(errs['A1'][-1], 3), pred_A2_1s=round(errs['A2'][-1], 3),
               pred_A0_05s=round(errs['A0'][4], 3), pred_A2_05s=round(errs['A2'][4], 3),
               pred_ratio_1s=round(errs['A2'][-1] / errs['A0'][-1], 2),
               g_A0=float('%.1e' % gfin['A0']), g_A1=float('%.1e' % gfin['A1']), g_A2=float('%.1e' % gfin['A2']))
    say('5  one-second state RMSE: A0 %.3f, A1 %.3f, A2 %.3f (ratio A2/A0 %.2f); max|g| at 1 s: %.1e, %.1e, %.1e'
        % (errs['A0'][-1], errs['A1'][-1], errs['A2'][-1], out['pred_ratio_1s'], gfin['A0'], gfin['A1'], gfin['A2']))

    # ---- step 6: gate and reconciliation after an unannounced 25% step
    c1 = build_case('case39', load_scale=1.25)
    rng = np.random.default_rng(7)
    de0 = de + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
    th_s, _ = solve_theta(th, de0, c['Pd'], *args, 1e-12, 30)
    T_SHIFT, T_END = 1.0, 2.0
    lg2 = nt.simulate_shift(c, c1, T_SHIFT, th_s, de0, w0, ne.static_ctrl(c), T_END, (0.02, 3.0, 0.5))
    series = {}
    for name, rec in (('gate', False), ('reconciled', True)):
        tau, q, rho = nt.reconcile_series(c, coef, lg2, Q0_CASE39, A_GATE, rec, np.random.default_rng(1))
        series[name] = (tau, q, rho)
    t = lg2['t']
    pre = (t > 0) & (t < T_SHIFT - 1e-9)
    after = np.where(t >= T_SHIFT - 1e-9)[0]
    first_flag = next((int(j) for j in after if series['gate'][1][j] > 1.0), None)
    out.update(q0=round(Q0_CASE39, 4), a_gate=A_GATE,
               q_max_before=round(float(series['gate'][1][pre].max()), 2), tau_min_before=round(float(series['gate'][0][pre].min()), 2),
               gate_delay_s=round(float(t[first_flag] - T_SHIFT), 2) if first_flag is not None else None,
               q_first=round(float(series['gate'][1][first_flag]), 1) if first_flag is not None else None,
               tau_first=round(float(series['gate'][0][first_flag]), 2) if first_flag is not None else None,
               tau_min_after=round(float(series['gate'][0][after].min()), 2),
               rho_final=round(float(series['reconciled'][2][-5:].mean()), 3),
               tau_final_reconciled=round(float(series['reconciled'][0][-5:].mean()), 2),
               q_final_reconciled=round(float(series['reconciled'][1][-5:].mean()), 2),
               tau_final_gate=round(float(series['gate'][0][-5:].mean()), 2))
    say('6  step at %.1f s: q below %.2f and tau >= %.2f before it; first flag after %.2f s (q=%.1f, tau=%.2f); lowest tau %.2f; reconciled rho = %.3f (true 1.25), '
        'final q %.2f, tau %.2f; plain gate stays at tau %.2f'
        % (T_SHIFT, out['q_max_before'], out['tau_min_before'], out['gate_delay_s'] or -1, out['q_first'] or -1, out['tau_first'] or -1,
           out['tau_min_after'], out['rho_final'], out['q_final_reconciled'], out['tau_final_reconciled'], out['tau_final_gate']))
    return out


if __name__ == '__main__':
    o = compute(True)
    with open(RES / 'worked_example_case39.json', 'w') as f:
        json.dump(o, f, indent=1)
    print('wrote results/worked_example_case39.json')
