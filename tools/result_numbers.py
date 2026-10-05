"""Collects every number quoted in the manuscript from results/*.csv|json (single source of truth)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / 'results'


def f(x, d=3):
    return f'{x:.{d}f}'


def sci(x, d=1):
    m, e = f'{x:.{d}e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def kv(name):
    d = pd.read_csv(R / name)
    return dict(zip(d.iloc[:, 0], d.iloc[:, 1]))


def collect():
    V = {}
    s = json.load(open(R / 'scalar_summary.json'))
    fit = kv('scalar_residual_fit.csv')
    V['s_rms_ex'] = f(fit['residual_rms_true_test']); V['s_rms_err'] = f(fit['residual_rms_error_after_fit'])
    V['s_red'] = f(fit['error_reduction_percent'], 0)
    V['s_rho_pt'] = f(fit['rho_pointwise_95']); V['s_cov_pt'] = f(fit['coverage_pointwise_test']); V['s_rho_blk'] = f(fit['rho_block_95']); V['s_cov_blk'] = f(fit['coverage_block_test']); V['s_nblk_test'] = int(fit['n_test_blocks'])
    cx = pd.read_csv(R / 'scalar_complexity.csv').set_index('threshold')
    V['s_cx_full'] = f(cx.loc[0.1, 'heldout_residual_rmse'], 4); V['s_cx_sparse'] = f(cx.loc[0.2, 'heldout_residual_rmse'], 4)
    cv = pd.read_csv(R / 'scalar_envelope_coverage.csv').set_index('test_source')
    V['s_covjoint_static'] = f(cv.loc['static law', 'cover_joint_all_k'], 3); V['s_covjoint_mpc'] = f(cv.loc['MPC (A2)', 'cover_joint_all_k'], 3)
    V['s_act'] = int(fit['active_terms']); V['s_lib'] = int(fit['library_terms']); V['s_deg'] = int(fit['library_degree'])
    ab = pd.read_csv(R / 'scalar_prediction_ablation.csv')
    for m in ('A0', 'A1', 'A2'):
        for h in (0.6, 1.2):
            V[f's_pred_{m}_{h}'] = f(ab[(ab.model == m) & (np.isclose(ab.horizon_s, h))].state_rmse.iloc[0])
    g = pd.read_csv(R / 'scalar_algebraic_residual.csv').set_index('model')
    for m in ('A0', 'A1', 'A2'):
        V[f's_g_{m}'] = sci(g.loc[m, 'mean_max_abs_g']); V[f's_gmax_{m}'] = sci(g.loc[m, 'max_abs_g'])
    pr = s['pred_ratio']; V['s_ratio'] = f(pr[0], 2); V['s_ratio_lo'] = f(pr[1], 2); V['s_ratio_hi'] = f(pr[2], 2)
    cl = pd.read_csv(R / 'scalar_closed_loop.csv').set_index('controller')
    for c in cl.index:
        for k in ('iae', 'last', 'viol', 'dpeak', 'solve_ms'):
            V[f's_cl_{c}_{k}'] = f(cl.loc[c, f'{k}_mean'], 3 if k != 'solve_ms' else 1)
        V[f's_cl_{c}_iae_ci'] = f'[{f(cl.loc[c, "iae_lo"], 3)}, {f(cl.loc[c, "iae_hi"], 3)}]'
    V['s_iae_drop'] = f(100 * (1 - cl.loc['A0', 'iae_mean'] / cl.loc['C0', 'iae_mean']), 0)
    pp = pd.read_csv(R / 'scalar_closed_loop_paired_vs_A0.csv')
    for _, r in pp.iterrows():
        V[f's_pair_{r.controller}_{r.metric}'] = f'{r.mean_diff:+.4f}\\ [{r.lo:+.4f},\\,{r.hi:+.4f}]'
    ev = s['eig_nom']; V['s_deq'] = f(s['delta_eq_nom']); V['s_eig_re'] = f(abs(ev[0][0]), 2); V['s_eig_im'] = f(abs(ev[0][1]), 2)
    V['s_basin_fb'] = f(100 * s['basin'][0], 0); V['s_basin_ol'] = f(100 * s['basin'][1], 0)
    te = pd.read_csv(R / 'scalar_tube_envelope.csv'); V['s_eps_d6'] = f(te.eps_delta.iloc[6]); V['s_eps_w6'] = f(te.eps_omega.iloc[6]); V['s_eps_d12'] = f(te.eps_delta.iloc[12])
    # network
    ns = json.load(open(R / 'network_summary.json'))
    for c in ('case9', 'case39'):
        sp = kv(f'{c}_spectrum.csv'); V[f'{c}_lam'] = f(sp['lambda_min_J_nominal'], 2); V[f'{c}_cond'] = f(sp['cond_J_nominal'], 1)
        V[f'{c}_lam0'] = f(sp['lambda_min_Lb_plus_G'], 2); V[f'{c}_cert'] = f(sp['cert_lower_bound_at_30deg'], 2)
        V[f'{c}_amax'] = f(sp['max_line_angle_deg_nominal'], 1); V[f'{c}_mu'] = f(sp['largest_continued_load_scale'], 3)
        V[f'{c}_zeta'] = f(sp['slowest_damping_ratio'], 2); V[f'{c}_f1'] = f(sp['lowest_mode_freq_hz'], 2); V[f'{c}_f2'] = f(sp['highest_mode_freq_hz'], 2)
        V[f'{c}_nb'] = int(sp['buses']); V[f'{c}_ng'] = int(sp['generators'])
        V[f'{c}_eqres'] = sci(sp['equilibrium_residual'])
        rf = kv(f'{c}_nom_residual_fit.csv')
        V[f'{c}_rms_ex'] = f(rf['residual_rms_true_test'], 2); V[f'{c}_rms_err'] = f(rf['residual_rms_error_after_fit'], 3); V[f'{c}_red'] = f(rf['error_reduction_percent'], 1)
        V[f'{c}_act'] = int(rf['active_terms_total']); V[f'{c}_avail'] = int(rf['terms_available'])
        cxn = pd.read_csv(R / f'{c}_complexity.csv').set_index('threshold')
        V[f'{c}_cx_hi_act'] = int(cxn.loc[1.0, 'active_terms']); V[f'{c}_cx_hi_red'] = f(cxn.loc[1.0, 'error_reduction_percent'], 1); V[f'{c}_cx_lo_red'] = f(cxn.loc[0.002, 'error_reduction_percent'], 1)
        oo = kv(f'{c}_out_of_library.csv'); V[f'{c}_ool_red'] = f(oo['error_reduction_percent'], 1); V[f'{c}_ool_ratio'] = f(oo['pred_ratio_A2_over_A0'], 2); V[f'{c}_ool_ci'] = f"[{f(oo['pred_ratio_lo'], 2)}, {f(oo['pred_ratio_hi'], 2)}]"
        ec = pd.read_csv(R / f'{c}_envelope_coverage.csv').set_index('test_source')
        V[f'net_cov{c[4:]}_static'] = f(ec.loc['static law', 'cover_k5'], 2); V[f'net_cov{c[4:]}_mpc'] = f(ec.loc['MPC (A2)', 'cover_k5'], 2); V[f'net_cov{c[4:]}_cl'] = f(ec.loc['closed-loop A2 realizations', 'cover_k5'], 2)
        V['net_ncal'] = int(ec.loc['static law', 'n_traj'] + ec.loc['MPC (A2)', 'n_traj'])
        a = pd.read_csv(R / f'{c}_prediction_ablation.csv')
        for m in ('A0', 'A1', 'A2'):
            for h in (0.5, 1.0):
                V[f'{c}_pred_{m}_{h}'] = f(a[(a.model == m) & np.isclose(a.horizon_s, h)].state_rmse.iloc[0], 3)
        g = pd.read_csv(R / f'{c}_algebraic_residual.csv').set_index('model')
        for m in ('A0', 'A1', 'A2'):
            V[f'{c}_g_{m}'] = sci(g.loc[m, 'mean_window_max_abs_g']); V[f'{c}_gmax_{m}'] = sci(g.loc[m, 'max_abs_g'])
        pr = ns[c]['pred_ratio']; V[f'{c}_ratio'] = f(pr[0], 2); V[f'{c}_ratio_ci'] = f'[{f(pr[1], 2)}, {f(pr[2], 2)}]'
        mr = ns[c]['mc_ratio']; V[f'{c}_mc'] = f(mr[0], 2); V[f'{c}_mc_ci'] = f'[{f(mr[1], 2)}, {f(mr[2], 2)}]'; V[f'{c}_mcn'] = ns[c]['mc_n']
        cl = pd.read_csv(R / f'{c}_closed_loop.csv').set_index('controller')
        for k in cl.index:
            V[f'{c}_cl_{k}_peak'] = f(cl.loc[k, 'angle_peak_deg_mean'], 2); V[f'{c}_cl_{k}_w'] = f(cl.loc[k, 'w_rms_mean'], 3)
            V[f'{c}_cl_{k}_u'] = f(cl.loc[k, 'u_energy_mean'], 1); V[f'{c}_cl_{k}_ms'] = f(cl.loc[k, 'solve_ms_mean'], 0); V[f'{c}_cl_{k}_lam'] = f(cl.loc[k, 'lam_min_mean'], 2)
        V[f'{c}_w_drop'] = f(100 * (1 - cl.loc['A2', 'w_rms_mean'] / cl.loc['C0', 'w_rms_mean']), 0)
        runs = pd.read_csv(R / f'{c}_closed_loop_runs.csv'); V[f'{c}_nreal'] = int(runs.run.nunique()); V[f'{c}_lam_obs_min'] = f(runs.lam_min.min(), 2)
        V[f'{c}_lam_obs_ratio'] = f(runs.lam_min.min() / sp['lambda_min_Lb_plus_G'], 2)
        V[f'{c}_gam_obs'] = f(np.degrees(np.arccos(min(1.0, runs.lam_min.min() / sp['lambda_min_Lb_plus_G']))), 0)
    tr = {c: pd.read_csv(R / f'{c}_transfer.csv') for c in ('case9', 'case39')}
    allr = np.concatenate([tr[c].ratio_single.values for c in tr])
    V['transfer_min'] = f(allr.min(), 2); V['transfer_max'] = f(allr.max(), 2)
    V['transfer_gain'] = f(max(np.abs(tr[c].ratio_single.values - tr[c].ratio_pooled.values).max() for c in tr), 3)
    tt = pd.read_csv(R / 'case9_tube_test.csv'); tj = json.load(open(R / 'case9_tube_test.json'))
    V['tube_ncal'] = 7; V['tube_limit'] = f(tj['limit_deg'], 1); V['tube_eps5_deg'] = f(np.degrees(tj['eps_rad'][5]), 1)
    for n in ('C0', 'A2', 'A3'):
        r = tt[tt.controller == n]; V[f'tube_{n}_peak'] = f(r.angle_peak_deg.mean(), 2); V[f'tube_{n}_frac'] = f(100 * r.frac_time_over_limit.mean(), 1); V[f'tube_{n}_viol'] = f(r.viol_integral_rad_s.mean() * 1e3, 2)
    lc = pd.read_csv(R / 'scalar_learning_curve.csv')
    for nz in (0.001, 0.01):
        for n in (2, 8, 32):
            V[f's_lc_{nz}_{n}'] = f(lc[(np.isclose(lc.noise_std, nz)) & (lc.n_train == n)].rmse_mean.iloc[0], 3)
    ch = pd.read_csv(R / 'regularity_chain.csv').set_index('case')          # Proposition 3 check (tools/regularity_chain_check.py)
    for c in ('case9', 'case39'):
        V[f'chain_{c}_dtheta'] = f(ch.loc[c, 'max_dtheta_ddelta'], 2); V[f'chain_{c}_kappa'] = f(ch.loc[c, 'S_gamma_bound'], 2)
    V['chain_fd_err_max'] = sci(ch['max_fd_error'].max(), 1)
    # exact-gradient study
    gc = pd.read_csv(R / 'network_gradient_check.csv').set_index('case')
    for c in ('case9', 'case39'):
        ac = pd.read_csv(R / f'{c}_analytic_closed_loop.csv').set_index('controller')
        V[f'grad_{c}_err'] = sci(gc.loc[c, 'max_rel_gradient_error'], 1)
        V[f'grad_{c}_ms_exact'] = f(gc.loc[c, 'ms_analytic'], 2); V[f'grad_{c}_ms_fd'] = f(gc.loc[c, 'ms_finite_difference'], 1)
        V[f'grad_{c}_wrms_exact'] = f(ac.loc['A2', 'w_rms_mean'], 4); V[f'grad_{c}_wrms_fd'] = f(ac.loc['A2fd', 'w_rms_mean'], 4)
        V[f'grad_{c}_step_exact_ms'] = f(ac.loc['A2', 'solve_ms_mean'], 1); V[f'grad_{c}_step_fd_ms'] = f(ac.loc['A2fd', 'solve_ms_mean'], 1)
    # operating-regime atlas, residual trust gate and its closed-loop test
    for c in ('case9', 'case39'):
        at = pd.read_csv(R / f'{c}_atlas.csv')
        k0 = at[at.dispatch_error == 0.0]; k0i = k0[k0.load_scale <= 2.5]
        V[f'atlas_{c}_k0_min'] = f(k0i.ratio_A2_over_A0.min(), 2); V[f'atlas_{c}_k0_max'] = f(k0i.ratio_A2_over_A0.max(), 2)
        edge = k0[k0.load_scale == 3.0].iloc[0]
        V[f'atlas_{c}_edge_ratio'] = f(edge.ratio_A2_over_A0, 2); V[f'atlas_{c}_edge_lam'] = f(edge.lambda_min_ratio, 2); V[f'atlas_{c}_edge_angle'] = f(edge.max_line_angle_deg, 1)
        for kap, tag in ((-0.1, 'm10'), (0.1, 'p10'), (0.2, 'p20'), (0.4, 'p40')):
            kk = at[(at.dispatch_error == kap) & (at.load_scale <= 2.5)]
            V[f'atlas_{c}_{tag}_min'] = f(kk.ratio_A2_over_A0.min(), 2); V[f'atlas_{c}_{tag}_max'] = f(kk.ratio_A2_over_A0.max(), 2)
        gp = pd.read_csv(R / f'{c}_gate_prediction.csv').set_index('scenario')
        for sc, tag in (('no change', 'nochg'), ('false-shift stress (8% load 3x noise)', 'stress'), ('announced mu 0.8', 'a08'), ('announced mu 1.25', 'a125'), ('announced mu 1.5', 'a15'),
                        ('unannounced mu 0.8', 'u08'), ('unannounced mu 1.25', 'u125'), ('unannounced mu 1.5', 'u15')):
            r_ = gp.loc[sc]
            V[f'gate_{c}_{tag}_a2a0'] = f(r_.ratio_A2_A0, 2); V[f'gate_{c}_{tag}_ga0'] = f(r_.ratio_gate_A0, 2); V[f'gate_{c}_{tag}_ga2'] = f(r_.ratio_gate_A2, 2)
            V[f'gate_{c}_{tag}_fb'] = f(100 * r_.duty_fallback, 0); V[f'gate_{c}_{tag}_ff'] = f(100 * r_.false_fallback_rate, 0)
            V[f'gate_{c}_{tag}_delay'] = 'none' if pd.isna(r_.detection_delay_s) else f(r_.detection_delay_s, 2)
        gc = pd.read_csv(R / f'{c}_gate_closed_loop.csv').set_index(['dispatch', 'controller'])
        for d in ('announced', 'unannounced'):
            for nm, tag in (('C0', 'c0'), ('A0', 'a0'), ('A2', 'a2'), ('physics+projection', 'pp'), ('SP-RTG', 'gate')):
                V[f'gcl_{c}_{d}_{tag}_wrms'] = f(gc.loc[(d, nm), 'w_rms'], 3); V[f'gcl_{c}_{d}_{tag}_ms'] = f(gc.loc[(d, nm), 'solve_ms'], 1)
            V[f'gcl_{c}_{d}_tau'] = f(gc.loc[(d, 'SP-RTG'), 'mean_tau_post'], 3)
        gv = pd.read_csv(R / f'{c}_gate_validation.csv')
        V[f'gval_{c}_spread'] = f(100 * (gv.mean_ratio_gate_over_A0.max() / gv.mean_ratio_gate_over_A0.min() - 1), 1)
    # dispatch reconciliation, recovery and mild-shift closed loop
    for c in ('case9', 'case39'):
        rp = pd.read_csv(R / f'{c}_reconcile_prediction.csv').set_index('scenario')
        for sc in rp.index:
            tag = sc.replace(' ', '_').replace('.', 'p')
            r_ = rp.loc[sc]
            for col, nm in (('rmse_A0', 'a0'), ('rmse_A2', 'a2'), ('rmse_oracle', 'or'), ('rmse_gate', 'gt'), ('rmse_reconcile', 'rc'), ('mean_rho_estimate', 'rho')):
                V[f'rec_{c}_{tag}_{nm}'] = f(r_[col], 3)
            if not pd.isna(r_['detection_delay_s']): V[f'rec_{c}_{tag}_delay'] = f(r_['detection_delay_s'], 2); V[f'rec_{c}_{tag}_mintau'] = f(r_['min_trust'], 2)
        ar = pd.read_csv(R / f'{c}_reconcile_after_restore.csv').set_index('scenario')
        for sc in ar.index:
            V[f'rec_{c}_{sc.replace(" ", "_").replace(".", "p")}_after_a2'] = f(ar.loc[sc, 'rmse_A2'], 3); V[f'rec_{c}_{sc.replace(" ", "_").replace(".", "p")}_after_rc'] = f(ar.loc[sc, 'rmse_reconcile'], 3)
        cl = pd.read_csv(R / f'{c}_reconcile_closed_loop.csv')
        for r_ in cl.itertuples():
            V[f'mild_{c}_{str(r_.mu_after).replace(".", "p")}_{r_.controller}_wrms'] = f(r_.w_rms, 3); V[f'mild_{c}_{str(r_.mu_after).replace(".", "p")}_{r_.controller}_bound'] = f(r_.input_at_bound_share, 2)
        ag = pd.read_csv(R / f'{c}_atlas_gate.csv'); hz = ag[ag.ratio_A2_over_A0 > 1]; bn = ag[ag.ratio_A2_over_A0 <= 1]
        V[f'agate_{c}_harm_flagged'] = f'{int((hz.gate_fallback_share > 0.5).sum())}/{len(hz)}'; V[f'agate_{c}_ben_flagged'] = f'{int((bn.gate_fallback_share > 0.5).sum())}/{len(bn)}'
        V[f'agate_{c}_k0_flagged'] = f"{int((ag[ag.dispatch_error == 0].gate_fallback_share > 0.5).sum())}/{len(ag[ag.dispatch_error == 0])}"
    # closed-loop recovery and fallback-only controller (case9)
    rc = pd.read_csv(R / 'case9_recovery_closed_loop.csv').set_index(['scenario', 'controller'])
    for sc, tag in (('mild 1.05', 'm105'), ('mild 1.10', 'm110'), ('recovery 1.10', 'rec110')):
        V[f'rcl_{tag}_pp_during'] = f(rc.loc[(sc, 'pp'), 'w_rms_during'], 3)
    for nm in ('A2', 'pp', 'gate', 'rec', 'oracle'):
        V[f'rcl_rec110_{nm}_during'] = f(rc.loc[('recovery 1.10', nm), 'w_rms_during'], 3); V[f'rcl_rec110_{nm}_after'] = f(rc.loc[('recovery 1.10', nm), 'w_rms_after_restore'], 3)
    V['rcl_rec110_gate_tau_after'] = f(rc.loc[('recovery 1.10', 'gate'), 'tau_mean_after_restore'], 3); V['rcl_rec110_rec_tau_after'] = f(rc.loc[('recovery 1.10', 'rec'), 'tau_mean_after_restore'], 3)
    # nuisance-shift test of the gate (parameter errors that are not a dispatch mismatch)
    nz = pd.concat([pd.read_csv(R / f'{c}_nuisance.csv') for c in ('case9', 'case39')])
    V['nuis_ff_max'] = f(100 * nz.false_fallback_share.max(), 1); V['nuis_trust_min'] = f(nz.mean_trust.min(), 2); V['nuis_trust_max'] = f(nz.mean_trust.max(), 2); V['nuis_rho_max'] = f(nz.max_abs_rho_minus_1.max(), 2)
    # held-out non-uniform dispatch family
    for c_ in ('case9', 'case39'):
        hd = pd.read_csv(R / f'{c_}_heldout_dispatch.csv')
        for r_ in hd.itertuples():
            tag = str(r_.mu_after).replace('.', 'p')
            for col, nm in (('rmse_A0', 'a0'), ('rmse_A2', 'a2'), ('rmse_oracle', 'or'), ('rmse_gate', 'gt'), ('rmse_reconcile', 'rc'), ('mean_rho_estimate', 'rho')):
                V[f'held_{c_}_{tag}_{nm}'] = f(getattr(r_, col), 3)
            V[f'held_{c_}_{tag}_delay'] = f(r_.detection_delay_s, 2)
    return V


if __name__ == '__main__':
    V = collect()
    for k, v in V.items(): print(k, v)
