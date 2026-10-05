"""Sanity tests that tie the manuscript's claims to the code (run: python tests/test_consistency.py)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import scalar_benchmark as sb
import network_benchmark as nb


def test_scalar_baseline_is_misspecified():
    # physics model must differ from the plant same state, same input, different acceleration
    d, w, u = 0.5, 0.2, -0.3
    v = sb.solve_v(d, u)
    assert abs(sb.acc_true(d, w, u, 0.0) - sb.acc_phys(d, w, u, v)) > 1e-2


def test_scalar_global_index_one():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d, u = rng.uniform(-6, 6), rng.uniform(-3, 3)
        v = sb.solve_v(d, u)
        assert abs(sb.g_res(d, v, u)) < 1e-12
        assert 1 + 3 * sb.CB * v * v >= 1


def test_network_jacobian_bound():
    rng = np.random.default_rng(1)
    for name in ('case9', 'case39'):
        c = nb.build_case(name)
        B = nb.alg_res_jac(np.zeros(c['n']), np.zeros(c['ng']), c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], True)[1]
        lam0 = np.linalg.eigvalsh(B).min()
        gamma = np.radians(30)
        for _ in range(50):
            th = rng.uniform(-gamma / 2, gamma / 2, c['n'])
            th = np.clip(th, -1, 1)
            # enforce limit on every line and rotor angle
            ok = np.all(np.abs(th[c['lf']] - th[c['lt']]) <= gamma)
            de = th[c['gbus']] + rng.uniform(-gamma, gamma, c['ng'])
            J = nb.alg_res_jac(th, de, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], True)[1]
            if ok:
                assert np.linalg.eigvalsh(J).min() >= np.cos(gamma) * lam0 - 1e-9


def test_newton_projection_consistency():
    c = nb.build_case('case9'); th, de, res = nb.equilibrium(c)
    assert res < 1e-10
    de2 = de + 0.05
    th2, _ = nb.solve_theta(th, de2, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], 1e-12, 30)
    g, _ = nb.alg_res_jac(th2, de2, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], False)
    assert np.max(np.abs(g)) < 1e-10


def test_dispatch_balance():
    for name in ('case9', 'case39'):
        c = nb.build_case(name)
        assert abs(c['Pm'].sum() - c['Pd'].sum()) < 1e-10


def test_conformal_quantile_small_n_is_max():
    from common import conformal_quantile
    x = np.random.default_rng(3).uniform(0, 1, 12)
    assert conformal_quantile(x, 0.05) == x.max()          # alpha = 0.05: n <= 38 -> largest score
    for n in (19, 20, 38):
        z = np.random.default_rng(n).uniform(0, 1, n); assert conformal_quantile(z, 0.05) == z.max()
    z = np.sort(np.random.default_rng(39).uniform(0, 1, 39)); assert conformal_quantile(z, 0.05) == z[37]     # n = 39 -> 38th order statistic
    y = np.arange(1, 101, dtype=float)
    assert conformal_quantile(y, 0.05) == 96.0             # ceil(101*0.95) = 96


def test_out_of_library_term_off_by_default():
    c0 = nb.build_case('case9'); c1 = nb.build_case('case9', ool=0.15)
    th, de, _ = nb.equilibrium(c0)
    w = np.full(c0['ng'], 0.1); u = np.zeros(c0['ng'])
    a0 = nb.swing_acc(de + 0.2, w, u, th, c0['gbus'], c0['bg'], c0['M'], c0['D'], c0['Pm'], True, c0['kappa'], c0['fric'])
    a1 = nb.swing_acc(de + 0.2, w, u, th, c1['gbus'], c1['bg'], c1['M'], c1['D'], c1['Pm'], True, c1['kappa'], c1['fric'])
    assert c0['kappa'][1] == 0.0 and np.max(np.abs(a1 - a0)) > 1e-3


def test_lipschitz_chain_bound():
    # Proposition 3: manifold sensitivity and rotor-angle sensitivity stay below kappa_gamma and 1 + kappa_gamma
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import regularity_chain_check as rc
    for name in ('case9', 'case39'):
        r = rc.run(name, n_samples=25, seed=3)
        assert r['max_dtheta_ddelta'] <= r['S_gamma_bound'] and r['max_dpsi_ddelta'] <= r['psi_bound'] and r['max_fd_error'] < 1e-4


def test_exact_gradient_matches_central_differences():
    # Section 4.5: exact gradient of the projected-predictor cost against central differences
    import network_experiments as ne
    from network_gradient import mpc_cost_net_analytic
    for name in ('case9', 'case39'):
        c = ne.build_case(name, load_scale=ne.STRESS[name]); th, de, _ = ne.equilibrium(c); ng = c['ng']
        rng = np.random.default_rng(3); coef = np.zeros((ng, ne.NF + 1)); coef[:, :ne.NF] = rng.normal(0, 0.05, (ng, ne.NF)); coef[:, ne.NF] = 1.0
        rel_ref = de - np.sum(c['M'] * de) / np.sum(c['M']); N = 5
        de0 = de + rng.normal(0, 0.1, ng); w0 = rng.normal(0, 0.3, ng); th0, _ = ne.solve_theta(th, de0, c['Pd'], *ne.args_of(c), 1e-12, 30)
        U = rng.normal(0, 0.5, N * ng) * np.tile(c['umax'], N)
        a = (de0, w0, th0, np.zeros(ng), N, ne.TS, ne.NSUB, 1, 1, coef, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'],
             rel_ref, c['umax'], 0.5, 20.0, 0.02, 0.2, np.radians(20.0), np.array([0, .015, .03, .04, .055, .06]), 3e3)
        J1, g1 = mpc_cost_net_analytic(U, *a); gfd = np.zeros_like(U)
        for j in range(len(U)):
            e = np.zeros_like(U); e[j] = 1e-6; gfd[j] = (ne.mpc_cost_net(U + e, *a) - ne.mpc_cost_net(U - e, *a)) / 2e-6
        assert abs(J1 - ne.mpc_cost_net(U, *a)) < 1e-9 * abs(J1) and np.linalg.norm(g1 - gfd) < 1e-6 * np.linalg.norm(gfd)


def test_trust_gate_separates_shift_from_no_shift():
    # Section 4.6: tau stays at 1 in ordinary operation and drops below 0.5 within 0.3 s of an unannounced dispatch step
    import numpy as np
    import network_validity_gate as g
    c0, th, de, coef = g.nominal_fit('case9'); lam_ref = g.regularity(c0, th, de)[0]
    q0, _ = g.calibrate_q0(c0, th, de, coef, lam_ref, 'case9')
    quiet = g.make_shift_runs(c0, th, de, None, 2, 9301, 0.04, True, 'case9')
    shifted = g.make_shift_runs(c0, th, de, 1.5, 2, 9325, 0.06, False, 'case9')
    for run in quiet:
        tau = g.gate_series(c0, coef, run[2], run[1], q0, 0.25, 0.0, lam_ref, np.random.default_rng(1))[0]
        assert tau[1:].min() > 0.5
    for run in shifted:
        tau = g.gate_series(c0, coef, run[2], run[1], q0, 0.25, 0.0, lam_ref, np.random.default_rng(1))[0]
        t = run[1]['t']; assert tau[(t > g.T_SHIFT - 1e-9) & (t < g.T_SHIFT + 0.3)].min() < 0.5


def test_reconciliation_recovers_dispatch_scale():
    # Section 4.6: after an unannounced dispatch step the estimated scale is within 3% of the true one, and it returns to one when the step is undone
    import numpy as np
    import network_reconcile as nr
    c0, th, de, coef = nr.nominal_fit('case9'); lam_ref = nr.regularity(c0, th, de)[0]
    q0, _ = nr.calibrate_q0(c0, th, de, coef, lam_ref, 'case9')
    run = nr.make_runs(c0, th, de, 1.25, nr.T_REST, 1, 9533, 0.06, 10.0, 'case9')[0]
    tau, q, rho = nr.reconcile_series(c0, coef, run[1], q0, nr.A_GATE, True, np.random.default_rng(1))
    t = run[1]['t']
    assert abs(rho[(t > nr.T_SHIFT + 0.8) & (t < nr.T_REST - 0.1)].mean() - 1.25) < 0.03
    assert abs(rho[t > nr.T_REST + 0.8].mean() - 1.0) < 0.03 and tau[t > nr.T_REST + 0.8].min() > 0.5


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
