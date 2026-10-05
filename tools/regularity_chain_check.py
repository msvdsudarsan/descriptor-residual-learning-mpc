"""Numerical check of the chain  regularity margin -> manifold sensitivity -> Lipschitz constant (Proposition 3, Lipschitz constant of the reduced network dynamics).

For the network realization, theta = theta(delta) is defined by g(delta, theta) = 0 and
    d theta / d delta = J^{-1} E C,   E = generator-to-bus incidence, C = diag(b_i cos psi_i),
    d psi   / d delta = I - E^T J^{-1} E C,        psi_i = delta_i - theta_sigma(i).
Proposition 2 gives ||J^{-1}|| <= 1/(cos(gamma) lambda_min(B)); hence
    ||d theta / d delta||_2 <= k b_max / (cos(gamma) lambda_min(B)) =: kappa_gamma  (the proof gives sqrt(k) here, k for psi),
    ||d psi / d delta||_2   <= 1 + S_gamma,
where k is the largest number of generators at one bus (k = 1 for case9 and case39).

The script samples states on the algebraic manifold, keeps those that satisfy the angle limit gamma on every line and
rotor, and compares the actual norms with the bounds.  Output: results/regularity_chain.csv and a printed table.
Usage:  python tools/regularity_chain_check.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import network_benchmark as nb

GAMMA = np.radians(30.0)


def incidence(c):
    E = np.zeros((c['n'], c['ng']))
    for i, j in enumerate(c['gbus']):
        E[j, i] = 1.0
    return E


def run(name, n_samples=400, seed=0):
    c = nb.build_case(name)
    n, ng = c['n'], c['ng']
    E = incidence(c)
    k = int(np.max(E.sum(axis=1)))
    B = nb.alg_res_jac(np.zeros(n), np.zeros(ng), c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], True)[1]
    lam_B = float(np.linalg.eigvalsh(B).min())
    bmax = float(c['bg'].max())
    S_gamma = k * bmax / (np.cos(GAMMA) * lam_B)               # kappa_gamma of Proposition 3
    th0, de0, _ = nb.equilibrium(c)
    rng = np.random.default_rng(seed)
    rows = []
    tries = 0
    while len(rows) < n_samples and tries < 20 * n_samples:
        tries += 1
        de = de0 + rng.uniform(-0.35, 0.35, ng) * rng.uniform(0.2, 1.0)
        th, _ = nb.solve_theta(th0.copy(), de, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], 1e-12, 50)
        g, J = nb.alg_res_jac(th, de, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], True)
        if np.max(np.abs(g)) > 1e-9:
            continue
        line_ok = np.all(np.abs(th[c['lf']] - th[c['lt']]) <= GAMMA)
        rotor_ok = np.all(np.abs(de - th[c['gbus']]) <= GAMMA)
        if not (line_ok and rotor_ok):
            continue
        C = np.diag(c['bg'] * np.cos(de - th[c['gbus']]))
        Sth = np.linalg.solve(J, E @ C)                      # d theta / d delta
        Spsi = np.eye(ng) - E.T @ Sth                         # d psi / d delta
        # finite-difference check of the sensitivity formula on one random direction
        v = rng.standard_normal(ng); v /= np.linalg.norm(v); eps = 1e-6
        thp, _ = nb.solve_theta(th.copy(), de + eps * v, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], 1e-13, 50)
        fd_err = np.linalg.norm((thp - th) / eps - Sth @ v)
        rows.append((np.linalg.norm(Sth, 2), np.linalg.norm(Spsi, 2), fd_err))
    a = np.array(rows)
    return dict(case=name, n_states=len(a), lambda_min_B=lam_B, b_max=bmax, S_gamma_bound=S_gamma, psi_bound=1 + S_gamma,
                max_dtheta_ddelta=a[:, 0].max(), max_dpsi_ddelta=a[:, 1].max(), max_fd_error=a[:, 2].max(),
                ratio_theta=a[:, 0].max() / S_gamma, ratio_psi=a[:, 1].max() / (1 + S_gamma))


if __name__ == '__main__':
    import pandas as pd
    out = pd.DataFrame([run('case9'), run('case39')])
    out.to_csv(ROOT / 'results' / 'regularity_chain.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 200)
    print(out.to_string(index=False))
    assert (out['max_dtheta_ddelta'] <= out['S_gamma_bound'] + 1e-9).all()
    assert (out['max_dpsi_ddelta'] <= out['psi_bound'] + 1e-9).all()
    assert (out['max_fd_error'] < 1e-4).all()
    print('bounds hold on all sampled states')
