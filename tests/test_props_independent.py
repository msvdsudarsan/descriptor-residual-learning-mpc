"""Independent numerical check of Propositions 2 and 3 with formulas written separately from the repository's solver code:
lambda_min(J) >= cos(gamma) lambda_min(B); ||d theta/d delta|| <= kappa_gamma (finite differences of the Newton solution);
Jacobian of the scaled reduced vector field below L_f.  Random states inside the 30 degree region.  Numerical evidence, not a proof."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from network_validity_gate import *
G = np.radians(30.0)


def mats(c, de, th):
    n, ng = c['n'], c['ng']; lf, lt, lb, gb, bg = c['lf'], c['lt'], c['lb'], c['gbus'], c['bg']
    J = np.zeros((n, n)); B = np.zeros((n, n))
    for l in range(len(lf)):
        a = np.zeros(n); a[lf[l]] = 1; a[lt[l]] = -1
        J += lb[l] * np.cos(th[lf[l]] - th[lt[l]]) * np.outer(a, a); B += lb[l] * np.outer(a, a)
    for i in range(ng):
        J[gb[i], gb[i]] += bg[i] * np.cos(de[i] - th[gb[i]]); B[gb[i], gb[i]] += bg[i]
    E = np.zeros((n, ng)); E[gb, np.arange(ng)] = 1
    return J, B, E


def inside(c, de, th):
    return (np.all(np.abs(th[c['lf']] - th[c['lt']]) <= G) and np.all(np.abs(de - th[c['gbus']]) <= G))


def check(name, nstates=25, seed=5):
    c, th_eq, de_eq, coef = nominal_fit(name); rng = np.random.default_rng(seed); ng = c['ng']
    k = max(np.bincount(c['gbus'])); _, B, E = mats(c, de_eq, th_eq); lamB = np.linalg.eigvalsh(B)[0]
    kap = k * c['bg'].max() / (np.cos(G) * lamB); Lf = 1.0 + np.max(c['Dphys'] / c['M']) + np.max(c['bg'] / c['M']) * (1 + kap); worst = [0, 0, 0]; cnt = 0
    while cnt < nstates:
        de = de_eq + rng.normal(0, 0.15, ng); th, _ = solve_theta(th_eq, de, c['Pd'], *args_of(c), 1e-13, 40)
        if not inside(c, de, th): continue
        cnt += 1; J, _, _ = mats(c, de, th)
        worst[0] = max(worst[0], np.cos(G) * lamB / np.linalg.eigvalsh(J)[0])                    # must stay <= 1
        C = np.diag(c['bg'] * np.cos(de - th[c['gbus']])); D1 = np.linalg.solve(J, E @ C)       # d theta / d delta from the implicit function
        h = 1e-6; Dfd = np.column_stack([(solve_theta(th, de + h * np.eye(ng)[:, i], c['Pd'], *args_of(c), 1e-14, 40)[0] - solve_theta(th, de - h * np.eye(ng)[:, i], c['Pd'], *args_of(c), 1e-14, 40)[0]) / (2 * h) for i in range(ng)])
        assert np.linalg.norm(D1 - Dfd, 2) < 1e-6 * (1 + np.linalg.norm(D1, 2))
        worst[1] = max(worst[1], np.linalg.norm(D1, 2) / kap)                                        # must stay <= 1
        def F(x):                                                                                     # scaled state (delta, T*omega), T = 1 s
            d, w = x[:ng], x[ng:]; t, _ = solve_theta(th, d, c['Pd'], *args_of(c), 1e-14, 40)
            return np.r_[w, swing_acc(d, w, np.zeros(ng), t, c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], False, c['kappa'], c['fric'])]
        x0 = np.r_[de, rng.normal(0, 0.3, ng)]; JF = np.column_stack([(F(x0 + h * np.eye(2 * ng)[:, i]) - F(x0 - h * np.eye(2 * ng)[:, i])) / (2 * h) for i in range(2 * ng)])
        worst[2] = max(worst[2], np.linalg.norm(JF, 2) / Lf)                                         # must stay <= 1
    return worst, kap, Lf


def test_props_case9():
    w, kap, Lf = check('case9'); print('case9 ratios (<=1 expected)', w, kap, Lf); assert max(w) <= 1.0


def test_props_case39():
    w, kap, Lf = check('case39', nstates=12); print('case39 ratios (<=1 expected)', w, kap, Lf); assert max(w) <= 1.0
