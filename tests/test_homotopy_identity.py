"""Sign and value check of the topology homotopy identity (manuscript Eq. (homotopy)); a known-type identity (implicit-function derivative integrated along a path), NOT a new theorem.
Setting: reduced lossless algebraic model g(theta)=0 of the repository, line l=(j,k) with susceptance b_l replaced by (1-s) b_l, s in [0,1] (s=1: outage).
Identity: Theta(delta;T*)-Theta(delta;T0) = int_0^1 J_s^{-1}(e_j-e_k) b_l sin(phi_l(s)) ds, with J_s=-dg/dtheta at T_s and phi_l(s) the angle across line l.
Assumptions: J_s nonsingular for all s (lines whose outage does not island the network; angles inside +-pi/2); delta, load fixed.
Tolerance: 1e-8 (Gauss-Legendre with 24 nodes, Newton tolerance 1e-13).  A reversed sign must FAIL by a margin of at least 1e-3 (negative control).
Run: python3 -m pytest tests/test_homotopy_identity.py   (or python3 tests/test_homotopy_identity.py)"""
import sys, numpy as np
from pathlib import Path
import scipy.sparse as sp, scipy.sparse.csgraph as cg
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from network_validity_gate import nominal_fit
from network_experiments import args_of
from network_benchmark import solve_theta, alg_res_jac
TOL = 1e-8

def connected(c, l):
    keep = np.arange(len(c['lf'])) != l; n = len(c['Pd'])
    A = sp.coo_matrix((np.ones(keep.sum()), (c['lf'][keep], c['lt'][keep])), shape=(n, n)); return cg.connected_components(A, directed=False)[0] == 1

def check(name, lines, sign=+1.0):
    c, th_eq, de_eq, _ = nominal_fit(name); lf, lt, lb, gbus, bg, Pd = c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['Pd']; n = len(Pd)
    rng = np.random.default_rng(3); de = de_eq + rng.normal(0, 0.05, c['ng']); worst = 0.0
    th0, _ = solve_theta(th_eq, de, Pd, lf, lt, lb, gbus, bg, 1e-13, 60)
    for l in lines:
        lb1 = lb.copy(); lb1[l] = 0.0                                   # outage: zero susceptance keeps the index set
        th1, _ = solve_theta(th0, de, Pd, lf, lt, lb1, gbus, bg, 1e-13, 80); target = th1 - th0
        xs, ws = np.polynomial.legendre.leggauss(24); xs = 0.5 * (xs + 1); ws = 0.5 * ws; acc = np.zeros(n); th = th0.copy()
        for s, w in zip(xs, ws):
            lbs = lb.copy(); lbs[l] = (1 - s) * lb[l]; th, _ = solve_theta(th0, de, Pd, lf, lt, lbs, gbus, bg, 1e-13, 80)
            g, J = alg_res_jac(th, de, Pd, lf, lt, lbs, gbus, bg, True); rhs = np.zeros(n); phi = th[lf[l]] - th[lt[l]]
            rhs[lf[l]] += 1.0; rhs[lt[l]] -= 1.0; rhs *= lb[l] * np.sin(phi); acc += w * np.linalg.solve(J, rhs)
        worst = max(worst, np.abs(sign * acc - target).max())
    return worst

def lines_of(name, k):
    c = nominal_fit(name)[0]; ok = [l for l in range(len(c['lf'])) if connected(c, l)]; return ok[:k]

def test_homotopy_case9():
    assert check('case9', lines_of('case9', 6)) < TOL
def test_homotopy_case39():
    assert check('case39', lines_of('case39', 8)) < TOL
def test_homotopy_wrong_sign_fails():
    assert check('case9', lines_of('case9', 2), sign=-1.0) > 1e-3

if __name__ == '__main__':
    for nm, k in (('case9', 6), ('case39', 8)): print(nm, 'max abs deviation over', k, 'outages:', check(nm, lines_of(nm, k)))
