"""Numerical checks of the two standard facts stated as remarks in the manuscript (run: python tests/test_standard_facts.py).
(1) Gronwall form is attained by F(x)=Lx+rho+w, Fhat(x')=Lx'.
(2) Split-conformal coverage: for exchangeable scores the k-th smallest of n scores covers a new score with probability exactly k/(n+1)
    (no ties), computed here by exact enumeration of rank patterns, and by simulation for the manuscript's n = 20."""
import itertools
from fractions import Fraction
import numpy as np
from scipy.integrate import solve_ivp


def test_gronwall_attained():
    for L, c in ((0.5, 0.3), (2.0, 0.05), (4.0, 1.0)):
        # plant x' = F(x) = L x + c, predictor x'' = Fhat(x') = L x', same initial state
        sol = solve_ivp(lambda t, y: [L * y[0] + c, L * y[1]], (0, 1.5), [1.0, 1.0], rtol=1e-12, atol=1e-12, dense_output=True)
        for t in (0.2, 0.7, 1.5):
            x, xp = sol.sol(t)
            e = x - xp
            assert abs(e - c * (np.exp(L * t) - 1) / L) <= 1e-8 * max(1.0, abs(e))
            assert e >= 0 and abs(abs((L * x + c) - L * xp) - (L * abs(e) + c)) <= 1e-8 * max(1.0, abs(x))   # hypothesis holds with equality


def test_conformal_coverage_exact_enumeration():
    # all orderings of n+1 exchangeable scores are equally likely; the new score is covered iff its rank among the n+1 is <= k
    for n in range(1, 8):
        for alpha in (Fraction(1, 20), Fraction(1, 10), Fraction(1, 4)):
            k = -((-(n + 1) * (1 - alpha)) // 1)             # ceiling of (n+1)(1-alpha)
            k = int(k)
            if k > n:
                continue                                       # threshold would be +infinity
            hits = tot = 0
            for perm in itertools.permutations(range(n + 1)):  # perm[i] = rank of score i; the new score is index n
                tot += 1
                cal_sorted = sorted(perm[:n])
                hits += perm[n] <= cal_sorted[k - 1]
            assert Fraction(hits, tot) == Fraction(k, n + 1) >= 1 - alpha


def test_conformal_coverage_n20_simulation():
    rng = np.random.default_rng(0)
    n, trials = 20, 200000
    s = rng.standard_normal((trials, n + 1))
    eps = s[:, :n].max(axis=1)                                 # k = 20 = n for alpha = 0.05
    cov = np.mean(s[:, n] <= eps)
    assert abs(cov - 20 / 21) < 0.003 and cov >= 0.95


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
