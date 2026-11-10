"""Lemma of the transfer proposition: inside the angle limits d theta / d delta = J^{-1} E C is entrywise nonnegative with unit row sums
(so every line angle and rotor-to-bus angle changes by at most the span of the rotor-angle change). Self-contained check on case9 and case39."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_props_independent import nominal_fit, mats, inside, solve_theta, args_of


def check(name, nstates, seed):
    c, th_eq, de_eq, _ = nominal_fit(name); rng = np.random.default_rng(seed); ng = c['ng']; cnt = 0; tries = 0
    while cnt < nstates and tries < 5000:
        tries += 1
        de = de_eq + rng.normal(0, rng.choice([0.05, 0.15, 0.3]), ng)
        th, _ = solve_theta(th_eq, de, c['Pd'] * rng.uniform(0.7, 1.4), *args_of(c), 1e-13, 60)
        if not inside(c, de, th): continue
        cnt += 1; J, _, E = mats(c, de, th); C = np.diag(c['bg'] * np.cos(de - th[c['gbus']])); S = np.linalg.solve(J, E @ C)
        assert S.min() >= -1e-12 and np.abs(S.sum(1) - 1).max() < 1e-10
        for l in range(len(c['lf'])):
            d = S[c['lf'][l]] - S[c['lt'][l]]
            assert abs(d.sum()) < 1e-10 and 0.5 * np.abs(d).sum() <= 1.0     # sup of |line-angle change| / span over span<=1 is 0.5*||row difference||_1
    assert cnt == nstates


def test_transfer_case9(): check('case9', 20, 1)
def test_transfer_case39(): check('case39', 10, 2)
