"""Shared utilities: optional numba, polynomial/Fourier libraries, STLSQ fitting,
split-conformal residual envelope, bootstrap helpers. Pure NumPy/SciPy otherwise."""
import itertools
import numpy as np

try:
    from numba import njit
except ImportError:                      # slower, but results are identical
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


def monomial_exponents(nvar, degree):
    """All exponent tuples with total degree <= degree (graded lexicographic order)."""
    exps = [e for e in itertools.product(range(degree + 1), repeat=nvar) if sum(e) <= degree]
    exps.sort(key=lambda e: (sum(e), e))
    return np.array(exps, dtype=np.int64)


def monomial_features(X, E):
    """X: (m, nvar); E: (nlib, nvar) -> (m, nlib)."""
    X = np.atleast_2d(X)
    out = np.ones((X.shape[0], E.shape[0]))
    for j in range(E.shape[0]):
        for k in range(E.shape[1]):
            if E[j, k]:
                out[:, j] *= X[:, k] ** E[j, k]
    return out


def stlsq(Theta, y, ridge=1e-6, thresh=0.02, iters=10):
    """Sequentially thresholded ridge regression (SINDy-style) on RMS-standardised features.
    A term is dropped when its RMS contribution is below thresh * std(y).
    Returns coefficients in the original feature scale."""
    sc = np.sqrt(np.mean(Theta ** 2, axis=0)) + 1e-12
    Z = Theta / sc
    n = Z.shape[1]
    cut = thresh * max(np.std(y), 1e-12)
    active = np.ones(n, bool)

    def fit(mask):
        A = Z[:, mask]
        w = np.zeros(n)
        w[mask] = np.linalg.solve(A.T @ A + ridge * len(y) * np.eye(A.shape[1]), A.T @ y)
        return w

    w = fit(active)
    for _ in range(iters):
        keep = np.abs(w) >= cut
        if keep.sum() == 0 or np.array_equal(keep, active):
            break
        active = keep
        w = fit(active)
    return w / sc


def conformal_quantile(abs_err, alpha=0.05):
    """Split-conformal (1-alpha) quantile with the finite-sample correction."""
    e = np.sort(np.asarray(abs_err).ravel())
    n = len(e)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(e[min(k, n) - 1])


def bootstrap_ci(x, B=4000, seed=0, stat=np.mean, level=0.95):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    s = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(x)), float(lo), float(hi)
