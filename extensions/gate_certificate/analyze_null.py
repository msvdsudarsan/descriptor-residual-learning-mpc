import sys, math
sys.path.insert(0, 'code')
from gatelib import *
from scipy.stats import kurtosis, skew
ALPHA = 0.05
REPO_Q0 = dict(case9=0.023839465238203714, case39=0.03706069563617859)

def maxsm(S): return np.array([smooth(s).max() for s in S])
def maxs(S): return np.nanmax(S[:, 1:], axis=1)
def cq(x, a): return conformal_quantile(np.asarray(x), a)

for name in ('case9', 'case39'):
    print('=' * 30, name)
    c0, th, de, coef = nominal_fit(name); lam_ref = regularity(c0, th, de)[0]
    q_rep, mx_rep = calibrate_q0(c0, th, de, coef, lam_ref, name)
    print('repo calibrate_q0 reproduced: q0 = %.9f (stored %.9f); n_cal = %d; max scores %s' % (q_rep, REPO_Q0[name], len(mx_rep), np.round(mx_rep, 5)))
    n = len(mx_rep); kk = math.ceil((n + 1) * (1 - ALPHA)); print('  n=%d alpha=%.2f: k = ceil((n+1)(1-a)) = %d; k>n -> repo takes the sample maximum, guaranteed level n/(n+1) = %.4f (NOT 0.95); smallest n for alpha=0.05 is %d' % (n, ALPHA, kk, n / (n + 1), math.ceil(1 / ALPHA - 1)))
    S = np.load(OUT / f'null_{name}_nominal.npy'); N = len(S); mx = maxsm(S)
    print('pool N=%d fresh null runs (seeds 100000+, independent of calibration seeds 9001/9002)' % N)
    print('  max smoothed score: mean %.4f median %.4f 95%% %.4f 99%% %.4f max %.4f; skew %.2f ex.kurt %.2f; CV %.2f' % (mx.mean(), np.median(mx), np.quantile(mx, .95), np.quantile(mx, .99), mx.max(), skew(mx), kurtosis(mx), mx.std() / mx.mean()))
    sv = S[:, 1:].ravel(); print('  per-sample score: mean %.4f rms %.4f ex.kurt %.2f (a Gaussian-vector rms would have small kurtosis)' % (sv.mean(), np.sqrt((sv ** 2).mean()), kurtosis(sv)))
    # (1) the repo's actual threshold, fresh runs
    for lab, q in (('repo q0 (n=10)', q_rep),):
        k = int((mx > q).sum()); lo, hi = cp_interval(k, N)
        print('  %s q0=%.5f: false-alarm freq over %d fresh runs = %d/%d = %.4f  CP95 [%.4f, %.4f]; guaranteed marginal <= 1/11 = %.4f (conditional on this q0 the rate is whatever it is)' % (lab, q, N, k, N, k / N, lo, hi, 1 / 11))
    print('  P(conditional false-alarm rate of a draw of the n=10 sample-max threshold exceeds the rate seen here) = (1-rate)^10 = %.3f' % ((1 - (mx > q_rep).mean()) ** 10))
    qn = np.quantile(mx, 1 - ALPHA)
    print('  large-sample 95%% quantile of the pool = %.5f (repo q0 / this = %.2f)' % (qn, q_rep / qn))
    # (2) marginal validity over calibration draws, permutation scheme and disjoint blocks
    rng = np.random.default_rng(12345); rows = []
    for n in (10, 19, 39, 99):
        k = math.ceil((n + 1) * (1 - ALPHA)); lvl = min(k, n) / (n + 1); theo = 1 - lvl
        kuse = min(k, n)
        R = 20000; cond = np.empty(R); xs = np.sort(mx)
        for r in range(R):
            idx = rng.choice(N, n, replace=False); cal = mx[idx]; q = np.sort(cal)[kuse - 1]
            exc_all = (mx > q).sum(); exc_cal = (cal > q).sum(); cond[r] = (exc_all - exc_cal) / (N - n)
        # disjoint blocks: first n runs calibrate, run n+1 tests (independent blocks => CP interval is exact for the marginal probability)
        nb = N // (n + 1); hits = 0
        for b in range(nb):
            blk = mx[b * (n + 1):(b + 1) * (n + 1)]; q = np.sort(blk[:n])[kuse - 1]; hits += int(blk[n] > q)
        lo, hi = cp_interval(hits, nb)
        # PAC-type view: fraction of calibration sets whose true conditional false-alarm rate exceeds alpha (truth proxied by the pool)
        print('  n_cal=%3d: guarantee P(no alarm) >= %.4f (false alarm <= %.4f). permutation mean false-alarm %.4f (sd across calib sets %.4f; 5%%/95%% of conditional rate %.3f/%.3f; P(cond rate > %.2f) = %.3f). disjoint blocks: %d/%d = %.4f CP95 [%.4f, %.4f]' % (n, lvl, 1 - lvl, cond.mean(), cond.std(), np.quantile(cond, .05), np.quantile(cond, .95), ALPHA, (cond > ALPHA).mean(), hits, nb, hits / nb, lo, hi))
    # (3) false fallback tau<0.5 vs false alarm q>1, for gate constants a
    for a in (0.25, 0.5, 1.0, 2.0):
        thr = 1 + math.log(2) / a; k = int((mx / q_rep > thr).sum()); lo, hi = cp_interval(k, N)
        print('  false FALLBACK (tau<0.5 needs q > 1+ln2/a = %.2f) for a=%.2f: %d/%d = %.4f CP95 [%.4f,%.4f]' % (thr, a, k, N, k / N, lo, hi))
    # (4) mismatched nulls
    for tag in ('noise3', 'inert105', 'inert110', 'load16'):
        f = OUT / f'null_{name}_{tag}.npy'
        if not f.exists(): continue
        M = maxsm(np.load(f)); out = []
        for q in (q_rep, qn):
            k = int((M > q).sum()); lo, hi = cp_interval(k, len(M)); out.append('%d/%d = %.3f [%.3f, %.3f]' % (k, len(M), k / len(M), lo, hi))
        print('  mismatched null %-8s median max-score %.4f  false alarm: repo q0: %s | pool-95%%: %s' % (tag, np.median(M), out[0], out[1]))
    f = OUT / f'null_{name}_T16.npy'
    if f.exists():
        S16 = np.load(f); M = maxsm(S16); k = int((M > q_rep).sum()); lo, hi = cp_interval(k, len(M)); kq = int((M > qn).sum()); lo2, hi2 = cp_interval(kq, len(M))
        print('  16 s nominal runs (threshold calibrated on 8 s runs): repo q0 %d/%d = %.3f [%.3f,%.3f]; pool-95%% %d/%d = %.3f [%.3f,%.3f]' % (k, len(M), k / len(M), lo, hi, kq, len(M), kq / len(M), lo2, hi2))
    np.save(OUT / f'q_pool95_{name}.npy', np.array([qn, np.quantile(maxs(S), 0.95), cq(maxs(S), 0.05)]))
    print('  unsmoothed per-run max score: conformal 95%% (n=%d) = %.5f' % (N, cq(maxs(S), 0.05)))
