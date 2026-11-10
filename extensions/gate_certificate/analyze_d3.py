import sys, math
import numpy as np
from scipy.stats import binom, beta, t as tdist
OUT = '.'
LAMS = [0, 0.5, 1, 1.5, 2, 3]; n = 400; seed = 90000; DELTA = 0.1
def cp_ub(k, n, d): return 1.0 if k == n else float(beta.ppf(1 - d, k + 1, n - k))
def h1(a, b):
    a = min(max(a, 1e-300), 1 - 1e-16); return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
def p_binom(k, n, al): return float(binom.cdf(k, n, al))
def p_hb(k, n, al):
    rh = k / n
    if rh >= al: return 1.0
    return float(min(math.exp(-n * h1(rh, al)) if rh > 0 else (1 - al) ** n, math.e * binom.cdf(math.ceil(n * rh), n, al)))
for lim in (23, 30):
    R = {l: np.load(OUT + f'd3_lim{lim}_lam{l:g}_n{n}_s{seed}.npy') for l in LAMS}
    print('=' * 20, 'limit %d deg, case9, n = %d i.i.d. trajectories per lambda (common disturbance seeds %d..), delta = %.2f' % (lim, n, seed, DELTA))
    print('lambda | mean peak [deg] | mean time-frac above | any-violation k | frac>0.05 k | frac>0.10 k | frac>0.20 k | tracking rms(de-ref) [rad] | w_rms | u-energy')
    for l in LAMS:
        r = R[l]; print('%5.1f  | %.3f | %.4f | %3d | %3d | %3d | %3d | %.5f | %.5f | %.4f' % (l, r[:, 0].mean(), r[:, 1].mean(), (r[:, 1] > 0).sum(), (r[:, 1] > 0.05).sum(), (r[:, 1] > 0.10).sum(), (r[:, 1] > 0.20).sum(), r[:, 4].mean(), r[:, 3].mean(), r[:, 5].mean()))
    print('paired differences A3(lambda) - A2 (lambda=0), same 400 disturbance realizations:')
    for l in LAMS[1:]:
        d = R[l] - R[0]; dp = d[:, 0]; dw = d[:, 4]
        h = lambda x: tdist.ppf(.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
        print('  lambda %.1f: max|d peak| %.4f deg, mean d peak %+.4f (+-%.4f); trajectories with identical frac %d/%d; loss(any) flips A2->A3: %d lost / %d gained; mean d tracking %+.2e (+-%.1e) (rel. %.2f%%); mean d w_rms %+.2e' % (
            l, np.abs(dp).max(), dp.mean(), h(dp), int((d[:, 1] == 0).sum()), n, int(((R[0][:, 1] > 0) & (R[l][:, 1] == 0)).sum()), int(((R[0][:, 1] == 0) & (R[l][:, 1] > 0)).sum()), dw.mean(), h(dw), 100 * dw.mean() / R[0][:, 4].mean(), d[:, 3].mean()))
    for lname, rthr in (('any violation (frac>0)', 0.0), ('time-fraction above limit > 0.05', 0.05), ('time-fraction above limit > 0.10', 0.10), ('time-fraction above limit > 0.20', 0.20)):
        K = {l: int((R[l][:, 1] > rthr).sum()) for l in LAMS}
        print('--- loss: %s;  k=%s' % (lname, K))
        print('  CP upper bound on risk at confidence 1-delta/6 (Bonferroni): ' + ', '.join('l=%g: %.4f' % (l, cp_ub(K[l], n, DELTA / 6)) for l in LAMS))
        for al in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            pb = {l: p_binom(K[l], n, al) for l in LAMS}; ph = {l: p_hb(K[l], n, al) for l in LAMS}
            bon = [l for l in LAMS if pb[l] <= DELTA / 6]; bon_hb = [l for l in LAMS if ph[l] <= DELTA / 6]
            seq = []
            for l in sorted(LAMS, reverse=True):          # fixed sequence, most conservative first
                if pb[l] <= DELTA: seq.append(l)
                else: break
            cost = lambda l: R[l][:, 4].mean()
            pick = lambda S: ('lambda=%g (cost %.5f)' % (min(S, key=lambda l: (cost(l), l)), cost(min(S, key=lambda l: (cost(l), l))))) if S else 'none'
            print('  alpha=%.2f: p(binom) %s | Bonferroni(delta/6) certified %s -> min-cost %s | HB-Bonferroni %s | fixed-seq from lambda=3 certified %s -> %s' % (al, '/'.join('%.1e' % pb[l] for l in LAMS), bon, pick(bon), bon_hb, sorted(seq), pick(seq)))

# A2 (lambda=0): identical across limits 23 and 30?  then loss at any limit L in [23,30] can be read from the peak array
R23 = np.load(OUT + f'd3_lim23_lam0_n{n}_s{seed}.npy'); R30 = np.load(OUT + f'd3_lim30_lam0_n{n}_s{seed}.npy')
print('A2 trajectories at limit 23 vs limit 30: max |d peak| = %.2e deg, max |d tracking| = %.2e, max |d w_rms| = %.2e' % (np.abs(R23[:, 0] - R30[:, 0]).max(), np.abs(R23[:, 4] - R30[:, 4]).max(), np.abs(R23[:, 3] - R30[:, 3]).max()))
pk = np.sort(R30[:, 0]); print('A2 peak line angle over 400 runs: min %.3f median %.3f 90%% %.3f 99%% %.3f max %.3f deg' % (pk[0], np.median(pk), np.quantile(pk, .9), np.quantile(pk, .99), pk[-1]))
for al in (0.05, 0.10, 0.20):
    kmax = max(k for k in range(n) if binom.cdf(k, n, al) <= DELTA)
    L = pk[n - kmax - 1]      # limit L such that at most kmax peaks exceed L : (kmax+1)-th largest peak
    print('A2 alone (fixed policy, no Bonferroni), P(any violation of limit L) <= %.2f at confidence 0.90: need #exceed <= %d of %d -> smallest certifiable L = %.3f deg (peaks strictly above L: %d)' % (al, kmax, n, L, int((pk > L).sum())))
print('Pr[peak > 23] empirical = %.3f; CP95 = [%.3f, %.3f]' % ((pk > 23).mean(), beta.ppf(.025, (pk > 23).sum(), n - (pk > 23).sum() + 1), beta.ppf(.975, (pk > 23).sum() + 1, n - (pk > 23).sum())))
