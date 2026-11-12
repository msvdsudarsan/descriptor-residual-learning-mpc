import sys, math
sys.path.insert(0, 'code')
from gatelib import *
from scipy.stats import t as tdist
name = sys.argv[1]; kap0 = float(sys.argv[2]); kfd_s = eval(sys.argv[3])      # kappa0 and {mu: kappa_fd} copied from the step_delay log
q0 = dict(case9=0.023839465238203714, case39=0.03706069563617859)[name]; A_STAR = 0.25
B = float(np.load(OUT / f'q_pool95_{name}.npy')[2])         # conformal 95% quantile of per-run max UNSMOOTHED null score (pool)
Z = np.load(OUT / f'steps_{name}.npz'); mus = sorted({float(k[2:]) for k in Z.files if k.startswith('S_')})
KS = 30; rng = np.random.default_rng(5)
def mci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; h = tdist.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), h
S0 = Z[f'S_{mus[0]}']; pre = S0[:, 10:KS + 1]; s0sq = float((pre ** 2).mean()); s0bar = float(pre.mean())
print('%s: pre-step score: mean %.5f rms %.5f (q0 = %.5f; q0/mean = %.2f); kappa0 = %.4f; B (95%% conformal, max unsmoothed null score) = %.5f' % (name, s0bar, math.sqrt(s0sq), q0, q0 / s0bar, kap0, B))
thr_alarm, thr_trust, thr_fb = 1.0, 1 + math.log(1 / 0.8) / A_STAR, 1 + math.log(2) / A_STAR
print('thresholds on q = smoothed/q0: alarm 1; tau<0.8 at %.2f; tau<0.5 (a=0.25) at %.2f' % (thr_trust, thr_fb))
def first_cross(Q, thr):            # Q[:, k] smoothed ratio, k >= KS+1; returns delay in samples (nan if none) 
    out = np.full(len(Q), np.nan)
    for i, q in enumerate(Q):
        idx = np.where(q[KS + 1:] > thr)[0]
        if len(idx): out[i] = idx[0] + 1
    return out
def plug_in(spost, thr):
    for j in range(1, 40):
        m = ((3 - j) * s0bar + j * spost) / 3 if j < 3 else spost
        if m / q0 > thr: return j
    return float('nan')
def bound_delay(kd):
    for j in range(1, 40):
        L = (j / 3) * max(kd - B, 0) if j < 3 else max(kd - B, 0)
        if L > q0: return j
    return float('nan')
print('%-6s %-6s | %-27s | %-27s | %-22s | %-22s | %-18s' % ('step', 'N', 'mean score j=1..3 meas [CI half]', 'RSS-pred j>=3 (k0 | kfd) ; noise-free meas/(k0*D)', 'alarm delay (samples)', 'fallback tau<.5 delay', 'plug-in j* / bound'))
rows = []
for mu in mus:
    S = Z[f'S_{mu}']; Snf = Z[f'Snf_{mu}']; D = mu - 1; N = len(S)
    Q = np.full(S.shape, np.nan)
    for k in range(3, S.shape[1]): Q[:, k] = S[:, k - 2:k + 1].mean(axis=1) / q0
    mj = [mci(S[:, KS + j]) for j in (1, 2, 3)]; late = S[:, KS + 3:KS + 11].mean()
    sp0 = math.sqrt(s0sq + (kap0 * D) ** 2); spf = math.sqrt(s0sq + (kfd_s[mu] * D) ** 2)
    nfr = Snf[:, KS + 1] / (kap0 * abs(D))
    da = first_cross(Q, thr_alarm); df = first_cross(Q, thr_fb)
    ja = plug_in(sp0, thr_alarm); jf = plug_in(sp0, thr_fb); jlb = bound_delay(kfd_s[mu] * abs(D))
    det = int(np.isfinite(da).sum()); pre_alarm = float(np.mean(Q[:, KS] > 1))
    dm = np.nanmean(da); dmed = np.nanmedian(da); bs = [np.nanmean(da[rng.integers(0, N, N)]) for _ in range(2000)]
    p_le = lambda j: (lambda k: (k / N,) + cp_interval(k, N))(int(np.sum(da <= j)))
    dfm = np.nanmean(df) if np.isfinite(df).any() else float('nan')
    print('%+5.0f%% %-6d | %.4f %.4f %.4f (+-%.4f) | pred %.4f | %.4f ; nf ratio %.2f (sd %.2f) | alarm mean %.2f med %.0f bootCI [%.2f,%.2f] detected %d/%d pre-alarm@30 %.3f | fallback detected %d/%d mean %.2f | j*=%s fb j*=%s bound=%s' % (
        100 * D, N, mj[0][0], mj[1][0], mj[2][0], mj[0][1], sp0, spf, nfr.mean(), nfr.std(), dm, dmed, np.quantile(bs, .025), np.quantile(bs, .975), det, N, pre_alarm, int(np.isfinite(df).sum()), N, dfm, ja, jf, jlb))
    # coverage of the plug-in and of the bound
    if np.isfinite(ja):
        pj = p_le(int(ja)); print('        P(alarm delay <= plug-in j*=%d) = %.3f CP95 [%.3f,%.3f]; P(delay <= j*+1) = %.3f; late-mean score (j=4..10) %.4f vs pred %.4f' % (ja, pj[0], pj[1], pj[2], p_le(int(ja) + 1)[0], late, sp0))
    if np.isfinite(jlb):
        pb = p_le(int(jlb)); print('        bound (B+kappa_fd): guaranteed alarm by sample %d; empirical P(delay <= %d) = %.3f CP95 [%.3f,%.3f] (nominal >= 0.95 if B and linearity hold)' % (jlb, jlb, pb[0], pb[1], pb[2]))
    else: print('        bound gives no guarantee (kappa_fd*|D| - B = %.4f <= q0 = %.4f)' % (kfd_s[mu] * abs(D) - B, q0))
    # pointwise first-order check: noise-free innovation vs kappa D, and per-sample lower bound s >= kd - B
    kd = kfd_s[mu] * abs(D); vio = float(np.mean(S[:, KS + 3:KS + 11] < kd - B)) if kd > B else float('nan')
    print('        noise-free innovation / (kappa0|D|) at sample 31: mean %.3f; at 33: %.3f; fraction of noisy post-step scores (j=3..10) below kappa_fd|D| - B: %s' % (nfr.mean(), np.mean(Snf[:, KS + 3] / (kap0 * abs(D))), 'n/a' if not np.isfinite(vio) else '%.4f' % vio))
    rows.append((mu, D, mj[0][0], sp0, dm, ja, jlb))
