"""Independent-calibration check of the trajectory-level split-conformal envelope (replaces the resampling design of
network_conformal_validation.py, whose 2000 splits were drawn from one finite pool and are therefore not independent experiments).

Design (static law, nominal loading, reduced model, deterministic plant: a trajectory is a function of its random initial state and load):
  * R disjoint calibration sets of n_base fresh trajectories each (seeds 50000+i); sets of size n_base/2 and 2*n_base are built from the same
    disjoint trajectories (first half; two consecutive sets), so every size has independent sets;
  * one fresh test pool of M trajectories (seed 60000) that is disjoint from every calibration trajectory;
  * a pilot set of 10 trajectories (seed 40000) that fixes the step weights w_k = mean pilot score, used only for the joint envelope
    eps_k^joint = lambda * w_k, where lambda is the r-th smallest of max_k s_k/w_k over the calibration set (a scalar score, hence valid for joint coverage);
  * a shifted pool with about twice the load fluctuation (seed 70000).
Coverage of one calibration set = fraction of the M test trajectories whose score is below the envelope.  Intervals: t-interval over the R independent sets for
the mean coverage; Clopper-Pearson interval for the coverage of the shifted pool pooled over sets is NOT given (the shifted pool is shared)."""
import sys, json
import numpy as np
from scipy.stats import t as tdist, beta
from network_validity_gate import *

ALPHA = 0.05


def scores(c, th_eq, de_eq, coef, n, seed, lamp_hi=0.06, lamp_lo=0.0, T=4.0):
    return envelope_traj(c, make_test_runs(c, th_eq, de_eq, n, seed, lamp_hi, T=T, lamp_lo=lamp_lo), coef)


def eps_of(S, alpha=ALPHA):
    return np.array([conformal_quantile(S[:, k], alpha) for k in range(S.shape[1])])


def run(name, nbase, R, M, nshift):
    c, th_eq, de_eq, coef = nominal_fit(name)
    pilot = scores(c, th_eq, de_eq, coef, 10, 40000); w = pilot.mean(axis=0)
    cal = [scores(c, th_eq, de_eq, coef, nbase, 50000 + i) for i in range(R)]; print(name, 'calibration sets done', flush=True)
    test = scores(c, th_eq, de_eq, coef, M, 60000); shifted = scores(c, th_eq, de_eq, coef, nshift, 70000, lamp_hi=0.14, lamp_lo=0.10)
    sets = {nbase // 2: [S[: nbase // 2] for S in cal], nbase: cal, 2 * nbase: [np.vstack(cal[2 * i:2 * i + 2]) for i in range(R // 2)]}
    rows = []
    for n, family in sorted(sets.items()):
        r = int(np.ceil((n + 1) * (1 - ALPHA))); lvl = r / (n + 1) if r <= n else n / (n + 1)
        cm, cj, sm, sj = [], [], [], []
        for S in family:
            eps = eps_of(S); ok = test <= eps[None, :]; cm.append(ok.mean(axis=0)); okj = (test / w[None, :]).max(axis=1)
            lam = np.sort((S / w[None, :]).max(axis=1))[min(r, n) - 1]; cj.append(np.mean(okj <= lam))
            sm.append((shifted <= eps[None, :]).mean(axis=0)); sj.append(np.mean((shifted / w[None, :]).max(axis=1) <= lam))
            # joint coverage of the per-step envelope (not guaranteed): all five steps below their own eps
        cm = np.array(cm); cj = np.array(cj); sm = np.array(sm); sj = np.array(sj); Rn = len(family); tq = tdist.ppf(0.975, Rn - 1)
        pj = np.mean([np.mean(np.all(test <= eps_of(S)[None, :], axis=1)) for S in family])
        sd_th = np.sqrt(beta.var(r, n + 1 - r)) if r <= n else np.nan
        for k in range(cm.shape[1]):
            rows.append((n, Rn, lvl, 'marginal step %d' % (k + 1), cm[:, k].mean(), tq * cm[:, k].std(ddof=1) / np.sqrt(Rn), cm[:, k].std(ddof=1), sd_th, sm[:, k].mean()))
        rows.append((n, Rn, lvl, 'steps 1-5 jointly; per-step envelopes (not guaranteed)', pj, np.nan, np.nan, np.nan, np.nan))
        rows.append((n, Rn, lvl, 'steps 1-5 jointly; scaled envelope (guaranteed)', cj.mean(), tq * cj.std(ddof=1) / np.sqrt(Rn), cj.std(ddof=1), sd_th, sj.mean()))
        print(name, 'n', n, 'sets', Rn, 'level', round(lvl, 4), 'marginal mean', np.round(cm.mean(axis=0), 3), 'joint scaled', round(cj.mean(), 3), 'joint per-step', round(pj, 3),
              'shifted', np.round(sm.mean(axis=0), 3), flush=True)
    csv(f'{name}_conformal_independent.csv', ['n_cal', 'n_sets', 'guaranteed_level', 'quantity', 'mean_coverage', 'ci95_half_width', 'sd_over_sets', 'sd_beta_theory', 'shifted_coverage'], rows)
    json.dump(dict(case=name, nbase=nbase, R=R, M=M, nshift=nshift, seeds=dict(pilot=40000, calibration='50000+i', test=60000, shifted=70000), w=w.tolist()), open(RES / f'{name}_conformal_independent_meta.json', 'w'))


if __name__ == '__main__':
    nm = sys.argv[1]
    if nm == 'case9': run('case9', nbase=20, R=400, M=2000, nshift=200)
    else: run('case39', nbase=20, R=160, M=1000, nshift=100)
