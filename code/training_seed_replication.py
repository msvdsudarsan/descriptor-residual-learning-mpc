"""Training-seed replication of the regime-map PREDICTION comparison P2 vs P1c (addresses the stated limitation that the
intervals omit training-seed variability).  Same test runs (SEED_TEST), same plants, same margins as the regime map;
only the training/validation seeds (and hence the identified coefficients) change.  Reads nothing from the stored results.
Usage (package root, PYTHONPATH=code): python3 training_seed_replication.py CASE OUTDIR SEEDPAIRS
  SEEDPAIRS = comma list of train seeds; validation seed = train seed + 1."""
import sys, json, csv, numpy as np
sys.path.insert(0, 'code')
from network_regime_map import *      # same helpers as the regime map
from scipy import stats
import csv as pycsv
name, outdir = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(',')]
c0 = build_case(name); th_eq, de_eq, _ = equilibrium(c0); rows = []; M = 0.02
for sd in seeds:
    base_tr = make_training(c0, th_eq, de_eq, 30, sd, 1e-3); base_vl = make_training(c0, th_eq, de_eq, 8, sd + 1, 1e-3); ib = identify(c0, th_eq, base_tr, base_vl)
    for kind in ('M0', 'M1', 'M2'):
        cp = regime_plant(c0, kind); runs = test_runs(cp, c0, th_eq, de_eq, NRUN_PRED[name], SEED_TEST, kind)
        for setting in ('a', 'b'):
            if kind == 'M0' and setting == 'b': continue
            if setting == 'a':
                tr = make_training_reg(cp, c0, th_eq, de_eq, 30, sd, 1e-3, kind); vl = make_training_reg(cp, c0, th_eq, de_eq, 8, sd + 1, 1e-3, kind); idn = identify(c0, th_eq, tr, vl)
            else: idn = ib
            c1c = idn['c1c']; z = np.zeros((c0['ng'], NF + 1))
            models = [('A0', c0, 'A0', idn['coef2']), ('P2', c0, 'A2', idn['coef2']), ('P1c', c1c, 'A2', z)]
            r = {m: [] for m, *_ in models}
            for run in runs:
                for mn, cp_, mode, co in models:
                    ss, n = pred_run(cp_, c0['M'], run, mode, co); r[mn].append(np.sqrt(ss / n))
            rat = lambda m: np.array(r[m]) / np.array(r['A0']); d = rat('P2') - rat('P1c'); n = len(d)
            h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); lo, hi = d.mean() - h, d.mean() + h
            v = 'equivalent' if (lo > -M and hi < M) else ('P2 better' if hi < -M else ('P1c better' if lo > M else 'INCONCLUSIVE'))
            rows.append((name, sd, kind + setting, n, round(d.mean(), 4), round(lo, 4), round(hi, 4), v)); print(*rows[-1], flush=True)
w = pycsv.writer(open(f'{outdir}/{name}_training_seed_replication.csv', 'w')); w.writerow(['case', 'train_seed', 'cell', 'n_test_runs', 'mean_P2_minus_P1c', 'ci_lo', 'ci_hi', 'verdict(margin 0.02)']); w.writerows(rows)
