"""Holm-adjusted analysis of the 60 cellwise verdicts of the regime map (reads only the stored paired CSV files).
For every comparison the verdict is re-expressed as a one-sided test against the pre-set margin M:
  'first better'  : H0 diff >= -M   -> p = P(T <= (mean+M)/se)
  'equivalent'    : TOST, p = max(P(T >= (mean+M)/se)... i.e. max of the two one-sided p-values
  'INCONCLUSIVE'  : no test claimed (p = 1)
Holm step-down at family-wise alpha = 0.05 is applied across all 60 comparisons (the 95% CI rule corresponds to a
per-test alpha of 0.025 per side, so the unadjusted baseline is also reported).  Nothing in the manuscript is changed."""
import csv, sys, numpy as np
from scipy import stats
import json, collections
R = sys.argv[1] if len(sys.argv) > 1 else 'results/regime_map/'
rows = []
def verdict(lo, hi, M): return 'equivalent' if (lo > -M and hi < M) else ('first better' if hi < -M else ('second better' if lo > M else 'INCONCLUSIVE'))
def add(case, kind, cell, cmp_, diff, M):
    n = len(diff); mu = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
    lo, hi = mu - t * se, mu + t * se; v = verdict(lo, hi, M)
    if v == 'equivalent': p = max(stats.t.sf((mu + M) / se, n - 1), stats.t.cdf((mu - M) / se, n - 1))
    elif v == 'first better': p = stats.t.cdf((mu + M) / se, n - 1)
    elif v == 'second better': p = stats.t.sf((mu - M) / se, n - 1)
    else: p = 1.0
    rows.append(dict(case=case, kind=kind, cell=cell, cmp=cmp_, n=n, mu=mu, lo=lo, hi=hi, M=M, verdict=v, p=p))
for nm in ('case9', 'case39'):
    pick = json.load(open(R + f'{nm}_prediction_meta.json'))['selection']; d = collections.defaultdict(dict)
    for x in csv.DictReader(open(R + f'{nm}_prediction_runs.csv')): d[(x['regime'], x['setting'])].setdefault(int(x['run']), {})[x['model']] = np.sqrt(float(x['sumsq']) / float(x['n']))
    for (k, s_), runs in d.items():
        ps = pick[k + s_]['pick']; rat = lambda m: np.array([runs[i][m] / runs[i]['A0'] for i in sorted(runs)])
        for a_, b_ in dict.fromkeys((('P2', ps), ('P2', 'P1c'), ('P3', 'P2'))): add(nm, 'pred', k + s_, f'{a_}-{b_}', rat(a_) - rat(b_), 0.02)
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for x in csv.DictReader(open(R + f'{nm}_closed_loop_runs.csv')): d[(x['regime'], x['authority'])][x['controller']][int(x['run'])] = x
    f = lambda cs, c, k: np.array([float(cs[c][i][k]) for i in sorted(cs[c])])
    for (g, a_), cs in d.items():
        mg = 0.02 * f(cs, 'P1*', 'w_rms').mean()
        for x_, y_ in (('P2', 'P1*'), ('P2', 'C0eq'), ('P2', 'C0'), ('P3', 'P2'), ('P2', 'A0')): add(nm, 'cl', f'{g}x{a_}', f'{x_}-{y_}', f(cs, x_, 'w_rms') - f(cs, y_, 'w_rms'), mg)
m = len(rows); order = np.argsort([x['p'] for x in rows]); adj = np.empty(m); run = 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m - k) * rows[i]['p'])); adj[i] = run
for x, a in zip(rows, adj): x['p_holm'] = a
out = open(R + 'multiplicity_holm.csv', 'w'); w = csv.writer(out); w.writerow(['case', 'kind', 'cell', 'comparison', 'n', 'verdict', 'p_unadjusted', 'p_holm', 'survives_holm_0.05'])
for x in rows: w.writerow([x['case'], x['kind'], x['cell'], x['cmp'], x['n'], x['verdict'], f"{x['p']:.3g}", f"{x['p_holm']:.3g}", x['p_holm'] < 0.05 if x['verdict'].lower() != 'inconclusive' else 'n/a'])
out.close()
print('comparisons', m)
for v in ('equivalent', 'first better', 'second better', 'INCONCLUSIVE'):
    s = [x for x in rows if x['verdict'].lower() == v.lower()]
    if s: print(f"{v:14s} n={len(s):3d}  p<0.05 unadjusted: {sum(x['p'] < .05 for x in s):3d}   Holm<0.05: {sum(x['p_holm'] < .05 for x in s):3d}")
print('--- non-equivalence findings (superiority) ---')
for x in rows:
    if x['verdict'].lower() in ('first better', 'second better'): print(x['case'], x['kind'], x['cell'], x['cmp'], x['verdict'], f"p={x['p']:.3g} holm={x['p_holm']:.3g}")
print('--- equivalence verdicts that do NOT survive Holm ---')
for x in rows:
    if x['verdict'].lower() == 'equivalent' and x['p_holm'] >= .05: print(x['case'], x['kind'], x['cell'], x['cmp'], f"n={x['n']} p={x['p']:.3g} holm={x['p_holm']:.3g}")
