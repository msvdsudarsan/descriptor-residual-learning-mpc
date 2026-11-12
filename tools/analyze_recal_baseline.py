"""Analysis for BASELINE_PROTOCOL: paired run-level differences, Student-t 95% CI, practical-equivalence verdict (margins locked: prediction 0.02 absolute; closed-loop w_rms 2% of the mean P1 value).
No resampling. Usage: python3 tools/analyze_recal_baseline.py"""
import csv, json, numpy as np
from pathlib import Path
from scipy.stats import t as tdist
R = Path(__file__).resolve().parents[1] / 'results' / 'baseline_recal'
DP, DC = 0.02, 0.02


def verdict(lo, hi, d):
    if lo >= -d and hi <= d: return 'EQUIVALENT (CI inside margin)'
    if hi < -d: return 'first model better (CI below -margin)'
    if lo > d: return 'second model better (CI above +margin)'
    return 'INCONCLUSIVE'


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h, h


out = []; lines = []
for nm in ('case9', 'case39'):
    f = R / f'{nm}_prediction_runs.csv'
    if not f.exists(): continue
    rows = list(csv.DictReader(open(f))); sel = json.load(open(R / f'{nm}_prediction_selection.json'))['selected_P1']
    P1 = {'P1a damping + constant': 'P1a', 'P1b damping + cubic speed + constant': 'P1b', 'P1c physical recalibration': 'P1c'}[sel]
    lines.append(f'\n## {nm} prediction (selected P1* by validation = {P1})')
    ratio = {}
    for r in rows:
        ratio.setdefault((r['scenario'], r['model']), {})[int(r['run'])] = float(r['sumsq']), int(r['n'])
    for scen in ('in-family', 'saliency x1.5', 'load lag 1.0 s'):
        per = lambda m: np.array([np.sqrt(ratio[(scen, m)][i][0] / ratio[(scen, m)][i][1]) / np.sqrt(ratio[(scen, 'A0')][i][0] / ratio[(scen, 'A0')][i][1]) for i in sorted(ratio[(scen, m)])])
        pooled = lambda m: np.sqrt(sum(v[0] for v in ratio[(scen, m)].values()) / sum(v[1] for v in ratio[(scen, m)].values()) / (sum(v[0] for v in ratio[(scen, 'A0')].values()) / sum(v[1] for v in ratio[(scen, 'A0')].values())))
        lines.append(f'- {scen}: pooled ratio to A0: ' + ', '.join(f'{m} {pooled(m):.4f}' for m in ('P2 library residual', 'P1a', 'P1b', 'P1c', 'P3')))
        for a, b in tuple(dict.fromkeys((('P2 library residual', P1), ('P2 library residual', 'P1c'), ('P2 library residual', 'P1a'), ('P3', P1), ('P3', 'P2 library residual')))):
            d = per(a) - per(b); m, lo, hi, h = tci(d); v = verdict(lo, hi, DP)
            lines.append(f'    {a} - {b}: mean d {m:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] (n={len(d)}, half-width {h:.4f}, margin {DP}) -> {v}')
            out.append((nm, 'prediction', scen, a, b, len(d), m, lo, hi, h, DP, v))
    f = R / f'{nm}_closed_loop_runs.csv'
    if not f.exists(): continue
    rows = list(csv.DictReader(open(f))); cs = json.load(open(R / f'{nm}_closed_loop_selection.json'))
    pickc = min([('P1a', cs['P1a']), ('P1b', cs['P1b']), ('P1c', cs['validation_weak_form_rmse_P1c'])], key=lambda x: x[1])[0]
    cname = {'P1a': 'damping + constant', 'P1b': 'damping + cubic speed + constant', 'P1c': 'P1c physical recalibration'}
    W, E = {}, {}
    for r in rows:
        W.setdefault(r['controller'], {})[int(r['run'])] = float(r['w_rms']); E.setdefault(r['controller'], {})[int(r['run'])] = float(r['input_energy'])
    runs = sorted(W['library residual (paper)'])
    g = lambda D, k: np.array([D[k][i] for i in runs])
    lines.append(f'\n## {nm} closed loop (selected P1* by validation = {pickc}; n={len(runs)} runs)')
    lines.append('- mean w_rms: ' + ', '.join(f'{k} {g(W, k).mean():.4f}' for k in W) + '\n- mean input energy: ' + ', '.join(f'{k} {g(E, k).mean():.3f}' for k in E))
    P2 = 'library residual (paper)'
    for a, b in dict.fromkeys(((P2, cname[pickc]), (P2, 'P1c physical recalibration'), ('P3 residual on P1c', cname[pickc]), ('P3 residual on P1c', P2), (P2, 'C0 energy-matched'), (P2, 'C0 static law'))):
        d = g(W, a) - g(W, b); m, lo, hi, h = tci(d); dm = DC * g(W, b).mean(); v = verdict(lo, hi, dm)
        lines.append(f'    {a} - {b}: per-seed d {np.round(d, 4).tolist()}, mean {m:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] (half-width {h:.4f}, margin {dm:.4f}) -> {v}; energy ratio {g(E, a).mean() / g(E, b).mean():.3f}')
        out.append((nm, 'closed_loop', 'seed-level', a, b, len(d), m, lo, hi, h, dm, v))
with open(R / 'recal_baseline_summary.csv', 'w') as f:
    f.write('case,part,scenario,model_a,model_b,n,mean_d,ci_lo,ci_hi,half_width,margin,verdict\n')
    for r in out: f.write(','.join(str(x) for x in r) + '\n')
(R / 'recal_baseline_summary.md').write_text('\n'.join(lines))
print('\n'.join(lines))
