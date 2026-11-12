"""Tables and figure from results/*.json (no new simulation).  Usage (repo root): PYTHONPATH=code python3 extensions/ac_recal/summarize_ac_recal.py"""
import json, glob, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import bootstrap_ci
HERE = Path(__file__).resolve().parent; R = HERE / 'results'; OUT = HERE / 'tables'; OUT.mkdir(exist_ok=True)
pts = [json.load(open(f)) for f in sorted(glob.glob(str(R / '*.json')))]
order = {'case9': 0, 'case39': 1, 'pq': 0, 'z': 1}; pts.sort(key=lambda d: (order[d['case']], order[d['loadmodel']], d['lam']))
def rule(lo, hi):
    if hi < 0.95: return 'adds value'
    if lo >= 0.95 and hi <= 1.05: return 'equivalent'
    if lo > 1.05: return 'recalibration better'
    return 'inconclusive'
rows = []; sec = []
for d in pts:
    if d['status'] != 'OK':
        rows.append(dict(case=d['case'], load_model=d['loadmodel'], lam=d['lam'], status='FAILED: ' + d['reason'])); continue
    s = d['summary']; p = s['paired']; ok = d['n_common_runs'] > 0
    row = dict(case=d['case'], load_model=d['loadmodel'], lam=d['lam'], status='OK', sigma_min_Jg=round(d['sigma_min_Jg'], 3), V_min=round(d['V_min'], 4), runs_used=d['n_common_runs'], runs_excluded=d['n_excluded_runs'], gens_at_bound=d['recal_gens_at_bound'])
    for m in ('A0', 'P0', 'P1', 'P2', 'P3'): row[f'rmse1s_{m}'] = round(s[m]['rmse_1s'], 4)
    for m in ('A0', 'P0', 'P1', 'P2', 'P3'): row[f'Vrms1s_{m}'] = round(s[m]['V_rms_1s'], 5)
    for m in ('A0', 'P1', 'P2', 'P3'): row[f'gmax_max_{m}'] = float(f"{s[m]['gmax_max']:.2e}")
    for m in ('P0', 'P1', 'P2', 'P3'): row[f'newton_fail_{m}'] = s[m]['newton_fail_windows']
    for k in ('P2/P1', 'P3/P1'):
        v = p[f'{k} mse_pooled']; row[k + ' pooled'] = round(v['mean'], 3); row[k + ' lo'] = round(v['lo'], 3); row[k + ' hi'] = round(v['hi'], 3); row[k + ' rule'] = rule(v['lo'], v['hi'])
        w = p[f'{k} mse_1s']; row[k + ' 1s'] = round(w['mean'], 3); row[k + ' 1s lo'] = round(w['lo'], 3); row[k + ' 1s hi'] = round(w['hi'], 3)
    v = p['P1/A0 mse_pooled']; row['P1/A0 pooled'] = round(v['mean'], 3); row['P2/A0 pooled'] = round(p['P2/A0 mse_pooled']['mean'], 3)
    # secondary (comparable with the main paper's margin 0.02 on the ratio to A0): per-run difference P1/A0 - P2/A0 and P1/A0 - P3/A0 (pooled)
    pa = np.array(p['P1/A0 mse_pooled']['per_run']); p2a = np.array(p['P2/A0 mse_pooled']['per_run'])
    p3a = pa * np.array(p['P3/P1 mse_pooled']['per_run'])
    for nm, arr in (('P2', p2a), ('P3', p3a)):
        m_, lo_, hi_ = bootstrap_ci(pa - arr); row[f'diff P1-{nm} (ratio to A0)'] = round(m_, 4); row[f'diff P1-{nm} lo'] = round(lo_, 4); row[f'diff P1-{nm} hi'] = round(hi_, 4)
    row['t_fit_recal_s'] = round(d['t_fit_recal_s'], 2); row['t_fit_residual_s'] = round(d['t_fit_residual_s'], 1); row['t_fit_P3_s'] = round(d['t_fit_P3_s'], 1)
    for m in ('A0', 'P1', 'P2', 'P3'): row[f'ms_per_window_{m}'] = round(1e3 * s[m]['sec_per_window'], 3)
    row['active_terms_P2'] = d['active_terms_P2']; row['active_terms_P3'] = d['active_terms_P3']
    rows.append(row)
cols = []; [cols.append(k) for r in rows for k in r if k not in cols]
with open(OUT / 'ac_recal_all_points.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r['status'] == 'OK']
print('points', len(rows), 'failed', len(rows) - len(ok), 'excluded runs', sum(r['runs_excluded'] for r in ok), 'newton failures', sum(r[f'newton_fail_{m}'] for r in ok for m in ('P0', 'P1', 'P2', 'P3')))
from collections import Counter
for k in ('P2/P1', 'P3/P1'): print(k, dict(Counter(r[k + ' rule'] for r in ok)))
for case in ('case9', 'case39'):
    for lm in ('pq', 'z'):
        sub = [r for r in ok if r['case'] == case and r['load_model'] == lm]
        print(case, lm, 'P1/A0', min(r['P1/A0 pooled'] for r in sub), max(r['P1/A0 pooled'] for r in sub), '| P3/P1', min(r['P3/P1 pooled'] for r in sub), max(r['P3/P1 pooled'] for r in sub),
              '| diff P1-P3 (A0 ratio) max', max(r['diff P1-P3 (ratio to A0)'] for r in sub), 'hi', max(r['diff P1-P3 hi'] for r in sub))
print('gmax_max projected', max(r[f'gmax_max_{m}'] for r in ok for m in ('P1', 'P2', 'P3')), 'A0', max(r['gmax_max_A0'] for r in ok))
print('ms/window P1,P2,P3', np.mean([r['ms_per_window_P1'] for r in ok]), np.mean([r['ms_per_window_P2'] for r in ok]), np.mean([r['ms_per_window_P3'] for r in ok]))
print('fit s recal / residual / P3', np.mean([r['t_fit_recal_s'] for r in ok]), np.mean([r['t_fit_residual_s'] for r in ok]), np.mean([r['t_fit_P3_s'] for r in ok]))
print('V rms 1s (A0 vs P3) mean', np.mean([r['Vrms1s_A0'] for r in ok]), np.mean([r['Vrms1s_P3'] for r in ok]))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, case in zip(ax, ('case9', 'case39')):
    for lm, mk, ls in (('pq', 'o', '-'), ('z', 's', '--')):
        sub = [r for r in ok if r['case'] == case and r['load_model'] == lm]; x = [r['lam'] for r in sub]
        a.plot(x, [r['P3/P1 pooled'] for r in sub], ls, marker=mk, color='k', label=f"P3/P1, {'constant power' if lm == 'pq' else 'constant impedance'}")
        a.fill_between(x, [r['P3/P1 lo'] for r in sub], [r['P3/P1 hi'] for r in sub], color='0.8', alpha=0.6)
    a.axhline(1.0, color='0.4', lw=0.8); a.axhline(0.95, color='0.4', lw=0.8, ls=':'); a.set_title(case); a.set_xlabel('loading factor')
ax[0].set_ylabel('paired RMSE ratio P3/P1'); ax[0].legend(fontsize=7, loc='lower right'); ax[0].set_ylim(0.8, 1.02)
fig.tight_layout(); fig.savefig(OUT / 'Figure_ac_recal.pdf'); fig.savefig(OUT / 'Figure_ac_recal.png', dpi=150)
