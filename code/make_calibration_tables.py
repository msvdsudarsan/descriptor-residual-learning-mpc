"""Tables and figure for the calibration and baseline checks (reads results/*_conformal_independent.csv, *_cl_recal_*.csv, *_param_baseline_*.csv)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
# ---- A: independent calibration sets
rows = []; fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7)); plt.rcParams.update({'font.size': 8})
for j, nm in enumerate(['case9', 'case39']):
    d = pd.read_csv(R / f'{nm}_conformal_independent.csv'); ns = sorted(d.n_cal.unique()); g = []; m = []; e = []; s = []
    for n in ns:
        a = d[(d.n_cal == n) & (d.quantity == 'marginal step 3')].iloc[0]; js = d[(d.n_cal == n) & d.quantity.str.contains('scaled')].iloc[0]; jp = d[(d.n_cal == n) & d.quantity.str.contains('not guaranteed')].iloc[0]
        rows.append(f'{nm} & {n} & {int(a.n_sets)} & {a.guaranteed_level:.3f} & ${a.mean_coverage:.3f}\\pm{a.ci95_half_width:.3f}$ & ${js.mean_coverage:.3f}\\pm{js.ci95_half_width:.3f}$ & {jp.mean_coverage:.3f} & {a.shifted_coverage:.3f} \\\\')
        g.append(a.guaranteed_level); m.append(a.mean_coverage); e.append(a.ci95_half_width); s.append(a.shifted_coverage)
    x = np.arange(len(ns)); ax[j].bar(x - 0.27, g, 0.27, color='#bbbbbb', label='guaranteed level'); ax[j].bar(x, m, 0.27, yerr=e, color='#2ca02c', label='independent test pool'); ax[j].bar(x + 0.27, s, 0.27, color='#d62728', label='doubled load fluctuation')
    ax[j].set_xticks(x); ax[j].set_xticklabels([f'n={n}' for n in ns]); ax[j].set_ylim(0, 1.05); ax[j].set_title(nm); ax[j].set_ylabel('coverage at step 3'); ax[j].grid(alpha=.25)
ax[0].legend(fontsize=6.5, loc='center left'); fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'Figure_conformal.pdf', bbox_inches='tight'); fig.savefig(ROOT / 'figures' / 'Figure_conformal.png', dpi=300, bbox_inches='tight')
(R / 'table_conformal_independent.tex').write_text(r'\begin{tabular}{lccccccc}' + '\n' + r'\toprule' + '\n' + r'network & $n$ & sets & guaranteed & marginal, step 3 & joint, scaled & joint, per-step & doubled load\\' + '\n' + r'\midrule' + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')
# ---- B: policy-matched recalibration
rows = []
for nm in ['case9', 'case39']:
    d = pd.read_csv(R / f'{nm}_cl_recal_coverage.csv'); old = pd.read_csv(R / f'{nm}_tube_envelope_stress.csv').eps_line_angle_rad.values; new = pd.read_csv(R / f'{nm}_tube_envelope_stress_recal.csv').eps_line_angle_rad.values
    mg = d[d.quantity.str.startswith('marginal step') & d.calibration.str.startswith('policy')]; js = d[d.quantity.str.contains('scaled')].iloc[0]; jp = d[d.quantity.str.contains('not guaranteed')].iloc[0]; pr = d[d.quantity.str.contains('marginal step 3 on the test')].iloc[0]; prj = d[d.calibration.str.startswith('production envelope (n=40)')].iloc[0]
    rows.append(f'{nm} & {np.degrees(old[5]):.1f} / 100\\% & {np.degrees(new[5]):.1f} & {mg.coverage.min():.2f}--{mg.coverage.max():.2f} & ${js.coverage:.3f}\\pm{js.ci95_half_width:.3f}$ & {jp.coverage:.3f} & {pr.coverage:.2f} / {prj.coverage:.2f} \\\\')
(R / 'table_cl_recal.tex').write_text(r'\begin{tabular}{lcccccc}' + '\n' + r'\toprule' + '\n' + r'network & original $\varepsilon_5$ [deg] / coverage & new $\varepsilon_5$ [deg] & marginal, $n=20$ & joint, scaled & joint, per-step & $n=40$: step 3 / joint\\' + '\n' + r'\midrule' + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')
# ---- C: parametric baseline (prediction in three scenarios; closed-loop speed deviation with paired interval)
from scipy.stats import t as _t
rows = []
for nm in ['case9', 'case39']:
    p = pd.read_csv(R / f'{nm}_param_baseline_prediction.csv'); cl = pd.read_csv(R / f'{nm}_param_baseline_closed_loop.csv'); w = cl.groupby('controller').w_rms.mean()
    for k in ['library residual (paper)', 'damping + constant', 'damping + cubic speed + constant']:
        q = p[p.correction == k].set_index('scenario').rmse_ratio_to_A0; row = f'{nm} & {k} & {q["in-family"]:.3f} & {q["saliency x1.5"]:.3f} & {q["load lag 1.0 s"]:.3f} & {w[k]:.4f}'
        if k != 'library residual (paper)':
            a = cl[cl.controller == k].sort_values('run').w_rms.values; b = cl[cl.controller == 'library residual (paper)'].sort_values('run').w_rms.values; d = a - b
            row += f' & ${d.mean():+.4f}\\pm{_t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)):.4f}$'
        else: row += ' & --'
        rows.append(row + r' \\')
    rows.append(f'{nm} & physics only (A0) / static law (C0) & 1 & 1 & 1 & {w["A0 physics only"]:.4f} / {w["C0 static law"]:.4f} & -- ' + chr(92) * 2)
    rows.append(r'\midrule' if nm == 'case9' else '')
(R / 'table_param_baseline.tex').write_text(r'\begin{tabular}{llcccrc}' + '\n' + r'\toprule' + '\n' + r'network & correction & \multicolumn{3}{c}{prediction RMSE / A0} & $\omega_{\rm rms}$ [rad/s] & difference to library (paired)\\' + '\n' + r'\cmidrule(lr){3-5}' + '\n' + r' & & in-family & saliency $\times1.5$ & load lag 1\,s & & \\' + '\n' + r'\midrule' + '\n' + '\n'.join(x for x in rows if x) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')
# ---- D: A3 with the recalibrated envelope against A2
rows = []
for nm in ['case9', 'case39']:
    p = pd.read_csv(R / f'{nm}_cl_recal_performance.csv'); g = p.groupby(['scenario', 'controller']).mean(numeric_only=True)
    for sc in p.scenario.unique():
        for c_ in ('C0', 'A2', 'A3'):
            r = g.loc[(sc, c_)]; rows.append(f'{nm} & {sc.replace("limit ", "limit $").replace(" deg", "^\\circ$")} & {c_} & {r.w_rms:.4f} & {r.peak_deg:.2f} & {r.viol_integral_rad_s:.4f} & {100 * r.frac_time_over_limit:.2f} & {r.solve_ms:.1f} \\\\')
    rows.append(r'\midrule' if nm == 'case9' else '')
(R / 'table_a3_vs_a2.tex').write_text(r'\begin{tabular}{lllccccc}' + '\n' + r'\toprule' + '\n' + r'network & scenario & ctrl. & $\omega_{\rm rms}$ [rad/s] & peak [deg] & viol. [rad\,s] & time over limit [\%] & solve [ms]\\' + '\n' + r'\midrule' + '\n' + '\n'.join(x for x in rows if x) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')
for f in ('table_conformal_independent.tex', 'table_cl_recal.tex', 'table_param_baseline.tex', 'table_a3_vs_a2.tex'): print((R / f).read_text())
