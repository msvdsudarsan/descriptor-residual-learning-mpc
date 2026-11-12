"""Figure and table for the baseline / load-lag experiments (reads results/*_baselines_*.csv)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RES, FIG = ROOT / 'results', ROOT / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
M = ['OURS (sparse; weak form)', 'SF (sparse; strong form)', 'RIDGE (dense; weak form)', 'MLP (2x32; weak form)']
LAB = ['sparse, weak form (paper)', 'sparse, strong form', 'dense ridge, weak form', 'MLP 2x32, weak form']
COL = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
SC = ['in-family', 'load-lag tau=0.5', 'load-lag tau=1.0', 'saliency x1.5']; SCL = ['in-family', 'load lag 0.5 s', 'load lag 1.0 s', 'saliency x1.5']
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
for j, nm in enumerate(['case9', 'case39']):
    d = pd.read_csv(RES / f'{nm}_baselines_loaddyn.csv'); a = ax[0, j]; w = 0.2
    for k, m in enumerate(M):
        v = [d[(d.scenario == s) & (d.method == m)].ratio_to_A0.iloc[0] for s in SC]
        a.bar(np.arange(4) + (k - 1.5) * w, v, w, color=COL[k], label=LAB[k])
    a.set_xticks(range(4)); a.set_xticklabels(SCL, rotation=15); a.set_ylabel('1 s RMSE / physics-only A0'); a.set_title(f'{nm}: scenarios'); a.set_ylim(0, 0.52)
    e = pd.read_csv(RES / f'{nm}_baselines_noise.csv'); b = ax[1, j]
    for k, m in enumerate(M):
        s = e[e.method == m]; b.plot(s.noise_sigma, s.ratio_to_A0, 'o-', color=COL[k], label=LAB[k])
    b.set_xscale('log'); b.set_xlabel('measurement noise $\\sigma$ in training data'); b.set_ylabel('1 s RMSE / physics-only A0'); b.set_title(f'{nm}: noise sweep, in-family'); b.set_ylim(0, 0.3)
ax[0, 0].legend(fontsize=6.5, loc='upper left', ncol=2)
fig.tight_layout(); fig.savefig(FIG / 'Figure_baselines.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Figure_baselines.png', dpi=300, bbox_inches='tight')
rows = []
for nm in ['case9', 'case39']:
    d = pd.read_csv(RES / f'{nm}_baselines_loaddyn.csv')
    for s, sl in zip(SC, SCL):
        r = [d[(d.scenario == s) & (d.method == m)].ratio_to_A0.iloc[0] for m in M]
        rows.append(f'{nm} & {sl} & ' + ' & '.join(f'{x:.3f}' for x in r) + r' \\')
    rows.append(r'\midrule' if nm == 'case9' else '')
tex = (r'\begin{tabular}{llcccc}' + '\n' + r'\toprule' + '\n' + r'network & scenario & sparse, weak & sparse, strong & dense ridge & MLP \\' + '\n' + r'\midrule' + '\n' + '\n'.join(x for x in rows if x) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')
(RES / 'table_baselines.tex').write_text(tex); print(tex)
