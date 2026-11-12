"""Builds the robustness figure and table from the stored regime-map paired CSV (original training seed) and the
training-seed replication CSV.  Usage: python3 make_robustness_figure.py PAIRED_DIR REPL_DIR OUTDIR"""
import csv, sys, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P, Rd, O = sys.argv[1:4]
cells = ['M0a', 'M1a', 'M1b', 'M2a', 'M2b']; lab = {'M0a': 'M0', 'M1a': 'M1 identify', 'M1b': 'M1 deploy', 'M2a': 'M2 identify', 'M2b': 'M2 deploy'}
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
tab = []
for ax, nm in zip(axs, ('case9', 'case39')):
    orig = {}
    for r in csv.DictReader(open(f'{P}/{nm}_prediction_paired.csv')):
        if r['difference'] == 'P2-P1c': orig[r['regime'] + r['setting']] = (float(r['mean']), float(r['ci_lo']), float(r['ci_hi']), r['verdict(margin 0.02)'])
    rep = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{Rd}/{nm}_training_seed_replication.csv')): rep[r['cell']].append((float(r['mean_P2_minus_P1c']), float(r['ci_lo']), float(r['ci_hi']), r['verdict(margin 0.02)']))
    for i, c in enumerate(cells):
        pts = [orig[c] + ('orig',)] if c in orig else []
        pts = [(*orig[c],)] + rep[c] if c in orig else rep[c]
        for j, p in enumerate(pts):
            x = i + (j - (len(pts) - 1) / 2) * 0.12
            ax.plot([x, x], [p[1], p[2]], color='0.3', lw=1.2); ax.plot(x, p[0], 'o', ms=3.5, color='tab:red' if j == 0 else 'tab:blue')
        vs = [p[3].replace('first better', 'P2 better') for p in pts]
        tab.append((nm, lab[c], len(pts), sum(v == 'equivalent' for v in vs), sum(v == 'P2 better' for v in vs), sum(v not in ('equivalent', 'P2 better') for v in vs), min(p[0] for p in pts), max(p[0] for p in pts)))
    for m in (-0.02, 0.02): ax.axhline(m, color='k', ls=':', lw=0.8)
    ax.axhline(0, color='0.7', lw=0.6); ax.set_xticks(range(5)); ax.set_xticklabels([lab[c] for c in cells], rotation=30, ha='right', fontsize=8); ax.set_title(nm, fontsize=10)
axs[0].set_ylabel('P2 minus P1c, ratio to physics-only'); fig.suptitle('Red: original training seed; blue: replication seeds; dotted: margin $\\pm0.02$', fontsize=8, y=0.99)
fig.tight_layout(); fig.savefig(f'{O}/Figure_robustness.pdf'); fig.savefig(f'{O}/Figure_robustness.png', dpi=200)
with open(f'{O}/tab_robustness.tex', 'w') as f:
    f.write('\\begin{tabular}{llrrrrr}\\toprule\nNetwork & Cell & Fits & Equivalent & P2 better & Other & Range of mean\\\\\\midrule\n')
    for t in tab: f.write(f'{t[0]} & {t[1]} & {t[2]} & {t[3]} & {t[4]} & {t[5]} & [{t[6]:.3f}, {t[7]:.3f}]\\\\\n')
    f.write('\\bottomrule\\end{tabular}\n')
print(open(f'{O}/tab_robustness.tex').read())
