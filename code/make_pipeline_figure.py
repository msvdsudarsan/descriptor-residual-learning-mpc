"""Draws the conceptual pipeline figure: Fig. 1 of the manuscript and the graphical abstract.

Pure matplotlib drawing in inch coordinates; it shows no data and no result.
Outputs (figures/): fig_pipeline.pdf (manuscript, 5.4 in wide), graphical_abstract.png (2656 x 1062 px).
Usage:  python code/make_pipeline_figure.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})

VARIANTS = [('C0', 'static', 'feedback'), ('A0', 'mismatched', 'physics'), ('A1', '+ missing', 'physics'),
            ('A2', '+ manifold', 'projection'), ('A3', '+ empirical', 'margin'), ('G', '+ trust gate,', 'reconciliation')]
LAYERS = [(0, 0, 'I  physics'), (1, 1, 'II  learning'), (2, 2, 'III  structure'), (3, 4, 'IV  trust, adaptation'), (5, 5, 'V  control')]


def build(W, H, fs, top, title, bnd, name_pdf=None, name_png=None, dpi=300, s=None):
    fig = plt.figure(figsize=(W, H), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    m, gap = 0.08 * W / 5.4, 0.12 * W / 5.4
    w = (W - 2 * m - 5 * gap) / 6
    lh = fs * 1.55 / 72                                         # line height in inches
    s = H / 3.7 if s is None else s

    def box(x, y, h, lines, fc='#ffffff'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.04', fc=fc, ec='#333333', lw=0.8))
        n = len(lines); y0 = y + h / 2 + (n - 1) * lh / 2
        for i, t in enumerate(lines):
            ax.text(x + w / 2, y0 - i * lh, t, ha='center', va='center', fontsize=fs, fontweight='bold' if i == 0 else 'normal')

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=6, lw=0.8, color='#333333'))

    y_title = H - 0.14 * s + (0.0 if title else 0.08)
    if title:
        ax.text(m, y_title, title, fontsize=fs + 0.8, fontweight='bold', ha='left', va='center')
    h_top = (max(len(t) for t in top) + 1.1) * lh
    y_top = y_title - 0.38 * s - h_top
    for k, lines in enumerate(top):
        x = m + k * (w + gap)
        box(x, y_top, h_top, lines, fc='#ececec' if k == 0 else '#ffffff')
        if k:
            arrow(x - gap + 0.01, x - 0.01, y_top + h_top / 2)
    for k0, k1, lab in LAYERS:
        x0 = m + k0 * (w + gap); x1 = m + k1 * (w + gap) + w; yl = y_top + h_top + 0.07 * s
        ax.plot([x0, x1], [yl, yl], color='#777777', lw=0.7)
        ax.text((x0 + x1) / 2, yl + 0.06 * s, lab, fontsize=fs - 0.4, style='italic', ha='center', va='bottom', color='#444444')
    y_lab = y_top - 0.17 * s
    ax.text(m, y_lab, 'Matched variants (each step isolates one ingredient)', fontsize=fs, style='italic', ha='left', va='center')
    h_var = 4.1 * lh
    y_var = y_lab - 0.14 * s - h_var
    for k, (a, b, c) in enumerate(VARIANTS):
        x = m + k * (w + gap)
        box(x, y_var, h_var, [a, b, c])
        if k:
            arrow(x - gap + 0.01, x - 0.01, y_var + h_var / 2)
    h_b = (len(bnd) + 1.2) * lh
    y_b = max(0.06, y_var - 0.16 * s - h_b)
    ax.add_patch(FancyBboxPatch((m, y_b), W - 2 * m, h_b, boxstyle='round,pad=0,rounding_size=0.04', fc='#ffffff', ec='#333333', lw=0.8, ls=(0, (4, 2))))
    ys = y_b + h_b / 2 + (len(bnd) - 1) * lh / 2
    for i, t in enumerate(bnd):
        ax.text(W / 2, ys - i * lh, t, ha='center', va='center', fontsize=fs, fontweight='bold' if i == 0 else 'normal')
    if name_pdf:
        fig.savefig(OUT / name_pdf)
    if name_png:
        fig.savefig(OUT / name_png, dpi=dpi)
    plt.close(fig)


# manuscript figure, 5.4 in wide
top_ms = [
    ['Known', 'physics', r'$\dot x=f+d$', r'$0=g(x,z,u)$'],
    ['Learned', 'residual', 'weak-form', 'ridge fit'],
    ['Manifold', 'projection', 'Newton:', r'$g=0$'],
    ['Trust', 'gate', 'innovation', r'$\tau\in(0,1]$'],
    ['Dispatch', 'reconcile', r'estimate $\rho$', 'restore trust'],
    ['Receding', 'horizon', 'exact gradient', 'margin'],
]
bnd_ms = ['Reported validity boundary',
          'benefit depends on the dispatch error more than on the loading;',
          'a trust gate falls back to physics after an unannounced shift, a',
          'reconciliation step restores the correction; tightening inactive on the networks']
build(5.4, 2.62, 5.4, top_ms, '', bnd_ms, name_pdf='fig_pipeline.pdf', s=0.72)

# graphical abstract: 2656 x 1062 px at 300 dpi
top_ga = [
    ['Known physics', r'$\dot x=f+d,\ \ 0=g(x,z,u)$', r'$f,\,g$ known'],
    ['Learned residual', 'weak-form ridge fit', 'noisy trajectories'],
    ['Manifold projection', r'Newton solve of $g=0$', 'exact gradient'],
    ['Trust gate', 'innovation score', r'$\tau\hat d$, physics fallback'],
    ['Reconciliation', r'estimate dispatch $\rho$', 'restore the trust'],
    ['Receding horizon', 'split-conformal margin', 'control'],
]
bnd_ga = ['Reported validity boundary',
          'the benefit depends on the dispatch error more than on the loading; a trust gate falls back to physics after an unannounced shift,',
          'a reconciliation step re-estimates the dispatch and restores the correction; tightening inactive on the networks; reduced lossless models']
build(8.854, 3.54, 6.0, top_ga, 'Structure-preserving residual learning and predictive control of a nonlinear descriptor model', bnd_ga,
      name_png='graphical_abstract.png')
print('figures written to', OUT)
