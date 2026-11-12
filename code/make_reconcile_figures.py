"""Figures for the atlas with the gate boundary and for the recovery test (from results/*.csv and *.npz)."""
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from network_experiments import RES, FIG, COL, save, plt


def atlas_boundary_figure():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, nm in zip(axs, ('case9', 'case39')):
        at = pd.read_csv(RES / f'{nm}_atlas_gate.csv'); mus = sorted(at.load_scale.unique()); kaps = sorted(at.dispatch_error.unique())
        Z = np.full((len(kaps), len(mus)), np.nan); F = np.zeros_like(Z)
        for r in at.itertuples(): Z[kaps.index(r.dispatch_error), mus.index(r.load_scale)] = r.ratio_A2_over_A0; F[kaps.index(r.dispatch_error), mus.index(r.load_scale)] = r.gate_fallback_share
        im = ax.imshow(np.log2(Z), origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
        for i in range(len(kaps)):
            for j in range(len(mus)):
                ax.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', fontsize=5.5, color='k')
                if F[i, j] > 0.5: ax.add_patch(Polygon([[j - .47, i + .47], [j - .47, i + .2], [j - .2, i + .47]], closed=True, color='k'))
        eta = at.groupby('load_scale').eta_J.first()
        ax.set_xticks(range(len(mus))); ax.set_xticklabels([f'{m:g}\n({eta[m]:.2f})' for m in mus], fontsize=6)
        ax.set_yticks(range(len(kaps))); ax.set_yticklabels([f'{k:+.1f}' for k in kaps], fontsize=6)
        ax.set_xlabel('load scale $\\mu$  (and $\\eta_J$)'); ax.set_ylabel('dispatch error $\\kappa$'); ax.set_title(f'({"ab"[("case9", "case39").index(nm)]}) {nm}'); ax.grid(False)
    fig.subplots_adjust(bottom=0.2, left=0.08, right=0.89, wspace=0.12)
    cax = fig.add_axes([0.915, 0.2, 0.014, 0.68]); cb = fig.colorbar(im, cax=cax); cb.set_label('$\\log_2$ (A2/A0 error ratio)', fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, 'fig_validity_atlas')


def recovery_figure():
    fig, axs = plt.subplots(3, 2, figsize=(7.6, 5.4), sharex='col')
    for k, nm in enumerate(('case9', 'case39')):
        d = np.load(RES / f'{nm}_recovery_trace.npz'); t = d['t']; ts, tr = float(d['t_shift']), float(d['t_rest']); mu1 = float(d['mu1'])
        ax = axs[0, k]; ax.step([0, ts, tr, t[-1]], [1, mu1, 1, 1], where='post', color='k', lw=1.0, label='true dispatch scale'); ax.plot(t, d['rho'], color=COL['A3'], lw=1.1, label='estimate $\\hat\\rho$')
        ax.set_ylabel('dispatch scale'); ax.set_title(f'({"ab"[k]}) {nm}: step to $\\mu={mu1:g}$ at {ts:g} s, undone at {tr:g} s'); ax.legend(fontsize=6, loc='center right')
        ax = axs[1, k]; ax.plot(t, d['tau_g'], color=COL['A1'], lw=1.1, label='gate'); ax.plot(t, d['tau_r'], color=COL['A3'], lw=1.1, ls='--', label='gate with reconciliation'); ax.set_ylabel('trust $\\tau$'); ax.legend(fontsize=6, loc='center left')
        ax = axs[2, k]; W = d['W']
        for col, lab, c_, ls in ((1, 'A0', COL['A0'], '-'), (2, 'A2', COL['A2'], '-'), (3, 'A2, true dispatch', 'k', ':'), (4, 'gate', COL['A1'], '-'), (5, 'gate + reconciliation', COL['A3'], '--')):
            ax.semilogy(W[:, 0], np.sqrt(W[:, col]), color=c_, lw=1.0, ls=ls, label=lab)
        ax.set_ylabel('1 s prediction RMSE'); ax.set_xlabel('window start [s]')
        ax.legend(fontsize=5.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.32), frameon=False)
        for a in axs[:, k]: a.axvspan(ts, tr, color='k', alpha=.05)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.17); save(fig, 'fig_recovery')


if __name__ == '__main__':
    atlas_boundary_figure(); recovery_figure(); print('figures written')
