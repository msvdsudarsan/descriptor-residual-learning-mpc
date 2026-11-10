"""Figure of the application study (reads results/case39_application_scan.csv and results/case39_application_authority.csv)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RES, FIG = ROOT / 'results', ROOT / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
cols = ['mu_after', 'rmse_A0', 'rmse_A2', 'rmse_oracle', 'rmse_gate', 'rmse_reconcile', 'mean_rho_estimate', 'detection_rate', 'detection_delay_s', 'min_trust']
b = pd.read_csv(RES / 'case39_reconcile_prediction.csv'); b = b[b.scenario.str.startswith('persistent') & (b.mu_after > 1)]
s = pd.concat([pd.read_csv(RES / 'case39_application_scan.csv')[cols], b[cols]]).sort_values('mu_after').reset_index(drop=True)
s.to_csv(RES / 'case39_application_scan_merged.csv', index=False); s = s[s.mu_after <= 1.2501].reset_index(drop=True); x = 100 * (s.mu_after - 1).values
a = ax[0]
a.plot(x, s.rmse_A0.values, 's-', color='0.45', label='A0, physics only'); a.plot(x, s.rmse_A2, 'o-', color='#d62728', label='A2, stale dispatch')
a.plot(x, s.rmse_reconcile, '^-', color='#1f77b4', label='gate + reconciliation'); a.plot(x, s.rmse_oracle, 'v:', color='#2ca02c', label='A2, true dispatch')
a.axvline(1.5, color='k', lw=.8, ls='--'); a.text(1.7, 2.3, 'smallest step\nthe gate sees', fontsize=6.5, va='top')
d = (s.rmse_A0 - s.rmse_A2).values; k = int(np.where(np.diff(np.sign(d)))[0][0]); xc = x[k] + (x[k + 1] - x[k]) * d[k] / (d[k] - d[k + 1])
a.axvline(xc, color='#d62728', lw=.8, ls='--'); a.text(xc + .3, 2.3, 'A2 loses to A0', fontsize=6.5, va='top', color='#d62728')
a.set_xlabel('unannounced load and dispatch step [%]'); a.set_ylabel('1 s state RMSE after the step'); a.set_title('(a) prediction, case39'); a.set_ylim(0, 2.35); a.set_xlim(0, 26); a.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(0.30, 0.90))
p = RES / 'case39_application_authority.csv'
b = ax[1]
if p.exists():
    c = pd.read_csv(p); names = [('C0', 'static law', '0.45', 's'), ('A0', 'A0', '#ff7f0e', 'D'), ('A2', 'A2, stale dispatch', '#d62728', 'o'), ('rec', 'gate + reconciliation', '#1f77b4', '^'), ('oracle', 'A2, true dispatch', '#2ca02c', 'v')]
    for key, lab, col, mk in names:
        q = c[c.controller == key]; b.plot(q.authority_scale, q.w_rms, mk + '-', color=col, label=lab)
    b.set_xscale('log', base=2); b.set_xticks(sorted(c.authority_scale.unique())); b.set_xticklabels([f'{v:g}' for v in sorted(c.authority_scale.unique())])
    b.set_xlabel('actuator authority (multiple of the nominal input bound)'); b.set_ylabel('RMS speed deviation [rad/s]'); b.set_title('(b) closed loop, 5% step'); b.set_ylim(0, 1.2); b.legend(fontsize=6.5, ncol=2, loc='upper right')
fig.tight_layout(); fig.savefig(FIG / 'Figure_application.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Figure_application.png', dpi=300, bbox_inches='tight'); print('crossing at %.2f percent' % xc)
