"""Numbers for the application example of the manuscript (case39, unannounced dispatch step of 25 %).
Reads results/case39_reconcile_prediction.csv and results/case39_analytic_closed_loop.csv; nothing is re-simulated."""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'


def numbers():
    d = pd.read_csv(R / 'case39_reconcile_prediction.csv').set_index('scenario').loc['persistent mu 1.25']
    c = pd.read_csv(R / 'case39_analytic_closed_loop.csv').set_index('controller')
    o = dict(A0=float(d.rmse_A0), A2=float(d.rmse_A2), oracle=float(d.rmse_oracle), gate=float(d.rmse_gate), rec=float(d.rmse_reconcile),
             rho=float(d.mean_rho_estimate), delay=float(d.detection_delay_s), retrust=float(d.retrust_delay_s))
    o['rec_over_A0'] = o['rec'] / o['A0']; o['rec_over_A2'] = o['rec'] / o['A2']
    o['share_of_oracle_gain'] = (o['A2'] - o['rec']) / (o['A2'] - o['oracle'])
    o['ms_exact'] = float(c.loc['A2', 'solve_ms_mean']); o['ms_fd'] = float(c.loc['A2fd', 'solve_ms_mean']); o['ms_A0'] = float(c.loc['A0', 'solve_ms_mean'])
    o['budget_share'] = o['ms_exact'] / 100.0
    return o


if __name__ == '__main__':
    out = numbers()
    for k, v in out.items(): print(f'{k:20s} {v:.4f}')
    with open(R / 'case39_application_example.csv', 'w') as f:
        f.write('quantity,value\n'); [f.write(f'{k},{v:.6g}\n') for k, v in out.items()]
