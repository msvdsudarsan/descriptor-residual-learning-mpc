"""Ties the numbers of the application example (manuscript Table 9) to the stored CSV files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from application_example_case39 import numbers


def test_application_numbers():
    o = numbers()
    assert round(o['A0'], 3) == 0.992 and round(o['A2'], 3) == 1.893 and round(o['gate'], 3) == 1.011
    assert round(o['rec'], 3) == 0.479 and round(o['oracle'], 3) == 0.453 and round(o['rho'], 3) == 1.251
    assert round(o['A2'] / o['A0'], 2) == 1.91 and round(o['rec_over_A0'], 2) == 0.48 and round(o['rec_over_A2'], 2) == 0.25
    assert round(o['share_of_oracle_gain'], 2) == 0.98 and abs(o['delay'] - 0.10) < 1e-9
    # timings depend on the machine: check only their order and the budget (0.1 s interval), not the quoted milliseconds
    assert o['ms_exact'] < o['ms_A0'] < o['ms_fd'] and o['ms_exact'] < 100.0 and 0 < o['budget_share'] < 1


def test_scan_authority_and_detectability():
    import pandas as pd
    R = Path(__file__).resolve().parents[1] / 'results'
    s = pd.read_csv(R / 'case39_application_scan_merged.csv').set_index('mu_after')
    assert (s.detection_rate == 1).all()
    d = (s.rmse_A0 - s.rmse_A2).values; k = [i for i in range(len(d) - 1) if d[i] > 0 >= d[i + 1]][0]
    xs = 100 * (s.index.values - 1); xc = xs[k] + (xs[k + 1] - xs[k]) * d[k] / (d[k] - d[k + 1])
    assert abs(xc - 7.7) < 0.1                                    # A2 with stale dispatch loses to A0 at about 8 percent
    assert ((s.rmse_reconcile / s.rmse_oracle).loc[:1.2501] < 1.15 + 1e-9).all()
    a = pd.read_csv(R / 'case39_application_authority.csv').set_index(['authority_scale', 'controller'])
    assert abs(a.loc[(3.0, 'A0'), 'w_rms'] - 0.346) < 5e-4 and abs(a.loc[(3.0, 'A2'), 'w_rms'] - 0.138) < 5e-4 and abs(a.loc[(3.0, 'rec'), 'w_rms'] - 0.037) < 5e-4
    assert a.loc[(1.0, 'A2'), 'input_at_bound_share'] == 1.0
    q = pd.read_csv(R / 'detectability.csv').set_index(['case', 'mu'])
    assert abs(q.loc[('case39', 1.25), 'rho_after_one_GN_step'] - 1.25) < 0.002 and abs(q.loc[('case9', 1.25), 'rho_after_one_GN_step'] - 1.25) < 0.002
    k0 = q.loc['case39'].kappa_lowest_order.iloc[0]; kf = q.loc['case39'].kappa_fd.iloc[0]
    assert abs(k0 / kf - 1) < 0.05 and abs(0.0371 / k0 - 0.0148) < 5e-4


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
