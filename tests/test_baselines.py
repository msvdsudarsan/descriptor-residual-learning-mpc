"""Ties the statements of manuscript Section 'Other residual models and an unmodelled load lag' to the stored CSV files."""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'
W, S, D, N, P = 'OURS (sparse; weak form)', 'SF (sparse; strong form)', 'RIDGE (dense; weak form)', 'MLP (2x32; weak form)', 'P0 (physics; projected)'


def load(nm):
    return pd.read_csv(R / f'{nm}_baselines_loaddyn.csv'), pd.read_csv(R / f'{nm}_baselines_noise.csv')


def r(d, sc, m):
    return float(d[(d.scenario == sc) & (d.method == m)].ratio_to_A0.iloc[0])


def test_statements():
    for nm, lo, hi in (('case9', 0.13, 0.145), ('case39', 0.175, 0.20)):
        d, e = load(nm)
        assert set(d.scenario) == {'in-family', 'load-lag tau=0.5', 'load-lag tau=1.0', 'saliency x1.5'}
        for sc in set(d.scenario):
            assert 1.00 <= r(d, sc, P) <= 1.035                                    # projection alone does not reduce the error
            assert r(d, sc, S) < min(r(d, sc, W), r(d, sc, D), r(d, sc, N))         # strong form lowest in every row
            assert max(r(d, sc, m) for m in (W, S, D, N)) < 0.36
        for m in (W, S, D, N):
            assert lo <= r(d, 'in-family', m) <= hi
        assert max(abs(r(d, 'in-family', W) - r(d, 'in-family', m)) for m in (D, N)) <= 0.003
        p = d[d.method == P].set_index('scenario').rmse_1s
        assert all(abs(p[k] - p['in-family']) / p['in-family'] < 0.02 for k in ('load-lag tau=0.5', 'load-lag tau=1.0'))
        w = e[e.method == W].set_index('noise_sigma'); s = e[e.method == S].set_index('noise_sigma')
        gap = w.ratio_to_A0 - s.ratio_to_A0
        assert 0 < float(gap.min()) and float(gap.max()) <= 0.03
        assert int(w.loc[1e-3, 'active_terms']) in (24, 118) and int(w.loc[0.03, 'active_terms']) in (9, 45)


if __name__ == '__main__':
    test_statements(); print('ok test_statements')
