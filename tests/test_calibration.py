"""Tests of the calibration and baseline checks (stored results): coverage equals the guaranteed level over independent sets (within the interval),
the spread over sets follows the Beta law, the shifted pool is not covered, recalibration does not change the closed loop, a 2-3 coefficient correction matches the library residual."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'


def test_independent_design():
    for nm in ('case9', 'case39'):
        d = pd.read_csv(R / f'{nm}_conformal_independent.csv')
        for n in d.n_cal.unique():
            a = d[(d.n_cal == n) & (d.quantity == 'marginal step 3')].iloc[0]; j = d[(d.n_cal == n) & d.quantity.str.contains('scaled')].iloc[0]
            assert abs(a.mean_coverage - a.guaranteed_level) <= a.ci95_half_width + 0.004, (nm, n)
            assert abs(j.mean_coverage - j.guaranteed_level) <= j.ci95_half_width + 0.004, (nm, n)
            assert a.shifted_coverage < 0.3
            if n >= 20: assert abs(a.sd_over_sets / a.sd_beta_theory - 1) < 0.2
        assert abs(float(d[(d.n_cal == 10) & (d.quantity == 'marginal step 3')].guaranteed_level.iloc[0]) - 10 / 11) < 1e-3


def test_recalibration_and_baseline():
    for nm in ('case9', 'case39'):
        c = pd.read_csv(R / f'{nm}_cl_recal_coverage.csv'); j = c[c.quantity.str.contains('scaled')].iloc[0]
        assert 0.90 <= j.coverage <= 1.0 and float(c.iloc[0].coverage) == 1.0                   # original envelope is conservative on the policy-matched test pool
        p = pd.read_csv(R / f'{nm}_cl_recal_performance.csv'); g = p[p.scenario == 'limit 30 deg'].groupby('controller').w_rms.mean()
        assert abs(g['A3'] - g['A2']) < 1e-3 * g['A2']                                          # recalibrated margin does not change the closed loop
        q = pd.read_csv(R / f'{nm}_param_baseline_prediction.csv').query("scenario=='in-family'").set_index('correction').rmse_ratio_to_A0
        w = pd.read_csv(R / f'{nm}_param_baseline_closed_loop.csv').groupby('controller').w_rms.mean()
        assert abs(q['damping + constant'] - q['library residual (paper)']) < 0.02
        assert abs(w['damping + constant'] - w['library residual (paper)']) < 0.002
