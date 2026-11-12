"""Checks the numbers quoted in the manuscript text about calibration and the parametric baseline against results/*.csv.
paper/values_calibration.json holds the strings as printed in the manuscript; the script recomputes them from the CSV files."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'; V = json.load(open(ROOT / 'paper' / 'values_calibration.json'))
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'; f4 = lambda x: f'{x:.4f}'


def ind(nm, n, q): return pd.read_csv(R / f'{nm}_conformal_independent.csv').query('n_cal==@n').pipe(lambda d: d[d.quantity.str.contains(q, regex=False)]).iloc[0]


calc = {}
calc['ind_c9_n20_marg'] = f3(ind('case9', 20, 'marginal step 3').mean_coverage); calc['ind_c9_n20_ci'] = f3(ind('case9', 20, 'marginal step 3').ci95_half_width)
calc['ind_c9_n20_level'] = f3(ind('case9', 20, 'marginal step 3').guaranteed_level)
calc['ind_n10_c9'] = f3(ind('case9', 10, 'marginal step 3').mean_coverage); calc['ind_n10_c39'] = f3(ind('case39', 10, 'marginal step 3').mean_coverage)
calc['ind_sd_obs'] = f3(ind('case9', 20, 'marginal step 3').sd_over_sets); calc['ind_sd_th'] = f3(ind('case9', 20, 'marginal step 3').sd_beta_theory)
sj = [ind(nm, n, 'scaled').mean_coverage for nm in ('case9', 'case39') for n in (10, 20, 40)]; pj = [ind(nm, n, 'not guaranteed').mean_coverage for nm in ('case9', 'case39') for n in (10, 20, 40)]
sh = [ind(nm, n, 'marginal step 3').shifted_coverage for nm in ('case9', 'case39') for n in (10, 20, 40)]
calc['ind_scaled_lo'] = f3(min(sj)); calc['ind_scaled_hi'] = f3(max(sj)); calc['ind_perstep_lo'] = f2(min(pj)); calc['ind_perstep_hi'] = f2(max(pj)); calc['ind_shift_lo'] = f2(min(sh)); calc['ind_shift_hi'] = f2(max(sh))
for nm, t in (('case9', 'c9'), ('case39', 'c39')):
    d = pd.read_csv(R / f'{nm}_cl_recal_coverage.csv'); mg = d[d.quantity.str.startswith('marginal step') & d.calibration.str.startswith('policy')]
    calc[f'rec_{t}_marg_lo'] = f2(mg.coverage.min()); calc[f'rec_{t}_marg_hi'] = f2(mg.coverage.max())
    js = d[d.quantity.str.contains('scaled')].iloc[0]; calc[f'rec_{t}_joint_scaled'] = f3(js.coverage); calc[f'rec_{t}_joint_scaled_ci'] = f3(js.ci95_half_width)
    calc[f'rec_{t}_joint_perstep'] = f2(d[d.quantity.str.contains('not guaranteed')].iloc[0].coverage); calc[f'rec_{t}_prod_step3'] = f2(d[d.quantity.str.contains('marginal step 3 on the test')].iloc[0].coverage)
    old = pd.read_csv(R / f'{nm}_tube_envelope_stress.csv').eps_line_angle_rad.values[5]; new = pd.read_csv(R / f'{nm}_tube_envelope_stress_recal.csv').eps_line_angle_rad.values[5]
    calc[f'rec_{t}_eps5_old'] = f'{np.degrees(old):.1f}'; calc[f'rec_{t}_eps5_new'] = f'{np.degrees(new):.1f}'
    p = pd.read_csv(R / f'{nm}_cl_recal_performance.csv'); g = p[p.scenario == 'limit 30 deg'].groupby('controller').mean(numeric_only=True)
    calc[f'rec_{t}_wrms_A2'] = f4(g.loc['A2', 'w_rms']); calc[f'rec_{t}_wrms_A3'] = f4(g.loc['A3', 'w_rms'])
    q = pd.read_csv(R / f'{nm}_param_baseline_prediction.csv').query("scenario=='in-family'").set_index('correction').rmse_ratio_to_A0; c = pd.read_csv(R / f'{nm}_param_baseline_closed_loop.csv').groupby('controller').w_rms.mean()
    calc[f'par_{t}_pred_lib'] = f3(q['library residual (paper)']); calc[f'par_{t}_pred_dc'] = f3(q['damping + constant']); calc[f'par_{t}_wrms_lib'] = f4(c['library residual (paper)']); calc[f'par_{t}_wrms_dc'] = f4(c['damping + constant'])
p9 = pd.read_csv(R / 'case9_cl_recal_performance.csv'); g23 = p9[p9.scenario == 'limit 23 deg'].groupby('controller').mean(numeric_only=True)
calc['rec_c9_23_frac_A2'] = f'{100 * g23.loc["A2", "frac_time_over_limit"]:.2f}'; calc['rec_c9_23_frac_A3'] = f'{100 * g23.loc["A3", "frac_time_over_limit"]:.2f}'
calc['rec_c9_23_cov'] = f2(pd.read_csv(R / 'case9_cl_recal_coverage_limit23.csv').iloc[0].coverage); calc['rec_c9_23_joint_perstep'] = f2(pd.read_csv(R / 'case9_cl_recal_coverage_limit23.csv').iloc[1].coverage)
for nm, t in (('case9', 'c9'), ('case39', 'c39')):
    d = pd.read_csv(R / f'{nm}_cl_recal_coverage.csv'); calc[f'rec_{t}_prod_joint'] = f2(d[d.calibration.str.startswith('production envelope (n=40)')].iloc[0].coverage)
bad = [(k, V[k], calc.get(k)) for k in V if V[k] != calc.get(k)]
print(f'{len(V)} calibration values checked, {len(bad)} differences')
for b in bad: print('  ', b)
sys.exit(1 if bad else 0)
