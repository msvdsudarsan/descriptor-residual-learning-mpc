"""Paired bootstrap intervals for the closed-loop speed-regulation changes (results/*_closed_loop_runs.csv).
The intervals describe variability over the simulated load and initial conditions, not uncertainty about a physical system."""
import pandas as pd, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'; rng = np.random.default_rng(1); out = []
for nm in ('case9', 'case39'):
    p = pd.read_csv(R / f'{nm}_closed_loop_runs.csv').pivot(index='run', columns='controller', values='w_rms'); n = len(p)
    for c in ('A0', 'A2'):
        r = (p[c] / p['C0']).values; bs = [100 * (1 - np.mean(r[rng.integers(0, n, n)])) for _ in range(10000)]
        out.append((nm, c, n, 100 * (1 - r.mean()), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((r < 1).sum())))
        print(f'{nm} {c} vs C0: mean change {out[-1][3]:.1f}% [{out[-1][4]:.1f}, {out[-1][5]:.1f}], better in {out[-1][6]}/{n} runs')
pd.DataFrame(out, columns=['case', 'controller', 'runs', 'mean_reduction_pct', 'ci_low', 'ci_high', 'runs_better']).to_csv(R / 'closed_loop_intervals.csv', index=False)
