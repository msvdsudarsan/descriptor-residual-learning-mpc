"""N10: topology error.  One line is out of service in the PLANT (its susceptance is set to zero) while the predictor, the
physics model and the learned residual keep the NOMINAL network and the nominal dispatch.  For every line whose outage leaves a
regular, connected network with an equilibrium, the one-second prediction error of A2 (residual + projection) is compared with
A0 (physics only).  Nothing is re-fitted.  Reported per line: A2/A0 RMSE ratio and the number of samples."""
import sys
import numpy as np
from network_experiments import *


def run(name, max_lines=None):
    c0 = build_case(name); th_eq, de_eq, res = equilibrium(c0)
    train = make_training(c0, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c0, th_eq, de_eq, 8, 1002, 1e-3)
    coef, _, _ = fit_net(c0, train, val, th_eq)
    nl = len(c0['lf']); lines = list(range(nl)) if max_lines is None else list(np.linspace(0, nl - 1, max_lines).astype(int))
    rows = []
    for l in lines:
        cp = build_case(name); cp['lb'] = cp['lb'].copy(); cp['lb'][l] = 0.0
        try:
            thp, dep, rp = equilibrium(cp)
        except Exception as ex:
            rows.append((l, int(c0['lf'][l]), int(c0['lt'][l]), 'no_equilibrium', float('nan'), float('nan'), float('nan'))); print(name, rows[-1], flush=True); continue
        if not (rp < 1e-8) or not np.all(np.isfinite(thp)):
            rows.append((l, int(c0['lf'][l]), int(c0['lt'][l]), 'no_equilibrium', float('nan'), float('nan'), float('nan'))); print(name, rows[-1], flush=True); continue
        runs = make_test_runs(cp, thp, dep, 4, 5000 + l, 0.06)
        e0 = predict_windows(c0, runs, 'A0', coef)[0]; e2 = predict_windows(c0, runs, 'A2', coef)[0]
        r0 = float(np.sqrt(np.mean(np.vstack(e0) ** 2))); r2 = float(np.sqrt(np.mean(np.vstack(e2) ** 2)))
        rows.append((l, int(c0['lf'][l]), int(c0['lt'][l]), 'ok', r0, r2, r2 / r0)); print(name, rows[-1], flush=True)
    csv(f'{name}_topology.csv', ['line_index', 'from_bus_idx', 'to_bus_idx', 'status', 'rmse_A0', 'rmse_A2', 'ratio_A2_over_A0'], rows)


if __name__ == '__main__':
    nm = sys.argv[1]; mx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(nm, mx)
