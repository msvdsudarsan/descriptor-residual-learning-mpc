"""N7 out-of-library plant mismatch and N8 library-complexity sweep for case9 / case39."""
import sys, json
import numpy as np
from network_experiments import *

OOL = 0.15      # coefficient of the plant term  ool * b_i * psi*|psi|  (not representable by the feature library)


def ool_test(name):
    c = build_case(name, ool=OOL); th_eq, de_eq, _ = equilibrium(c)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3); test = make_training(c, th_eq, de_eq, 16, 1003, 0.0)
    coef, _, _ = fit_net(c, train, val, th_eq)
    rms_ex, rms_err, max_err = exact_resid_stats(c, test, coef)
    runs = make_test_runs(c, th_eq, de_eq, 10, 2001, 0.08)
    ratios = []
    for r in runs:
        e0 = predict_windows(c, [r], 'A0', coef)[0]; e2 = predict_windows(c, [r], 'A2', coef)[0]
        ratios.append(np.sqrt(np.mean(np.vstack(e2) ** 2) / np.mean(np.vstack(e0) ** 2)))
    m, lo, hi = bootstrap_ci(np.array(ratios))
    g = predict_windows(c, runs, 'A2', coef)[1]
    csv(f'{name}_out_of_library.csv', ['quantity', 'value'], [
        ('ool_coefficient', OOL), ('residual_rms_true_test', rms_ex), ('residual_rms_error_after_fit', rms_err),
        ('error_reduction_percent', 100 * (1 - rms_err / rms_ex)), ('pred_ratio_A2_over_A0', m), ('pred_ratio_lo', lo), ('pred_ratio_hi', hi),
        ('active_terms', int(np.sum(coef[:, :NF] != 0))), ('mean_window_max_abs_g_A2', float(np.mean(g)))])
    print(name, 'OOL', rms_ex, rms_err, (m, lo, hi), flush=True)


def complexity(name):
    c = build_case(name); th_eq, de_eq, _ = equilibrium(c)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); test = make_training(c, th_eq, de_eq, 16, 1003, 0.0)
    Tr = [weak_form_net(c, t, th_eq) for t in train]
    TH = np.concatenate([a for a, _ in Tr]); Y = np.concatenate([b for _, b in Tr])
    rows = []
    for thr in (0.002, 0.01, 0.03, 0.1, 0.3, 1.0):
        coef = np.zeros((c['ng'], NF + 1))
        for i in range(c['ng']):
            coef[i, :NF] = stlsq(TH[:, i, :], Y[:, i], ridge=1e-4, thresh=thr); coef[i, NF] = 1.5 * np.max(np.abs(Y[:, i]))
        rms_ex, rms_err, _ = exact_resid_stats(c, test, coef)
        rows.append((thr, int(np.sum(coef[:, :NF] != 0)), c['ng'] * NF, rms_err, 100 * (1 - rms_err / rms_ex)))
    csv(f'{name}_complexity.csv', ['threshold', 'active_terms', 'terms_available', 'heldout_residual_rmse', 'error_reduction_percent'], rows)
    print(name, 'complexity', rows, flush=True)


if __name__ == '__main__':
    for nm in ('case9', 'case39'):
        ool_test(nm); complexity(nm)
