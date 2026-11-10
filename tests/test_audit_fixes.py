"""Regression tests for two defects found in the audit: (1) the trust weight must scale the clipped residual, d_eff = tau*clip(d_raw);
(2) the error margin must be componentwise (every line angle and rotor angle), not the difference of the largest angles."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from network_validity_gate import *


def test_gate_scales_clipped_residual():
    c = build_case('case9'); ng = c['ng']
    coef = np.zeros((ng, NF + 1)); coef[:, 11] = 10.0; coef[:, NF] = 1.0              # raw residual 10, cap 1
    z = np.zeros(ng); th = np.zeros(c['n'])
    for tau in (1.0, 0.5, 0.2):
        d = res_acc(z, z, z, th, c['gbus'], coef * tau)                               # the code path used by the gate: coef * tau
        assert np.allclose(d, tau * 1.0), (tau, d)
    assert np.allclose(res_acc(z, z, z, th, c['gbus'], coef * 0.0), 0.0)


def test_margin_is_componentwise():
    c, th_eq, de_eq, coef = nominal_fit('case9')
    runs = make_test_runs(c, th_eq, de_eq, 2, 91, 0.06, T=4.0)
    _, _, ang = predict_windows(c, runs, 'A2', coef, N=5)
    assert all(np.all(a >= 0) for a in ang)                                           # componentwise maxima are nonnegative
    # reference: recompute for one window by hand
    load, lg = runs[0]; k = 0; N = 5
    Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2]))
    de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
    for j in range(N):
        for _ in range(NSUB):
            de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, *MODES['A2'][::1], coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
    tt = lg['th'][k + N]
    want = max(np.max(np.abs((th[c['lf']] - th[c['lt']]) - (tt[c['lf']] - tt[c['lt']]))), np.max(np.abs((de - th[c['gbus']]) - (lg['de'][k + N] - tt[c['gbus']]))))
    _, _, ang1 = predict_windows(c, [runs[0]], 'A2', coef, N=N, stride=1000)
    assert abs(float(ang1[N - 1][0]) - float(want)) < 1e-12
