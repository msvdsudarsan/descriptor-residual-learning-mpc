"""Decomposition of the weak-form residual target (plant minus nominal predictor acceleration) into a per-generator constant and a state-varying part, on the TRAINING trajectories only (seed 1001), before and after the P1c recalibration (protocol/v2.1, diagnostic D).
Output: results/baseline_recal/mismatch_decomposition.csv. No test seed is used."""
import sys, numpy as np
from network_recal_baseline import recal_physics, csvw, OUT
from network_validity_gate import *
from network_baselines_loaddyn import stack

rows = []
for name in ('case9', 'case39'):
    for label, scale in (('nominal', 1.0), ('elevated', STRESS[name])):
        c = build_case(name, load_scale=scale); th, de, _ = equilibrium(c)
        train = make_training(c, th, de, 30, 1001, 1e-3); val = make_training(c, th, de, 8, 1002, 1e-3)
        TH, Y = stack([weak_form_net(c, t, th) for t in train]); THv, Yv = stack([weak_form_net(c, t, th) for t in val])
        c1c, info, _ = recal_physics(c, TH, Y, THv, Yv)
        Y1 = stack([weak_form_net(c1c, t, th) for t in train])[1]
        for tag, Z in (('before recalibration', Y), ('after P1c', Y1)):
            ms = float(np.mean(Z ** 2)); const = float(np.mean(Z.mean(0) ** 2)); var = float(np.mean(Z.var(0)))
            rows.append((name, label, tag, round(np.sqrt(ms), 4), round(const / ms, 4), round(var / ms, 4)))
            print(rows[-1], flush=True)
csvw(OUT / 'mismatch_decomposition.csv', ['case', 'loading', 'residual', 'rms', 'constant_share', 'varying_share'], rows)
