"""Tests for the worked case9 example (run: python tests/test_worked_example.py).
They recompute every step and compare with (a) the stored results file and (b) the numbers printed in the manuscript."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import worked_example_case9 as we

# numbers as printed in the manuscript (Section 'Worked example on case9')
PAPER = dict(lam_min_J=2.237, lam_bound=1.949, lam_min_B=2.250, Jinv_norm=0.447, Jinv_norm_bound=0.513, max_line_angle_deg=7.83,
             max_rotor_angle_deg=9.22, wf_target=-2.4266, wf_model=-2.3404, wf_rms_target=1.749, wf_rms_resid=0.091,
             wf_active_terms_gen1=7, pred_A0_1s=0.582, pred_A1_1s=0.084, pred_A2_1s=0.109, pred_ratio_1s=0.19,
             pred_A0_05s=0.446, pred_A2_05s=0.052, g_A0=0.011, g_A1=0.011, g_A2=2.7e-11,
             q_max_before=0.21, tau_min_before=1.0, gate_delay_s=0.1, q_first=3.8, tau_first=0.49, tau_min_after=0.08,
             rho_final=1.253, q_final_reconciled=0.21, tau_final_reconciled=1.0)


def test_worked_example_matches_results_and_paper():
    o = we.compute(verbose=False)
    stored = json.load(open(ROOT / 'results' / 'worked_example_case9.json'))
    for k, v in stored.items():
        assert o[k] == v or (isinstance(v, float) and abs(o[k] - v) <= 1e-9 * max(1.0, abs(v))), (k, o[k], v)
    for k, v in PAPER.items():
        assert abs(o[k] - v) <= 1e-12 + 1e-9 * abs(v), (k, o[k], v)


def test_worked_example_structure():
    o = we.compute(verbose=False)
    assert o['lam_min_J'] >= o['lam_bound']                      # Proposition 2 bound holds at the equilibrium
    assert o['Jinv_norm'] <= o['Jinv_norm_bound']
    assert o['newton_trace'][0] > 1e-6 and o['newton_trace'][1] < 1e-10 and o['newton_trace'][-1] < 1e-14   # quadratic convergence to round-off
    assert o['g_A2'] < 1e-9 < o['g_A0']                          # projection holds the constraint, tangent update does not
    assert o['pred_A2_1s'] < o['pred_A0_1s']                     # correction helps on this window
    assert o['q_max_before'] < 1.0 <= o['q_first'] and o['rho_final'] > 1.2 and abs(o['rho_final'] - 1.25) < 0.01


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
