"""First-order detectability of a dispatch step (Proposition 5 of the manuscript).

For an unannounced common-scale step of the dispatch the one-step innovation of the corrected predictor is
affine in the dispatch error, e(rho) = e(rho*) + (rho - rho*) * g, with g close to the lowest-order term
    g0 = ( Ts^2/2 (v - v_coi 1) ,  Ts v ),   v = M^{-1} P_m^nom,   v_coi = sum(P_m^nom)/sum(M).
The script compares g0 with the finite-difference slope, the one-step score with kappa_det |rho - rho*|, the dispatch
step at which the plain gate must trip, and the error of one Gauss-Newton step on rho.  Output: results/detectability.csv
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from network_validity_gate import *
from network_trust import innovation_vec

Q0 = dict(case9=0.0238, case39=0.0371)


def one_step_state(c0, th_eq, de_eq, mu):
    c1 = build_case(c0['name'], load_scale=mu); ng = c0['ng']
    de = de_eq.copy(); w = np.zeros(ng); th, _ = solve_theta(th_eq, de, c1['Pd'], *args_of(c1), 1e-12, 30)
    u = np.zeros(ng); hp = 0.0125; y0 = (de.copy(), w.copy()); t = 0.0
    for _ in range(int(round(TS / hp))):
        de, w, th = plant_step(de, w, th, u, hp, t, c1['Pd'], 0.0, 1.0, 0.0, c1['lf'], c1['lt'], c1['lb'], c1['gbus'], c1['bg'], c1['M'], c1['D'], c1['Pm'], c1['kappa'], c1['fric'])
        t += hp
    return c1, y0, (de, w), u


def run():
    rows = []
    for name in ('case9', 'case39'):
        c0, th_eq, de_eq, coef = nominal_fit(name); ng = c0['ng']; Ts = TS
        v = c0['Pm'] / c0['M']; vcoi = c0['Pm'].sum() / c0['M'].sum()
        d = c0['D'] / c0['M']; ph = (1 - np.exp(-d * Ts)) / d; pa = Ts / d - (1 - np.exp(-d * Ts)) / d ** 2
        g0 = np.r_[pa * (v - vcoi), ph * v]; kdet0 = float(np.sqrt(np.mean(g0 ** 2)))
        res = {}
        for mu in (1.01, 1.02, 1.03, 1.05, 1.10, 1.25):
            c1, y0, y1, u = one_step_state(c0, th_eq, de_eq, mu)
            f = lambda rho: innovation_vec(c0, coef, rho * c0['Pm'], y0, u, c1['Pd'], y1)
            e1 = f(1.0); em = f(mu); g = (f(mu + 0.01) - f(mu - 0.01)) / 0.02
            s1 = float(np.sqrt(np.mean(e1 ** 2))); kfd = float(np.sqrt(np.mean(g ** 2)))
            rho_gn = 1.0 - float(g @ e1) / float(g @ g)
            rows.append((name, mu, kdet0, kfd, s1, kdet0 * (mu - 1), s1 / Q0[name], rho_gn, abs(rho_gn - mu) / (mu - 1), float(np.sqrt(np.mean(em ** 2)))))
            print(name, mu, 'kappa0 %.3f kappa_fd %.3f s %.4f lin %.4f q %.2f rho_GN %.4f resid_at_true %.4f' % (kdet0, kfd, s1, kdet0 * (mu - 1), s1 / Q0[name], rho_gn, rows[-1][-1]))
        print(name, 'smallest step with s > q0 (one-step, linear): %.4f' % (Q0[name] / kdet0))
    csv('detectability.csv', ['case', 'mu', 'kappa_lowest_order', 'kappa_fd', 'score_one_step', 'score_lowest_order', 'q_one_step', 'rho_after_one_GN_step', 'GN_error_over_step', 'score_at_true_rho'], rows)


if __name__ == '__main__':
    run()
