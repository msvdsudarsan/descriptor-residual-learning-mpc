"""Analytic-gradient study for the projected network controllers (A2, A3).

1. Gradient check: exact gradient against central differences, and time per cost-and-gradient evaluation.
2. Closed loop at the elevated loading of Table 7, same realizations and calibrated envelope as in network_experiments.py,
   with the analytic gradient in A2 and A3 (A0 keeps finite differences; A2 with finite differences is re-timed in the same session).
"""
import time
import numpy as np
from network_experiments import *

RES_STRESS_ENV = lambda name: np.loadtxt(RES / f'{name}_tube_envelope_stress.csv', delimiter=',', skiprows=1)[:, 1]


def gradient_check(name):
    c = build_case(name, load_scale=STRESS[name]); th, de, _ = equilibrium(c); ng = c['ng']
    rng = np.random.default_rng(3); coef = np.zeros((ng, NF + 1)); coef[:, :NF] = rng.normal(0, 0.05, (ng, NF)); coef[:, NF] = 1.0
    rel_ref = de - np.sum(c['M'] * de) / np.sum(c['M']); N = 5
    errs = []; ta = []; tf = []
    for rep in range(8):
        de0 = de + rng.normal(0, 0.1, ng); w0 = rng.normal(0, 0.3, ng); th0, _ = solve_theta(th, de0, c['Pd'], *args_of(c), 1e-12, 30)
        U = rng.normal(0, 0.5, N * ng) * np.tile(c['umax'], N); eps = np.array([0, .015, .03, .04, .055, .06])
        a = (de0, w0, th0, np.zeros(ng), N, TS, NSUB, 1, 1, coef, c['Pd'], c['lf'], c['lt'], c['lb'], c['gbus'], c['bg'], c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'],
             rel_ref, c['umax'], 0.5, 20.0, 0.02, 0.2, np.radians(20.0 if rep % 2 else 30.0), eps, 3e3)
        J1, g1 = mpc_cost_net_analytic(U, *a); gfd = np.zeros_like(U)
        for j in range(len(U)):
            e = np.zeros_like(U); e[j] = 1e-6; gfd[j] = (mpc_cost_net(U + e, *a) - mpc_cost_net(U - e, *a)) / 2e-6
        errs.append(np.linalg.norm(g1 - gfd) / np.linalg.norm(gfd))
        t0 = time.perf_counter(); mpc_cost_net_analytic(U, *a); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); mpc_cost_net_grad(U, *a); tf.append(time.perf_counter() - t0)
    return float(np.max(errs)), 1e3 * float(np.median(ta)), 1e3 * float(np.median(tf))


def stress_setup(name):
    c = build_case(name, load_scale=STRESS[name]); th_eq, de_eq, _ = equilibrium(c)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    coef, _, _ = fit_net(c, train, val, th_eq)
    return c, th_eq, de_eq, coef


def closed_loop(name):
    cs, th_eq, de_eq, coef = stress_setup(name)
    eps_s = RES_STRESS_ENV(name)
    rel_ref = de_eq - np.sum(cs['M'] * de_eq) / np.sum(cs['M'])
    nreal = 6 if name == 'case9' else 4; Tcl = 6.0 if name == 'case9' else 5.0
    # warm-up: compile every code path once so that the timings contain no JIT time
    th0w, _ = solve_theta(th_eq, de_eq, cs['Pd'], *args_of(cs), 1e-12, 30)
    for nm, an in (('A0', False), ('A2', False), ('A2', True), ('A3', True)):
        make_net_mpc(cs, nm, coef, rel_ref, eps_s, analytic=an)(0.0, de_eq, np.zeros(cs['ng']), th0w, np.zeros(cs['ng']), cs['Pd'])
    rng = np.random.default_rng(6001)
    names = ['C0', 'A0', 'A2fd', 'A2', 'A3']; cl = {n: [] for n in names}
    for i in range(nreal):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
        de0 = de_eq + rng.normal(0, 0.03, cs['ng']); w0 = rng.normal(0, 0.1, cs['ng'])
        th0, _ = solve_theta(th_eq, de0, cs['Pd'], *args_of(cs), 1e-12, 30)
        for nm in names:
            if nm == 'C0': ctrl = static_ctrl(cs)
            elif nm == 'A2fd': ctrl = make_net_mpc(cs, 'A2', coef, rel_ref, eps_s)
            elif nm in ('A2', 'A3'): ctrl = make_net_mpc(cs, nm, coef, rel_ref, eps_s, analytic=True)
            else: ctrl = make_net_mpc(cs, nm, coef, rel_ref, eps_s)
            lg = simulate_plant(cs, th0, de0, w0, ctrl, Tcl, load)
            cl[nm].append(net_metrics(cs, lg))
        print(name, 'run', i, {n: (round(cl[n][-1]['angle_peak_deg'], 2), round(cl[n][-1]['w_rms'], 4), round(cl[n][-1]['solve_ms'], 1)) for n in names}, flush=True)
    keys = ['angle_peak_deg', 'viol_integral', 'w_rms', 'u_energy', 'solve_ms']
    csv(f'{name}_analytic_closed_loop.csv', ['controller'] + [f'{k}_mean' for k in keys], [(n,) + tuple(float(np.mean([m[k] for m in cl[n]])) for k in keys) for n in names])
    csv(f'{name}_analytic_closed_loop_runs.csv', ['run', 'controller'] + keys, [(i, n) + tuple(cl[n][i][k] for k in keys) for n in names for i in range(nreal)])


if __name__ == '__main__':
    rows = []
    for nm in ('case9', 'case39'):
        e, ta, tf = gradient_check(nm); rows.append((nm, e, ta, tf)); print(nm, 'gradient check', e, ta, tf, flush=True)
    csv('network_gradient_check.csv', ['case', 'max_rel_gradient_error', 'ms_analytic', 'ms_finite_difference'], rows)
    for nm in ('case9', 'case39'):
        closed_loop(nm)
