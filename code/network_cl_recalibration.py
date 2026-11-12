"""Policy-matched recalibration of the closed-loop envelope (elevated loading, controller A2, disturbance and initial-state generator of the closed-loop study).
Calibration sets, pilot set, test pool and the evaluation realizations of the closed-loop table use disjoint seeds, so no trajectory is reused.
  pilot  10 trajectories (seed 80000): step weights w_k for the joint envelope (a scalar score max_k s_k/w_k)
  R sets of n=20 fresh trajectories (seeds 81000+i): calibration; the production envelope uses sets 0 and 1 (n=40)
  M fresh test trajectories (seed 82000)
  closed-loop evaluation realizations of the main study (seed 6001, 6 or 4 runs), disjoint from all of the above.
A2 uses the analytic gradient here (identical closed-loop behaviour up to finite-difference noise, see the main text); this is the only difference from the generator of the main study.
What the theorem covers: marginal per-step (and, for the scaled envelope, joint) coverage of trajectory-level scores for NEW trajectories of the same policy and generator, under exchangeability.
What it does not cover: another policy, another disturbance law, constraint satisfaction or stability of the closed loop."""
import sys, json
import numpy as np
from scipy.stats import t as tdist
from network_experiments import *

ALPHA = 0.05; NCAL = 20


def setup(name):
    c = build_case(name, load_scale=STRESS[name]); th_eq, de_eq, _ = equilibrium(c)
    train = make_training(c, th_eq, de_eq, 30, 1001, 1e-3); val = make_training(c, th_eq, de_eq, 8, 1002, 1e-3)
    coef, _, _ = fit_net(c, train, val, th_eq); rel_ref = de_eq - np.sum(c['M'] * de_eq) / np.sum(c['M'])
    return c, th_eq, de_eq, coef, rel_ref


def cl_runs(c, th_eq, de_eq, coef, rel_ref, n, seed, T, gam=GAMMA, eps=None, ctrl_name='A2'):
    """Closed-loop trajectories with the generator of the main closed-loop study."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28))
        de0 = de_eq + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        ctrl = make_net_mpc(c, ctrl_name, coef, rel_ref, eps if eps is not None else np.zeros(6), gam=gam, analytic=True)
        out.append((load, simulate_plant(c, th0, de0, w0, ctrl, T, load)))
    return out


def eps_of(S): return np.array([conformal_quantile(S[:, k], ALPHA) for k in range(S.shape[1])])


def ci(x):
    x = np.asarray(x); return float(tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def metrics(c, lg, gam):
    ang = np.array([np.max(np.abs(t[c['lf']] - t[c['lt']])) for t in lg['th']])
    return dict(peak=float(np.degrees(ang.max())), viol=float(np.sum(np.maximum(0, ang - gam)) * TS), frac=float(np.mean(ang > gam)),
                w_rms=float(np.sqrt(np.mean(lg['w'][len(lg['w']) // 3:] ** 2))), solve_ms=1e3 * float(lg['solve']))


def run(name, R, M, quick=False):
    c, th_eq, de_eq, coef, rel_ref = setup(name); T = 6.0 if name == 'case9' else 5.0; nreal = 6 if name == 'case9' else 4
    old = np.loadtxt(RES / f'{name}_tube_envelope_stress.csv', delimiter=',', skiprows=1)[1:6, 1]
    S = lambda n, seed: envelope_traj(c, cl_runs(c, th_eq, de_eq, coef, rel_ref, n, seed, T), coef)
    pilot = S(10, 80000); w = pilot.mean(axis=0); cal = [S(NCAL, 81000 + i) for i in range(R)]; test = S(M, 82000); print(name, 'scores done', flush=True)
    rows = []; ok_old = test <= old[None, :]
    rows.append(('original 10/7-trajectory envelope (static law + A2 mix)', 7, 'marginal step 3', float(ok_old[:, 2].mean()), np.nan)); rows.append(('original envelope', 7, 'steps 1-5 jointly', float(np.mean(ok_old.all(axis=1))), np.nan))
    r = int(np.ceil((NCAL + 1) * (1 - ALPHA))); lvl = r / (NCAL + 1)
    cm = np.array([(test <= eps_of(X)[None, :]).mean(axis=0) for X in cal]); lam = [np.sort((X / w).max(axis=1))[r - 1] for X in cal]
    cj = np.array([np.mean((test / w).max(axis=1) <= l) for l in lam]); pj = np.array([np.mean(np.all(test <= eps_of(X)[None, :], axis=1)) for X in cal])
    for k in range(cm.shape[1]): rows.append(('policy-matched n=20', NCAL, f'marginal step {k+1}', float(cm[:, k].mean()), ci(cm[:, k])))
    rows.append(('policy-matched n=20', NCAL, 'steps 1-5 jointly; scaled envelope (guaranteed)', float(cj.mean()), ci(cj)))
    rows.append(('policy-matched n=20', NCAL, 'steps 1-5 jointly; per-step envelopes (not guaranteed)', float(pj.mean()), ci(pj)))
    both = np.vstack(cal[:2]); eps_new = eps_of(both); ok_new = test <= eps_new[None, :]
    rows.append(('production envelope sets 0+1 (n=40)', 40, 'marginal step 3 on the test pool', float(ok_new[:, 2].mean()), np.nan)); rows.append(('production envelope (n=40)', 40, 'steps 1-5 jointly (per-step envelopes)', float(np.mean(ok_new.all(axis=1))), np.nan))
    csv(f'{name}_cl_recal_coverage.csv', ['calibration', 'n_cal', 'quantity', 'coverage', 'ci95_half_width'], rows)
    csv(f'{name}_tube_envelope_stress_recal.csv', ['k', 'eps_line_angle_rad'], [(0, 0.0)] + [(k + 1, float(eps_new[k])) for k in range(5)])
    print(name, 'old eps', np.round(old, 4), 'new eps', np.round(eps_new, 4), flush=True)
    # control performance on the realizations of the closed-loop study (seed 6001), disjoint from all calibration trajectories
    rng = np.random.default_rng(6001); perf = []
    ee = np.r_[0.0, eps_new]
    for i in range(nreal):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_eq + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
        th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        for nm in ('C0', 'A2', 'A3'):
            ctrl = static_ctrl(c) if nm == 'C0' else make_net_mpc(c, nm, coef, rel_ref, ee, gam=GAMMA, analytic=True)
            perf.append((i, nm, 'limit 30 deg', *metrics(c, simulate_plant(c, th0, de0, w0, ctrl, T, load), GAMMA).values()))
    if name == 'case9':        # constraint-active scenario: line-angle limit 23 degrees, calibration with the 23-degree policy
        g23 = np.radians(23.0); cal23 = envelope_traj(c, cl_runs(c, th_eq, de_eq, coef, rel_ref, 40, 83000, T, gam=g23), coef); e23 = eps_of(cal23); t23 = envelope_traj(c, cl_runs(c, th_eq, de_eq, coef, rel_ref, 100, 84000, T, gam=g23), coef)
        rows2 = [('limit 23 deg policy; envelope n=40', 40, 'marginal step 3 on 100 fresh runs', float((t23 <= e23[None, :])[:, 2].mean()), np.nan), ('limit 23 deg policy; envelope n=40', 40, 'steps 1-5 jointly (per-step envelopes)', float(np.mean((t23 <= e23[None, :]).all(axis=1))), np.nan)]
        csv(f'{name}_cl_recal_coverage_limit23.csv', ['calibration', 'n_cal', 'quantity', 'coverage', 'ci95_half_width'], rows2)
        rng = np.random.default_rng(6001)
        for i in range(nreal):
            load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_eq + rng.normal(0, 0.03, c['ng']); w0 = rng.normal(0, 0.1, c['ng'])
            th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
            for nm in ('C0', 'A2', 'A3'):
                ctrl = static_ctrl(c) if nm == 'C0' else make_net_mpc(c, nm, coef, rel_ref, np.r_[0.0, e23], gam=g23, analytic=True)
                perf.append((i, nm, 'limit 23 deg', *metrics(c, simulate_plant(c, th0, de0, w0, ctrl, T, load), g23).values()))
    csv(f'{name}_cl_recal_performance.csv', ['run', 'controller', 'scenario', 'peak_deg', 'viol_integral_rad_s', 'frac_time_over_limit', 'w_rms', 'solve_ms'], perf)
    json.dump(dict(case=name, R=R, M=M, ncal=NCAL, seeds=dict(pilot=80000, calibration='81000+i', test=82000, limit23_cal=83000, limit23_test=84000, evaluation=6001), w=w.tolist()), open(RES / f'{name}_cl_recal_meta.json', 'w'))


if __name__ == '__main__':
    nm = sys.argv[1]; R = int(sys.argv[2]); M = int(sys.argv[3]); run(nm, R, M)
