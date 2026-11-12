import sys, time, math
sys.path.insert(0, 'code'); sys.path.insert(0, '.')
from gatelib import *
from detectability_check import one_step_state
name = sys.argv[1]; N = int(sys.argv[2]); steps = [float(x) for x in sys.argv[3].split(',')]
Q0 = dict(case9=0.023839465238203714, case39=0.03706069563617859)[name]; q0 = Q0
T = 5.0; KS = 30      # step at sample 30 (t = 3.0 s); first affected innovation is sample 31
c0, th_eq, de_eq, coef = nominal_fit(name); ng = c0['ng']
# analytic lowest-order sensitivity (as tools/detectability_check.py)
v = c0['Pm'] / c0['M']; vcoi = c0['Pm'].sum() / c0['M'].sum(); d = c0['D'] / c0['M']; ph = (1 - np.exp(-d * TS)) / d; pa = TS / d - (1 - np.exp(-d * TS)) / d ** 2
g0 = np.r_[pa * (v - vcoi), ph * v]; kappa0 = float(np.sqrt(np.mean(g0 ** 2)))
print(name, 'kappa0 (lowest order, rms over 2*ng comps) = %.4f; q0 = %.5f; smallest linear step kappa0*s=q0: s = %.4f' % (kappa0, q0, q0 / kappa0), flush=True)
kfd = {}
for mu in steps:
    c1, y0, y1, u = one_step_state(c0, th_eq, de_eq, mu)
    f = lambda rho: innovation_vec(c0, coef, rho * c0['Pm'], y0, u, c1['Pd'], y1)
    g = (f(mu + 0.01) - f(mu - 0.01)) / 0.02; kfd[mu] = float(np.sqrt(np.mean(g ** 2)))
print('kappa_fd at equilibrium state per step:', {k: round(x, 4) for k, x in kfd.items()}, flush=True)
res = {}; t0 = time.time()
for mu in steps:
    S = []; Snf = []
    for i in range(N):
        load, lg = gen_run(name, c0, th_eq, de_eq, 300000 + i, mu, T, 0.06)
        S.append(scores(c0, coef, lg, np.random.default_rng(400000 + i)))
        Snf.append(scores(c0, coef, lg, np.random.default_rng(0), sig=0.0, kmax=40))
    res[mu] = (np.array(S), np.array(Snf)); print(name, 'mu', mu, 'done', round(time.time() - t0, 1), flush=True)
np.savez(OUT / f'steps_{name}.npz', **{f'S_{mu}': res[mu][0] for mu in steps}, **{f'Snf_{mu}': res[mu][1] for mu in steps})
