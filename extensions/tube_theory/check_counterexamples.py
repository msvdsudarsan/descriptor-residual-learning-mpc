"""T-C: where does Lemma 1 (S = d theta/d delta >= 0, row sums 1, gain G<=1) fail, and what does that do to the margin transfer?
G := max over constrained angles e of (1/2)||z_e||_1, z_e = d A_e/d delta  (= best constant in |dA_e| <= G * span(d delta), since z_e sums to 0).
"""
import sys; sys.path.insert(0, 'code')
import numpy as np
from tubelib import *
rng = np.random.default_rng(5)

def gain(c, th, de):
    C = Cmat(c); J = J_of(c, th, de); kd = c['bg'] * np.cos(de - th[c['gbus']])
    ev = np.linalg.eigvalsh(J)
    if abs(ev[0]) < 1e-9: return None
    S = np.linalg.solve(J, C @ np.diag(kd)); Z = edge_Z(c, S)
    return dict(G=0.5 * np.abs(Z).sum(1).max(), minS=S.min(), rows=np.abs(S.sum(1) - 1).max(), lam=ev[0], PD=ev[0] > 0)

print('=== A. random (theta, delta) states on case9 / case39 with constrained angles up to 130 deg (Pd is irrelevant to J, any state is an equilibrium of some Pd) ===')
for name in ('case9', 'case39'):
    c = build_case(name); n, ng = c['n'], c['ng']; res = dict(in_ok=0, in_bad=0, in_maxG=0, out=0, out_Gbad=0, out_PD_Gbad=0, out_maxG=0, out_negS=0, rowbad=0)
    for it in range(30000):
        spread = rng.choice([0.3, 0.8, 1.5]); th = rng.normal(0, spread, n); de = th[c['gbus']] + rng.normal(0, spread, ng)
        r = gain(c, th, de)
        if r is None: continue
        inside = (np.cos(th[c['lf']] - th[c['lt']]) > 0).all() and (np.cos(de - th[c['gbus']]) > 0).all()
        res['rowbad'] = max(res['rowbad'], r['rows'] if r['rows'] < 1e-6 else 0)
        if inside:
            if r['G'] > 1 + 1e-9: res['in_bad'] += 1
            else: res['in_ok'] += 1
            res['in_maxG'] = max(res['in_maxG'], r['G'])
        else:
            res['out'] += 1; res['out_maxG'] = max(res['out_maxG'], r['G']); res['out_negS'] += r['minS'] < -1e-9
            if r['G'] > 1 + 1e-9:
                res['out_Gbad'] += 1; res['out_PD_Gbad'] += r['PD']
    print(name, res)

print('\n=== A2. explicit 3-bus counterexample: J positive definite, one line angle 100 deg (cos<0): Lemma 1 fails, margin transfer gain G>1 ===')
# bus0 (gen0), bus1 (load), bus2 (gen1); lines 0-1 (b=5), 1-2 (b=5); bg=10 each
c3 = dict(Pd=np.zeros(3), n=3, ng=2, lf=np.array([0, 1]), lt=np.array([1, 2]), lb=np.array([5., 5.]), gbus=np.array([0, 2]), bg=np.array([10., 10.]))
th = np.array([0.0, -np.radians(100.0), -np.radians(100.0) - np.radians(15.0)]); de = th[c3['gbus']] + np.radians([10.0, 10.0])
J = J_of(c3, th, de); print('J eigenvalues', np.linalg.eigvalsh(J), '(positive definite although cos(line 0-1) = %.3f < 0)' % np.cos(th[0] - th[1]))
r = gain(c3, th, de); print('G =', r['G'], ' min entry of S =', r['minS'], ' max|rowsum-1| =', r['rows'])
C = Cmat(c3); kd = c3['bg'] * np.cos(de - th[c3['gbus']]); S = np.linalg.solve(J, C @ np.diag(kd)); print('S =\n', S)
Z = edge_Z(c3, S); print('line-angle gains 1/2||z||_1 per edge (lines then gens):', 0.5 * np.abs(Z).sum(1))
# direct nonlinear demonstration: perturb rotor angles with span 0.01 rad, re-solve, compare line angle change
Pd = np.zeros(3)
g0 = alg_res_jac(th, de, Pd, *args_of(c3), False)[0]; Pd = -g0 * 0 + (alg_res_jac(th, de, np.zeros(3), *args_of(c3), False)[0])  # Pd := sin-sum so that g=0 at (th,de)
for sgn in (+1, -1):
    d2 = de + sgn * np.array([0.0, 0.01]); t2, _ = solve_theta(th, d2, Pd, *args_of(c3), 1e-13, 50)
    dA = np.abs((t2[c3['lf']] - t2[c3['lt']]) - (th[c3['lf']] - th[c3['lt']])).max(); print('  delta2 += %+0.3f: span=0.01 -> max line-angle change %.5f (ratio %.2f)' % (sgn * 0.01, dA, dA / 0.01))

print('\n=== B. generator-less component (bus 5 of case9 islanded by removing lines 4-5, 5-6): J singular, algebraic equations inconsistent ===')
c = build_case('case9'); keep = [l for l in range(len(c['lf'])) if not ({int(c['lf'][l]), int(c['lt'][l])} & {4} )]   # bus index 4 = bus 5
cc = dict(c); cc['lf'], cc['lt'], cc['lb'] = c['lf'][keep], c['lt'][keep], c['lb'][keep]
th0, de0, _ = equilibrium(c); J = J_of(cc, th0, de0); ev = np.linalg.eigvalsh(J)
print('lines removed:', len(c['lf']) - len(keep), ' smallest J eigenvalues:', ev[:2], ' Pd at the isolated bus: %.3f pu' % c['Pd'][4])
g = alg_res_jac(th0, de0, c['Pd'], *args_of(cc), False)[0]; print('residual g at isolated bus (cannot be removed by any theta, J row is zero):', g[4])
print('=> the descriptor model is not index-1 there: Newton step is undefined, S does not exist; margin transfer has no meaning.')

print('\n=== B2. islanding where every island keeps a generator (case9: remove lines 4-5 and 6-7 -> island {5,6,3} with generator 3): per-component Lemma 1 holds, but a shift of one island\'s rotors moves that island\'s angles ===')
c = build_case('case9'); th0, de0, _ = equilibrium(c)
bad = [(3, 4), (5, 6)]   # 0-based bus pairs (4-5) and (6-7)
keep = [l for l in range(len(c['lf'])) if (int(c['lf'][l]), int(c['lt'][l])) not in bad and (int(c['lt'][l]), int(c['lf'][l])) not in bad]
cc = dict(c); cc['lf'], cc['lt'], cc['lb'] = c['lf'][keep], c['lt'][keep], c['lb'][keep]
th0, _n = solve_theta(th0, de0, c['Pd'], *args_of(cc), 1e-13, 60); print('re-solved algebraic baseline on the islanded network: residual', np.abs(alg_res_jac(th0, de0, c['Pd'], *args_of(cc), False)[0]).max(), ' max line angle %.1f deg' % np.degrees(np.abs(th0[cc['lf']] - th0[cc['lt']]).max()))
C = Cmat(cc); J = J_of(cc, th0, de0); kd = cc['bg'] * np.cos(de0 - th0[cc['gbus']]); S = np.linalg.solve(J, C @ np.diag(kd)); Z = edge_Z(cc, S)
print('J min eig %.3f ; min S entry %.2e ; max|rowsum-1| %.1e ; gain G=%.3f (<=1 per component holds)' % (np.linalg.eigvalsh(J)[0], S.min(), np.abs(S.sum(1) - 1).max(), 0.5 * np.abs(Z).sum(1).max()))
sh = np.zeros(cc['ng']); sh[2] = 0.05   # shift only generator 3 (isolated island) by 0.05 rad
Pd = c['Pd']; t2, _ = solve_theta(th0, de0 + sh, Pd, *args_of(cc), 1e-12, 40)
print('shifting only gen 3 by 0.05 rad (a rotor-error span of 0.05 on the whole system, common-mode on that island): max change of ANY bus angle = %.4f rad; max change of any line/rotor-bus angle = %.4f rad (<= span 0.05)' % (np.abs(t2 - th0).max(), np.abs(np.r_[t2[cc['lf']] - t2[cc['lt']], de0 + sh - t2[cc['gbus']]] - np.r_[th0[cc['lf']] - th0[cc['lt']], de0 - th0[cc['gbus']]]).max()))
shall = np.full(cc['ng'], 0.05); t3, _ = solve_theta(th0, de0 + shall, Pd, *args_of(cc), 1e-12, 40)
print('common-mode shift of ALL rotors by 0.05: max change of constrained angles = %.2e (still zero: islands are shifted rigidly each)' % np.abs(np.r_[t3[cc['lf']] - t3[cc['lt']], de0 + shall - t3[cc['gbus']]] - np.r_[th0[cc['lf']] - th0[cc['lt']], de0 - th0[cc['gbus']]]).max())

print('\n=== C. line outage (not islanding): predictor keeps the pre-outage topology, rotor error zero (span = 0): angle error is O(1) ===')
for name in ('case9', 'case39'):
    c = build_case(name); th0, de0, _ = equilibrium(c); L = len(c['lf']); worst = []
    for l in range(L):
        keep = [k for k in range(L) if k != l]; cc = dict(c); cc['lf'], cc['lt'], cc['lb'] = c['lf'][keep], c['lt'][keep], c['lb'][keep]
        ev = np.linalg.eigvalsh(J_of(cc, th0, de0))[0]
        if ev <= 1e-6: worst.append((l, None, None)); continue
        th1, nit = solve_theta(th0, de0, c['Pd'], *args_of(cc), 1e-12, 50); g = alg_res_jac(th1, de0, c['Pd'], *args_of(cc), False)[0]
        if np.abs(g).max() > 1e-9: worst.append((l, None, None)); continue
        e = np.abs(np.r_[th1[c['lf']] - th1[c['lt']], de0 - th1[c['gbus']]] - np.r_[th0[c['lf']] - th0[c['lt']], de0 - th0[c['gbus']]])
        e[l] = 0   # the outaged line itself is not constrained afterwards
        worst.append((l, float(np.degrees(e.max())), float(np.degrees(np.abs(th1[cc['lf']] - th1[cc['lt']]).max()))))
    ok = [w for w in worst if w[1] is not None]; isl = [w[0] for w in worst if w[1] is None]
    print(name, 'lines', L, '| outages that island or break the model:', isl, '| max angle jump over the remaining ones: %.2f deg (median %.2f deg); max post-outage line angle %.2f deg' % (max(w[1] for w in ok), np.median([w[1] for w in ok]), max(w[2] for w in ok)), '; rotor-angle error span = 0 at the outage instant, so a span-based margin transfer predicts 0 error.')

print('\n=== D. approach to the loadability limit (cos still > 0): transfer gain stays <=1, but the load constant 1/lambda_min(J) blows up ===')
for name, mus in (('case9', (1.0, 2.0, 2.75, 3.0, 3.2, 3.4)), ('case39', (1.0, 1.5, 2.0, 2.25, 2.5, 2.7))):
    rows = loadability_continuation(name, mus)
    for mu, lam, cond, amax, gmax, res in rows:
        if np.isnan(lam): print(name, 'mu=%.2f: equilibrium not found (beyond loadability)' % mu); break
        c = build_case(name, load_scale=mu); th, de, _ = equilibrium(c); r = gain(c, th, de)
        print(name, 'mu=%.2f  lambda_min(J)=%.3f  max line angle %.1f deg  max rotor-bus angle %.1f deg  transfer gain G=%.3f  ||J^-1 Pd||_inf=%.3f rad' % (mu, lam, amax, gmax, r['G'], np.abs(np.linalg.solve(J_of(c, th, de), c['Pd'])).max()))
