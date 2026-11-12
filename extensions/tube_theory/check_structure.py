"""Verifies, on random admissible states, the structural facts used in THEORY.md:
 (S1) K = df/ddelta symmetric, 0 <= K <= Kd (Loewner), K 1 = 0
 (S2) K = sum_e w_e z_e z_e^T with w_e = b_e cos A_e, z_e = dA_e/ddelta (Kron-reduction representation)
 (S3) K >= c0 K1 where c0 = cos(gamma_reg), K1 = same with all cosines = 1
 (S4) -mu K <= dK/dt <= mu K with mu = max_e |tan A_e| * span(omega)   (finite-difference K-dot)
 (S5) J >= c0 J1 (Loewner)
 (S6) load sensitivity: d f / d Pd = Kd C J^-1 >= 0 and has column sums 1; (l1 contraction)
"""
import sys; sys.path.insert(0, 'code')
import numpy as np
from tubelib import *
GREG = np.radians(60.0)   # region where we test
def run(name, N, seed):
    c = build_case(name); th_eq, de_eq, _ = equilibrium(c); rng = np.random.default_rng(seed); ng = c['ng']
    C = Cmat(c); J1 = J1_of(c); Kd1 = np.diag(c['bg']); K1 = Kd1 - Kd1 @ C.T @ np.linalg.solve(J1, C @ Kd1); K1 = .5 * (K1 + K1.T)
    lam2 = np.linalg.eigvalsh(K1)[1]
    worst = dict(sym=0, psd=0, null=0, repr=0, c0=1e9, dotup=-1e9, dotlo=1e9, J=1e9, Pcol=0, Pneg=0)
    cnt = 0; tries = 0; ratio_max = 0
    while cnt < N and tries < 20000:
        tries += 1
        de = de_eq + rng.normal(0, rng.choice([0.05, 0.2, 0.5]), ng)
        Pd = c['Pd'] * rng.uniform(0.6, 1.5)
        th = solve_alg(c, th_eq, de, Pd)
        A = angles(c, th, de)
        if np.abs(A).max() > GREG or np.abs(A).max() < 1e-3: continue
        J = J_of(c, th, de)
        if np.linalg.eigvalsh(J)[0] <= 0: continue
        cnt += 1
        K, S, Kd = stiffness(c, th, de)
        worst['sym'] = max(worst['sym'], np.abs(K - K.T).max() / np.abs(K).max())
        ev = np.linalg.eigvalsh(K); worst['psd'] = min(worst['psd'], ev[0] / ev[-1]) if cnt > 1 else ev[0] / ev[-1]
        worst['null'] = max(worst['null'], np.abs(K @ np.ones(ng)).max() / np.abs(K).max())
        # (S1) K <= Kd
        worst.setdefault('Kle', -1e9); worst['Kle'] = max(worst['Kle'], np.linalg.eigvalsh(K - Kd)[-1] / np.abs(Kd).max())
        # (S2) representation
        Z = edge_Z(c, S); w = np.r_[c['lb'] * np.cos(A[:len(c['lf'])]), c['bg'] * np.cos(A[len(c['lf']):])]
        Kr = Z.T @ np.diag(w) @ Z; worst['repr'] = max(worst['repr'], np.abs(K - Kr).max() / np.abs(K).max())
        # (S3)
        c0 = np.cos(np.abs(A).max()); worst['c0'] = min(worst['c0'], np.linalg.eigvalsh(K - c0 * K1)[0] / np.abs(K).max())
        worst['J'] = min(worst['J'], np.linalg.eigvalsh(J - c0 * J1)[0] / np.abs(J).max())
        # (S4) Kdot by central difference along random velocity
        om = rng.normal(0, 1, ng); h = 1e-6
        def Kat(d):
            t = solve_alg(c, th, d, Pd); return stiffness(c, t, d)[0]
        Kdot = (Kat(de + h * om) - Kat(de - h * om)) / (2 * h)
        mu = np.abs(np.tan(A)).max() * (om.max() - om.min())
        # relative eigenvalues of (Kdot, K) on 1-perp
        P = np.eye(ng) - np.ones((ng, ng)) / ng
        Q = np.linalg.eigh(P)[1][:, 1:]    # basis of 1-perp
        Kp = Q.T @ K @ Q; Kdp = Q.T @ Kdot @ Q
        L = np.linalg.cholesky(Kp); Li = np.linalg.inv(L); G = Li @ Kdp @ Li.T
        g = np.linalg.eigvalsh(0.5 * (G + G.T)); worst['dotup'] = max(worst['dotup'], g[-1] / mu); worst['dotlo'] = min(worst['dotlo'], g[0] / mu)
        # (S6)
        Pm_ = Kd @ C.T @ np.linalg.inv(J)
        worst['Pcol'] = max(worst['Pcol'], np.abs(Pm_.sum(0) - 1).max()); worst['Pneg'] = min(worst['Pneg'], Pm_.min())
    print(name, 'states', cnt, 'lam2(K1)=%.4g' % lam2)
    for k, v in worst.items(): print('   %-6s %.3e' % (k, v))
    print('   (S4) expected: dotup/mu <= 1 and dotlo/mu >= -1  ->', 'OK' if worst['dotup'] <= 1 + 1e-4 and worst['dotlo'] >= -1 - 1e-4 else 'VIOLATED')
run('case9', 60, 11); run('case39', 40, 12)
