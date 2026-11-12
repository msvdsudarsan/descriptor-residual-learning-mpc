"""Task 1(a),(b): index-1 regularity and nonexpansion (row-stochastic) property of d(theta)/d(delta), AC vs lossless.
Random states: delta = delta_eq + N(0, sd), sd in {0.05, 0.15, 0.3}; load factor U(0.7, 1.4); keep states whose line angle differences and
rotor-terminal angle differences are <= 30 deg (same filter as ../../check_stochastic.py).  Usage: python ac_props.py case9|case39 [nstates]"""
import sys, numpy as np
from pathlib import Path
from ac_model import *
RES = Path(__file__).resolve().parents[1] / 'results'
G30 = np.radians(30.0)

def kspan(a, m):
    """sup of a.e over e with weighted mean zero (sum m e = 0) and span(e) <= 1."""
    at = a - a.sum() * m / m.sum()
    return max(at[at > 0].sum(), -at[at < 0].sum())

def run(name, N, npw, nqw, seed, flo=0.7, fhi=1.4):
    c = build_ac(name, npw=npw, nqw=nqw); z0, de0, _ = nominal_point(c); net = net_of(c); n, ng = c['n'], c['ng']
    rng = np.random.default_rng(seed); cnt = tries = rej_conv = rej_ang = 0
    sig = []; minent = 1e9; rowdev = 0.0; infn = 0.0; kl = 0.0; kpsi = 0.0; shift_line = 0.0; negs = 0; Vsens = 0.0; kv = 0.0; lossless_like = []
    gen_rows = np.zeros((ng, ng))
    while cnt < N and tries < 4000:
        tries += 1; sd = rng.choice([0.05, 0.15, 0.3]); fac = rng.uniform(flo, fhi)
        de = de0 + rng.normal(0, sd, ng)
        z, nit, gres = solve_z(z0, de, fac, net, 1e-12, 40)
        if gres > 1e-9 or not np.all(np.isfinite(z)) or z[n:].min() < 0.5:
            rej_conv += 1; continue
        th = z[:n]
        if not (np.all(np.abs(th[c['lf']] - th[c['lt']]) <= G30) and np.all(np.abs(de - th[c['gbus']]) <= G30)):
            rej_ang += 1; continue
        cnt += 1
        _, J = ac_res_jac(z, de, fac, net, True); sig.append(np.linalg.svd(J, compute_uv=False)[-1])
        S = -np.linalg.solve(J, dg_ddelta(z, de, net)); St = S[:n]; Sv = S[n:]
        minent = min(minent, St.min()); rowdev = max(rowdev, np.abs(St.sum(1) - 1).max()); infn = max(infn, np.abs(St).sum(1).max()); negs += (St < -1e-12).sum()
        Vsens = max(Vsens, np.abs(Sv).max()); kv = max([kv] + [kspan(Sv[j], c['M']) for j in range(n)])
        for l in range(len(c['lf'])):
            a = St[c['lf'][l]] - St[c['lt'][l]]; kl = max(kl, kspan(a, c['M'])); shift_line = max(shift_line, abs(St[c['lf'][l]].sum() - St[c['lt'][l]].sum()))
        for i in range(ng):
            a = -St[c['gbus'][i]].copy(); a[i] += 1.0; kpsi = max(kpsi, kspan(a, c['M']))
    sig = np.array(sig)
    if cnt == 0:
        row = dict(case=name, np=npw, nq=nqw, states=0, tries=tries, rej_nonconv=rej_conv, rej_angle=rej_ang, fac_range=(flo, fhi)); print(row, flush=True); return row
    row = dict(case=name, np=npw, nq=nqw, states=cnt, tries=tries, rej_nonconv=rej_conv, rej_angle=rej_ang, sigma_min_min=sig.min(), sigma_min_med=np.median(sig), sigma_min_nominal=np.linalg.svd(ac_res_jac(z0, de0, 1.0, net, True)[1], compute_uv=False)[-1],
               min_entry=minent, n_negative_entries=int(negs), max_rowsum_dev=rowdev, max_inf_norm=infn, max_line_Kspan=kl, max_psi_Kspan=kpsi, max_rowsum_diff_over_lines=shift_line, max_abs_dV_ddelta=Vsens, max_V_Kspan=kv, fac_range=(flo, fhi))
    print(row, flush=True); return row

def run_lossless(name, N, seed):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    from test_props_independent import mats, inside
    from network_validity_gate import nominal_fit
    from network_experiments import args_of
    from network_benchmark import solve_theta
    c, th_eq, de_eq, coef = nominal_fit(name); rng = np.random.default_rng(seed); ng = c['ng']; cnt = tries = 0
    minent = 1; rowdev = 0; infn = 0; kl = 0; kpsi = 0; sig = []
    while cnt < N and tries < 20000:
        tries += 1; sd = rng.choice([0.05, 0.15, 0.3]); de = de_eq + rng.normal(0, sd, ng)
        th, _ = solve_theta(th_eq, de, c['Pd'] * rng.uniform(0.7, 1.4), *args_of(c), 1e-13, 60)
        if not inside(c, de, th): continue
        cnt += 1; J, _, E = mats(c, de, th); Cm = np.diag(c['bg'] * np.cos(de - th[c['gbus']])); S = np.linalg.solve(J, E @ Cm)
        sig.append(np.linalg.svd(J, compute_uv=False)[-1]); minent = min(minent, S.min()); rowdev = max(rowdev, np.abs(S.sum(1) - 1).max()); infn = max(infn, np.abs(S).sum(1).max())
        for l in range(len(c['lf'])): kl = max(kl, kspan(S[c['lf'][l]] - S[c['lt'][l]], c['M']))
        for i in range(ng):
            a = -S[c['gbus'][i]].copy(); a[i] += 1.0; kpsi = max(kpsi, kspan(a, c['M']))
    row = dict(case=name, model='lossless', states=cnt, sigma_min_min=min(sig), min_entry=minent, max_rowsum_dev=rowdev, max_inf_norm=infn, max_line_Kspan=kl, max_psi_Kspan=kpsi)
    print(row, flush=True); return row

if __name__ == '__main__':
    name = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rows = [run_lossless(name, N, 1)] + [run(name, N, 0, 0, 11), run(name, N, 2, 2, 12), run(name, N, 0, 0, 13, 1.4, 1.9)]
    import json; (RES / f'{name}_ac_props.json').write_text(json.dumps(rows, indent=1, default=float))
