"""Task 1(b), nonlinear check: ratio  max(|d line angle|, |d (delta - theta_gen)|) / span(e)  when delta is perturbed by e (span(e) = 0.02 rad),
over random states and box-vertex directions e (the worst case of a linear map over {span(e) <= 1}); lossless vs AC.  Usage: python ac_props2.py case9|case39"""
import sys, json, numpy as np
from pathlib import Path
from ac_model import *
from ac_props import kspan
RES = Path(__file__).resolve().parents[1] / 'results'
G30 = np.radians(30.0)

def vertices(ng, rng, nmax):
    if 2 ** ng <= nmax:
        return [np.array([(i >> b) & 1 for b in range(ng)], float) for i in range(1, 2 ** ng - 1)]
    return [rng.integers(0, 2, ng).astype(float) for _ in range(nmax)]

def ac_ratio(name, npw, nqw, flo, fhi, N, seed, span=0.02):
    c = build_ac(name, npw=npw, nqw=nqw); z0, de0, _ = nominal_point(c); net = net_of(c); n, ng = c['n'], c['ng']; rng = np.random.default_rng(seed)
    worst = 0.0; cnt = 0; tries = 0; wv = 0.0; nfail = 0
    while cnt < N and tries < 4000:
        tries += 1; de = de0 + rng.normal(0, rng.choice([0.05, 0.15, 0.3]), ng); fac = rng.uniform(flo, fhi)
        z, _, gr = solve_z(z0, de, fac, net, 1e-12, 40)
        if gr > 1e-9 or z[n:].min() < 0.5: continue
        th = z[:n]
        if not (np.all(np.abs(th[c['lf']] - th[c['lt']]) <= G30) and np.all(np.abs(de - th[c['gbus']]) <= G30)): continue
        cnt += 1
        for v in vertices(ng, rng, 64):
            e = v - (c['M'] @ v) / c['M'].sum(); e = e * span / (e.max() - e.min())
            try: zn, _, gr2 = solve_z(z, de + e, fac, net, 1e-13, 40)
            except Exception: nfail += 1; continue
            if gr2 > 1e-9: nfail += 1; continue
            dth = (zn[c['lf']] - zn[c['lt']]) - (z[c['lf']] - z[c['lt']]); dps = e - (zn[c['gbus']] - z[c['gbus']])
            r = max(np.abs(dth).max(), np.abs(dps).max()) / span; worst = max(worst, r)
            wv = max(wv, np.abs(zn[n:] - z[n:]).max() / span)
    return dict(model=f'AC np={npw} nq={nqw}', fac=(flo, fhi), states=cnt, worst_ratio=worst, worst_V_ratio=wv, perturbed_solves_failed=nfail)

def lossless_ratio(name, N, seed, span=0.02):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    from test_props_independent import inside
    from network_validity_gate import nominal_fit
    from network_experiments import args_of
    from network_benchmark import solve_theta
    c, th_eq, de_eq, coef = nominal_fit(name); rng = np.random.default_rng(seed); ng = c['ng']; cnt = 0; worst = 0.0
    while cnt < N:
        de = de_eq + rng.normal(0, rng.choice([0.05, 0.15, 0.3]), ng); Pd = c['Pd'] * rng.uniform(0.7, 1.4)
        th, _ = solve_theta(th_eq, de, Pd, *args_of(c), 1e-13, 60)
        from network_benchmark import alg_res_jac
        if not np.all(np.isfinite(th)) or np.abs(alg_res_jac(th, de, Pd, *args_of(c)[:3], c['gbus'], c['bg'], False)[0]).max() > 1e-9 or not inside(c, de, th): continue
        cnt += 1
        for v in vertices(ng, rng, 64):
            e = v - (c['M'] @ v) / c['M'].sum(); e = e * span / (e.max() - e.min())
            try: tn, _ = solve_theta(th, de + e, Pd, *args_of(c), 1e-13, 60)
            except Exception: continue
            if not np.all(np.isfinite(tn)) or np.abs(alg_res_jac(tn, de + e, Pd, *args_of(c)[:3], c['gbus'], c['bg'], False)[0]).max() > 1e-9: continue
            dth = (tn[c['lf']] - tn[c['lt']]) - (th[c['lf']] - th[c['lt']]); dps = e - (tn[c['gbus']] - th[c['gbus']])
            worst = max(worst, max(np.abs(dth).max(), np.abs(dps).max()) / span)
    return dict(model='lossless', fac=(0.7, 1.4), states=cnt, worst_ratio=worst)

if __name__ == '__main__':
    name = sys.argv[1]; N = 100
    rows = [lossless_ratio(name, N, 1), ac_ratio(name, 0, 0, 0.7, 1.4, N, 2), ac_ratio(name, 2, 2, 0.7, 1.4, N, 3), ac_ratio(name, 0, 0, 1.4, 1.9, N, 4)]
    for r in rows: print(r, flush=True)
    (RES / f'{name}_ac_props2.json').write_text(json.dumps(rows, indent=1, default=float))
