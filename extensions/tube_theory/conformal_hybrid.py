"""Step 2(b): hybrid margin  eps_k = G * q_span,k + L_k  (conformal on the rotor-angle SPAN error + deterministic Lemma-1 transfer + load term).
Proved statement: THEORY.md Theorem 2.  Here: build the margin on case9 (stress loading, as in the paper's T-B regime) and verify coverage on fresh runs
(in-family, out-of-family).  Scores are RUN-LEVEL maxima over all windows of a run (as in the paper), so one number per run and step.
Usage: python3 conformal_hybrid.py [case9|case39] [stress]"""
import sys, json; sys.path.insert(0, 'code')
import numpy as np
from env import *
name = sys.argv[1] if len(sys.argv) > 1 else 'case9'; stress = (len(sys.argv) > 2 and sys.argv[2] == 'stress')
c, th_eq, de_eq, coef = make_env(name, stress=stress); mode, use_res = MODES['A2']; N = 5; ts = TS * np.arange(1, N + 1)
LO, HI = (0.07, 0.11) if stress else (0.0, 0.08)
def runs(n, seed, hi=HI, lo=LO, ic=0.10, wic=0.3, lom=(1.0, 6.0)):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        de0 = de_eq + rng.normal(0, ic, c['ng']); w0 = rng.normal(0, wic, c['ng']); th0, _ = solve_theta(th_eq, de0, c['Pd'], *args_of(c), 1e-12, 30)
        load = (rng.uniform(lo, hi), rng.uniform(*lom), rng.uniform(0, 6.28)); out.append((load, simulate_plant(c, th0, de0, w0, static_ctrl(c), 6.0, load)))
    return out
def scores(run):
    """run-level max over windows of: span error, total angle error, rotor-part angle error, load-part angle error (each per step k=1..N)"""
    load, lg = run; S = np.zeros((4, N)); G = 0.0
    for k in range(0, len(lg['t']) - 1 - N, 4):
        Pd_k = c['Pd'] * (1 + load[0] * np.sin(load[1] * lg['t'][k] + load[2])); de, w, th = lg['de'][k].copy(), lg['w'][k].copy(), lg['th'][k].copy()
        for j in range(N):
            for _ in range(NSUB):
                de, w, th = pred_step(de, w, th, lg['u'][k + j], TS / NSUB, Pd_k, mode, use_res, coef, *args_of(c), c['M'], c['Dphys'], c['Pm'], c['kappa'], c['fric'])
            dp, tp = lg['de'][k + j + 1], lg['th'][k + j + 1]; ta = solve_alg(c, tp, dp, Pd_k)
            tot = np.abs(angles(c, th, de) - angles(c, tp, dp)).max(); rot = np.abs(angles(c, th, de) - angles(c, ta, dp)).max(); ld = np.abs(angles(c, ta, dp) - angles(c, tp, dp)).max()
            for i, v in enumerate((np.ptp(de - dp), tot, rot, ld)): S[i, j] = max(S[i, j], v)
    return S
def conf_q(x, alpha):
    x = np.sort(x); n = len(x); r = int(np.ceil((n + 1) * (1 - alpha)))
    return x[r - 1] if r <= n else np.inf
ALPHA = 0.05
print(f'{name} stress={stress}: generating runs ...', flush=True)
cal = runs(39, 8101); fam = runs(100, 8301); ood1 = runs(40, 8401, hi=0.18, lo=0.12); ood2 = runs(40, 8501, ic=0.25, wic=0.7)
Sc = np.array([scores(r) for r in cal]); Sf = np.array([scores(r) for r in fam]); So1 = np.array([scores(r) for r in ood1]); So2 = np.array([scores(r) for r in ood2])
q_span = np.array([conf_q(Sc[:, 0, k], ALPHA) for k in range(N)]); q_tot = np.array([conf_q(Sc[:, 1, k], ALPHA) for k in range(N)])
q_rot = np.array([conf_q(Sc[:, 2, k], ALPHA) for k in range(N)]); q_ld = np.array([conf_q(Sc[:, 3, k], ALPHA / 2) for k in range(N)]); q_span2 = np.array([conf_q(Sc[:, 0, k], ALPHA / 2) for k in range(N)])
# deterministic load term (hypothesis: |lambda| <= lbar(t) = min(2 lamp_max, lamp_max*6*t), lamp_max = HI) with calibrated and rigorous constants
K = constants(c, coef, 60.0 if stress else 45.0, 2.4)
states = [(lg['th'][k], lg['de'][k]) for _, lg in cal for k in range(len(lg['t']))]; phi_c, lam_c = phi_load(c, states)
lbar = np.minimum(2 * HI, HI * 6.0 * ts); L_rig = K['Lam'] * lbar; L_cal = lam_c * lbar
print('Lam rigorous %.3f  calibrated %.3f (rad per unit lambda); lambda_bar(t)=%s' % (K['Lam'], lam_c, np.round(lbar, 3)))
H = {'direct conformal on total angle error (paper-type, alpha=0.05)': q_tot,
     'H1: q_span(0.05) + rigorous Lam*lbar  [G=1]': q_span + L_rig,
     'H2: q_span(0.05) + calibrated Lam*lbar [G=1]': q_span + L_cal,
     'H3: q_span(alpha/2) + q_load(alpha/2) [union, G=1]': q_span2 + q_ld,
     'H4: q_rot(0.05)[conformal on rotor part] + calibrated Lam*lbar': q_rot + L_cal}
print('step k (t[s])      :', ' '.join('%7.1f' % t for t in ts))
for k_, v in H.items(): print('%-62s' % k_, ' '.join('%7.3f' % np.degrees(x) for x in v), 'deg')
print('median observed total angle error (cal runs max-level):', ' '.join('%7.3f' % np.degrees(x) for x in np.median(Sc[:, 1, :], axis=0)), 'deg')
def cover(S, name_, marg):
    ok_tot = np.all(S[:, 1, :] <= marg[None, :], axis=1); return ok_tot.mean()
print('\nEMPIRICAL COVERAGE (fraction of fresh runs whose angle error stays <= margin at all k=1..5 for every window):')
fams = {'in-family (100 runs)': Sf, 'OOD: load amp 0.12-0.18 > design lamp_max (40)': So1, 'OOD: initial offsets x2.5, speeds x2.3 (40)': So2}
print('%-62s' % '', *['%-14s' % f[:14] for f in fams])
for k_, v in H.items(): print('%-62s' % k_, *['%-14.3f' % cover(S, f, v) for f, S in fams.items()])
print('span-only coverage (q_span, alpha=0.05; claim >= 0.95):', {f[:12]: float(np.mean(np.all(S[:, 0, :] <= q_span[None, :], axis=1))) for f, S in fams.items()})
print('Lemma-1 deterministic check on every run/step: rotor-part error <= span : violations', int(sum((S[2, :] > S[0, :] * (1 + 1e-9) + 1e-12).sum() for S in np.concatenate([Sc, Sf, So1, So2]))))
print('load-term deterministic check (calibrated Lam, in-family): load-part error <= Lam_cal*lbar violations', int((Sf[:, 3, :] > L_cal[None, :]).sum()), '; OOD amp:', int((So1[:, 3, :] > L_cal[None, :]).sum()))
tag = f'{name}{"_stress" if stress else ""}'
json.dump({k.split(':')[0]: v.tolist() for k, v in H.items()} | {'ts': ts.tolist(), 'L_rig': L_rig.tolist(), 'L_cal': L_cal.tolist(), 'q_span': q_span.tolist(), 'q_tot': q_tot.tolist()}, open(f'results/hybrid_margin_{tag}.json', 'w'))
