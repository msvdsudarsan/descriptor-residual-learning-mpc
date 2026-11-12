import json,glob,numpy as np
from scipy import stats
O=[]
def w(x): O.append(x)
# (1) certificate table summary + per-case
w(r"\subsection{Outage certificate: results on all tested outages}")
w(r"Table~\ref{tab:S6cert} gives, for each network, the extremes over the tested outages of the leverage-based quantities. $h=K\eta$ must be $\le\tfrac12$; $\rho_*/\text{exact}$ is the certified radius divided by the exact $M$-norm displacement $\|\theta^*-\theta_0\|_M$ obtained from a full Newton solve.")
w(r"\begin{table}[!ht]\centering\small\caption{Newton--Kantorovich certificate (Proposition~\ref{prop:S6}) on single-line outages.}\label{tab:S6cert}\begin{tabular}{lrcccc}\toprule Network & outages & max $h$ & all $h\le\frac12$ & $\rho_*/$exact (min) & $\rho_*/$exact (max)\\\midrule")
for c in ['case9','case39']:
    d=json.load(open(f'kantorovich_{c}.json')); h=[x['h'] for x in d]; r=[x['rho']/x['exact_E'] for x in d]
    w(f"{c} & {len(d)} & {max(h):.3f} & {'yes' if all(x['cert'] for x in d) else 'no'} & {min(r):.3f} & {max(r):.3f}\\\\")
w(r"\bottomrule\end{tabular}\end{table}")
ns=json.load(open('nearsingular.json'))
w(r"\subsection{Synthetic near-singular stress test}")
w(r"Two lines at one bus are multiplied by $\epsilon$ (a weak tie) and the load at that bus is scaled by $\epsilon$ so that the angles stay moderate, which makes the test gentler than an unscaled one; the exact post-change solution is computed by Newton's method started at the nominal angles. Rows are the two bus choices per network, concatenated in order of decreasing $\epsilon$. Failure of $h\le\frac12$ does not mean that no solution exists: Newton's method converged in $3$ iterations in every row.")
w(r"\begin{table}[!ht]\centering\small\caption{Near-singular stress test: $\lambda_{\min}(M)$, $h$ and certificate status.}\label{tab:nearsing}\begin{tabular}{lrrrcc}\toprule Network & $\epsilon$ & $\lambda_{\min}(M)$ & $h$ & certified & Newton converged\\\midrule")
for c in ['case9','case39']:
    for x in ns[c]:
        w(f"{c} & {x['eps']} & {x['lam_min_M']:.3f} & {x['h']:.3f} & {'yes' if x['cert'] else 'no'} & {'yes' if x['converged'] else 'no'}\\\\")
w(r"\bottomrule\end{tabular}\end{table}")
w(r"\begin{figure}[!ht]\centering\includegraphics[width=0.98\textwidth]{Figure_certificate.pdf}\caption{Left: certified radius $\rho_*$ against the exact post-outage displacement for the 41 outages. Right: $h=K\eta$ against $\lambda_{\min}(M)$ in the stress test; the certificate fails above $1/2$.}\label{fig:cert}\end{figure}")
# (2) twin bound
w(r"\subsection{Twin-discrepancy bound}")
tw=[json.load(open(f'verify_twin_bound_case39_seed{s}.json')) for s in (5001,5002)]
nwin=sum(x['n'] for d in tw for x in d); nviol=sum(x['viol'] for d in tw for x in d); ncv=sum(x['viol_cert'] for d in tw for x in d)
assert nviol==0 and ncv==0 and nwin==1470 and all(len(d)==35 for d in tw)
mc=[min(x['mincos'] for x in d) for d in tw]
w(r"Proposition~\ref{prop:S7} was checked on case39 for all $35$ outages and $21$ calibration trajectories (the left side $\|D_T-K_T\|_\infty$ never exceeded $\operatorname{span}(\hat\delta_T-\hat\delta_0)$: $0$ violations in $%d$ windows, $21$ per outage in each of two calibration draws (seeds 5001 and 5002); the certificate bound on $\|K_T\|_\infty$ was never violated either). The check was run on case39 only, on sampled calibration windows; the first version of the script computed $K_T$ without the removed line and was corrected before the stored run; the $21$ windows are the same for every outage, so $42$ distinct windows enter; the cosine hypothesis was checked at five points of each segment, and the smallest branch cosine was $%.2f$ (draw 1) and $%.2f$ (draw 2)." % (nwin,mc[0],mc[1]))
# (3) four draws
fs=sorted(glob.glob('results_setA/setArep*_case39_full_0.json'))
runs=[{r['line']:r for r in json.load(open(f))['rows']} for f in fs]
lines=sorted(set.intersection(*[set(r) for r in runs]))
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; hh=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-hh,c+hh
on={}
for f in ['setBrep_case39','setBrepcal5002_case39']:
    for r in json.load(open(f'results_setB/{f}_full_0.json'))['rows']: on.setdefault(r['line'],[]).append((r['online_traj_cov'],r['model_frac_le_eps']))
w(r"\subsection{Coverage over four calibration draws and the online margin}")
w(r"Each draw uses $20$ independent nominal calibration trajectories and $20$ fresh test trajectories per outage (seeds in the repository). `Twin' is the margin of Section~7 of the main text; `online' is $\varepsilon+\|D_T(w)\|_\infty$ per window with the nominal model-error margin $\varepsilon$, evaluated on two draws; `model $\le\varepsilon$' is the fraction of test trajectories whose model error stays below $\varepsilon$ (Assumption (X$_m$)).")
w(r"{\scriptsize\begin{longtable}{rcccccccc}\caption{Case39 single-line outages: twin coverage per draw, pooled coverage with a per-line $95\%$ Wilson interval, and online-margin results.}\label{tab:draws}\\\toprule Line & draw 1 & draw 2 & draw 3 & draw 4 & pooled & Wilson 95\% & online (mean of 2) & model $\le\varepsilon$\\\midrule\endfirsthead\toprule Line & draw 1 & draw 2 & draw 3 & draw 4 & pooled & Wilson 95\% & online & model $\le\varepsilon$\\\midrule\endhead")
tk=tn=0
for l in lines:
    cv=[r[l]['twin']['cov'] for r in runs]; k=sum(int(round(c*20)) for c in cv); n=20*len(runs); tk+=k; tn+=n; lo,hi=wil(k,n)
    o=on.get(l,[(np.nan,np.nan)]); oc=np.mean([a for a,b in o]); mm=np.mean([b for a,b in o])
    fl=r"\textbf{%.2f}" if k/n<0.9 else "%.2f"
    w(f"{l} & "+" & ".join(f"{c:.2f}" for c in cv)+" & "+(fl%(k/n))+f" & [{lo:.2f}, {hi:.2f}] & {oc:.2f} & {mm:.2f}\\\\")
w(r"\midrule all & & & & & "+f"{tk/tn:.3f} & & & \\\\")
w(r"\bottomrule\end{longtable}}")
C=np.array([[r[l]['twin']['cov'] for r in runs] for l in lines]); rngb=np.random.default_rng(0)
bc=[C[np.ix_(rngb.integers(0,len(lines),len(lines)),rngb.integers(0,C.shape[1],C.shape[1]))].mean() for _ in range(20000)]
bd=[C[:,rngb.integers(0,C.shape[1],C.shape[1])].mean() for _ in range(20000)]
ci_c=np.percentile(bc,[2.5,97.5]); ci_d=np.percentile(bd,[2.5,97.5])
misrows=json.load(open('results_setB/setBrepmis_case39_full_0.json'))['rows']; nmis=len(misrows); mis=np.mean([x['online_traj_cov'] for x in misrows]); c9=np.mean([x['online_traj_cov'] for x in json.load(open('results_setB/setBrep_case9_full_0.json'))['rows']])
w(r"Bold: pooled coverage below $0.9$. The per-line Wilson intervals treat the $80$ trajectories of a line as independent (each draw has its own calibration set). For the overall mean, a cluster bootstrap over calibration draws and lines gives $[%.2f, %.2f]$ and a bootstrap over draws alone $[%.2f, %.2f]$; no Wilson interval is given for the pooled $%d$ trajectories because they share calibration sets and learned models. The online margin is $\ge0.9$ on every line where the model error stays below $\varepsilon$ in at least $90\%%$ of trajectories; in the mismatch run (%d outages) the mean online coverage is $%.3f$, and on case9 it is $%.2f$ (the model error exceeds $\varepsilon$ after an outage in about half of the trajectories)." % (ci_c[0],ci_c[1],ci_d[0],ci_d[1],tn,nmis,mis,c9))
# (4) control
w(r"\subsection{Additional closed-loop realizations}")
w(r"Twelve independent realizations per network (new seed $7001$; initial rotor angles, speeds and load phase drawn anew; same controllers C0, A0 and A2 with the analytic gradient). The speed deviation is the RMS over the run; the percentage is the paired reduction relative to the static law C0, with a $95\%$ percentile bootstrap interval over runs.")
w(r"\begin{table}[!ht]\centering\small\caption{Additional closed-loop runs: change of RMS speed deviation relative to C0 and input energy.}\label{tab:ctrl12}\begin{tabular}{lcccc}\toprule Network & A2 vs C0 & A0 vs C0 & A2 better in & input energy A2/C0\\\midrule")
rng=np.random.default_rng(0)
for c in ['case9','case39']:
    d=json.load(open(f'control_more_{c}.json'))
    def arr(ct,k): return np.array([r[k] for r in sorted([r for r in d if r['controller']==ct],key=lambda r:r['run'])])
    c0=arr('C0','w_rms'); a2=arr('A2','w_rms'); a0=arr('A0','w_rms'); n=len(c0)
    def red(x): return 100*(1-x/c0)
    def ci(x):
        v=red(x); bs=[rng.choice(v,n).mean() for _ in range(4000)]; return v.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    m2=ci(a2); m0=ci(a0); better=int((a2<c0).sum())
    e=arr('A2','u_energy').mean()/arr('C0','u_energy').mean()
    w(f"{c} & {m2[0]:.1f}\\% [{m2[1]:.1f}, {m2[2]:.1f}] & {m0[0]:.1f}\\% [{m0[1]:.1f}, {m0[2]:.1f}] & {better} of {n} & {e:.2f}\\\\")
w(r"\bottomrule\end{tabular}\end{table}")
open('supp_new_tables.tex','w').write("\n".join(O))
print("\n".join(O)[-1500:])
