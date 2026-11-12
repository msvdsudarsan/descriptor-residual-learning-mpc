"""Bounded research round: dynamic correction of the contingency margin WITHOUT simulating the plant under the contingency.
Definitions (one set, used everywhere).  A(delta;P,T) = constrained angles (line angles, rotor-to-bus angles) at the algebraic solution Theta(delta;P,T).
Prediction error under contingency T*: e = A(d*;P*,T*) - A(dhat;P_k,T0)  =  Lam + K + R  with
  Lam = A(d*;P*,T*) - A(d*;P_k,T*)   (load change since the window start, true topology)
  K   = A(d*;P_k,T*) - A(d*;P_k,T0)   (static topology shift at the true rotor angles)
  R   = A(d*;P_k,T0) - A(dhat;P_k,T0) (rotor-state term, nominal topology)
and R = (model-approximation part, as seen in the nominal data) + (contingency-induced rotor response eta: d* - d*_nom).  [Claim 4 of the earlier report used Lam4 = Lam + K.]
Methods (all built from NOMINAL calibration windows only; thresholds fixed in advance: max over 20 trajectories of the trajectory-level max score):
 nominal      : eps
 Kstatic      : eps + K_static   (K_static at nominal equilibrium, one solve)
 Kwin         : e_nom(window) + K_T(delta_t,P_t)  (static shift at each nominal window, previous 'pred')
 twin         : e_nom(window) + [A(xhat_T;T) - A(xhat_0;T0)]  with xhat_T the PREDICTOR run under the post-contingency topology from the same window start (a predictor simulation, no plant simulation)
 lin          : e_nom(window) + (f/f0)[A(xhat_{T(f0)};T(f0)) - A(xhat_0;T0)], f0=0.02: first-order sensitivity in the line-parameter change, extrapolated to f
Test: 20 independent held-out trajectories of the plant WITH the contingency (seeds not used before); coverage = fraction of trajectories whose score <= margin.
Usage (from the repository root): PYTHONPATH=code python3 extensions/contingency/run_contingency.py case9|case39 partial|full [start stride]
Environment: SEEDOFF (0 original, 10000 replication, 20000 with MISMATCH), MISMATCH (fractional plant line-susceptance error, 0.05), CONT_OUT (output directory), CALSEED (seed of the 20 nominal calibration trajectories, default 5001; other values give files c2cal<seed>_*)."""
import sys, os, json, numpy as np
from pathlib import Path
ROOT=Path((__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..')))); sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step, solve_theta
import scipy.sparse as sp, scipy.sparse.csgraph as cg
name,mode=sys.argv[1],sys.argv[2]; start=int(sys.argv[3]) if len(sys.argv)>3 else 0; stride=int(sys.argv[4]) if len(sys.argv)>4 else 1
OUT=Path(os.environ.get('CONT_OUT',str(ROOT/'extensions/contingency/results'))); OUT.mkdir(parents=True,exist_ok=True); SEEDOFF=int(os.environ.get('SEEDOFF','0')); MIS=float(os.environ.get('MISMATCH','0')); LAMP=0.08; N=10; J=4; F0=0.02; NTEST=20
c0,th_eq,de_eq,coef=nominal_fit(name); nl=len(c0['lf'])
def angs(c,th,de): return np.r_[th[c0['lf']]-th[c0['lt']], de-th[c0['gbus']]]
def Theta(c,th0,de,Pd): return solve_theta(th0,de,Pd,*args_of(c),1e-13,60)[0]
def run_pred(c,lg,k,Pk):
    de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
    for j in range(J+1):
        for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
    return de,th
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; ctr=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [float(ctr-h),float(ctr+h)]
def variant(f,l,full):
    lb=c0['lb'].copy(); cp=dict(c0)
    if full:
        keep=np.arange(nl)!=l; cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    else: lb[l]*=(1-f); cp['lb']=lb
    return cp
def conn_ok(l):
    keep=np.arange(nl)!=l; A=sp.coo_matrix((np.ones(nl-1),(c0['lf'][keep],c0['lt'][keep])),shape=(c0['n'],c0['n'])); return cg.connected_components(A,directed=False)[0]==1
CALSEED=int(os.environ.get('CALSEED','5001')); cal=make_test_runs(c0,th_eq,de_eq,20,CALSEED,LAMP)
W=[]  # nominal windows
for load,lg in cal:
    tl=[]
    for k in range(0,len(lg['t'])-1-N,4):
        Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); de0,th0=run_pred(c0,lg,k,Pk); tt=lg['t'][k+J+1]
        e=angs(c0,lg['th'][k+J+1],lg['de'][k+J+1])-angs(c0,th0,de0); Pt=c0['Pd']*(1+load[0]*np.sin(load[1]*tt+load[2]))
        tl.append(dict(k=k,lg=lg,Pk=Pk,A0=angs(c0,th0,de0),e=e,dt=lg['de'][k+J+1],tht=lg['th'][k+J+1],Pt=Pt))
    W.append(tl)
eps=max(max(np.abs(w['e']).max() for w in tl) for tl in W)
th0e=Theta(c0,th_eq,de_eq,c0['Pd'])
def score(runs):
    sc=[]
    for load,lg in runs:
        best=0.0
        for k in range(0,len(lg['t'])-1-N,4):
            Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); de,th=run_pred(c0,lg,k,Pk)
            best=max(best,np.abs(angs(c0,lg['th'][k+J+1],lg['de'][k+J+1])-angs(c0,th,de)).max())
        sc.append(best)
    return np.array(sc)
cases=[]
if mode=='partial':
    lines=list(range(nl)) if name=='case9' else list(range(0,nl,5))
    for l in lines:
        for f in (0.1,0.2,0.4): cases.append((l,f,False))
else:
    for l in range(nl):
        if conn_ok(l): cases.append((l,1.0,True))
cases=cases[start::stride]; out=[]
for l,f,full in cases:
    try:
        cp=variant(f,l,full); cpe=variant(F0,l,False)
        Ks=np.abs(angs(c0,Theta(cp,th0e,de_eq,c0['Pd']),de_eq)-angs(c0,th0e,de_eq)).max()
        sc={m:[] for m in ('Kwin','twin','lin')}
        for tl in W:
            b={m:0.0 for m in sc}
            for w in tl:
                lg=w['lg']; Kt=angs(c0,Theta(cp,w['tht'],w['dt'],w['Pt']),w['dt'])-angs(c0,Theta(c0,w['tht'],w['dt'],w['Pt']),w['dt'])
                dT,thT=run_pred(cp,lg,w['k'],w['Pk']); DT=angs(c0,thT,dT)-w['A0']
                dE,thE=run_pred(cpe,lg,w['k'],w['Pk']); DL=(f/F0)*(angs(c0,thE,dE)-w['A0'])
                b['Kwin']=max(b['Kwin'],np.abs(w['e']+Kt).max()); b['twin']=max(b['twin'],np.abs(w['e']+DT).max()); b['lin']=max(b['lin'],np.abs(w['e']+DL).max())
            for m in sc: sc[m].append(b[m])
        marg=dict(nominal=eps,Kstatic=eps+Ks,Kwin=max(sc['Kwin']),twin=max(sc['twin']),lin=max(sc['lin']))
        cpl=dict(cp); cpl['lb']=cp['lb']*(1+MIS*np.random.default_rng(77+l).uniform(-1,1,len(cp['lb']))) if MIS>0 else cp['lb']
        seed=9101+SEEDOFF+l+int(1000*f)
        runs=make_test_runs(cpl,th_eq,de_eq,NTEST,seed,LAMP)
        if not all(np.isfinite(lg['de']).all() and np.abs(lg['de']).max()<20 for _,lg in runs): continue
        runs0=make_test_runs(c0,th_eq,de_eq,NTEST,seed,LAMP)     # same seed: same initial rotor states and loads, nominal topology
        modsc=[];cov_on=[];cov_win=[];ratio_on=[];obs=[]
        for (load,lg) in runs:
            bm=0.0;ok=True;nw=0;nc=0;rr=0.0;bo=0.0
            for k in range(0,len(lg['t'])-1-N,4):
                Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2]))
                de,th=run_pred(c0,lg,k,Pk); eo=angs(c0,lg['th'][k+J+1],lg['de'][k+J+1])-angs(c0,th,de)
                dT,thT=run_pred(cp,lg,k,Pk); DT=angs(c0,thT,dT)-angs(c0,th,de)
                so=np.abs(eo).max(); dn=np.abs(DT).max(); em=np.abs(eo-DT).max()
                bm=max(bm,em); bo=max(bo,so); nw+=1; nc+=int(so<=eps+dn); rr=max(rr,so/(eps+dn))
                ok=ok and (so<=eps+dn)
            modsc.append(bm);cov_on.append(ok);cov_win.append(nc/nw);ratio_on.append(rr);obs.append(bo)
        r=dict(line=l,f=f,full=full,K=float(Ks),obs=obs,modsc=modsc,cov_on=[bool(x) for x in cov_on],cov_win=cov_win,ratio_on=ratio_on,obs_max=float(max(obs)),obs_med=float(np.median(obs)))
        r['model_frac_le_eps']=float(np.mean(np.array(modsc)<=eps)); r['online_traj_cov']=float(np.mean(cov_on))
        out.append(r); print(l,'model<=eps %.2f online traj cov %.2f window cov %.3f max ratio %.3f'%(r['model_frac_le_eps'],r['online_traj_cov'],np.mean(cov_win),max(ratio_on)),flush=True); continue
        obs=np.array(obs); r=dict(line=l,f=f,full=full,K=float(Ks),obs=obs.tolist(),pred=pred,disc=disc,kmax=kmax,obs_max=float(obs.max()),obs_med=float(np.median(obs)))
        for m,v in marg.items():
            k=int((obs<=v).sum()); r[m]=dict(margin=float(v),cov=k/NTEST,ci=wilson(k,NTEST))
        out.append(r); print(l,f,'K %.3f obs_max %.3f | '%(Ks,obs.max())+' '.join('%s %.3f/%.2f'%(m,r[m]['margin'],r[m]['cov']) for m in marg),flush=True)
    except Exception as ex: print('fail',l,f,repr(ex),flush=True)
json.dump(dict(eps=float(eps),rows=out),open(OUT/('setB%s%s%s_%s_%s_%d.json'%('rep' if SEEDOFF else '','mis' if MIS>0 else '','cal%d'%CALSEED if CALSEED!=5001 else '',name,mode,start)),'w'),indent=1)
