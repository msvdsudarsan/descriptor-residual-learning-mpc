"""Diagnostic of Assumption (X) and trace of D_T for chosen full outages (same seeds as the original run, 20 held-out trajectories).
For every test window: e_T (true error), e_model = A(x*_T;T)-A(xhat_T;T), D_T = A(xhat_T;T)-A(xhat_0;T0); checks e_T = e_model + D_T (identity),
and compares score distributions: nominal calibration scores of e_nom, test scores of e_model,T, and of the margin summand ||e_nom + D_T|| vs the true ||e_T||.
Also records per-trajectory true scores and the timing of one twin run (cost).  Usage: python3 diag_X.py case9|case39 l1,l2,..."""
import sys, os, json, time, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step
name=sys.argv[1]; lines=[int(x) for x in sys.argv[2].split(',')]; LAMP=0.08; N=10; J=4; NTEST=20
c0,th_eq,de_eq,coef=nominal_fit(name); nl=len(c0['lf'])
def angs(th,de): return np.r_[th[c0['lf']]-th[c0['lt']], de-th[c0['gbus']]]
def run_pred(c,lg,k,Pk):
    de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
    for j in range(J+1):
        for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
    return de,th
cal=make_test_runs(c0,th_eq,de_eq,20,5001,LAMP); enom=[]
for load,lg in cal:
    b=0.0
    for k in range(0,len(lg['t'])-1-N,4):
        Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); de0,th0=run_pred(c0,lg,k,Pk)
        b=max(b,np.abs(angs(lg['th'][k+J+1],lg['de'][k+J+1])-angs(th0,de0)).max())
    enom.append(b)
eps=max(enom); out={}
for l in lines:
    keep=np.arange(nl)!=l; cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    runs=make_test_runs(cp,th_eq,de_eq,NTEST,9101+l+1000,LAMP)
    tr=dict(true=[],model=[],D=[],ident=0.0); tt=[]
    for load,lg in runs:
        bt=bm=bd=0.0
        for k in range(0,len(lg['t'])-1-N,4):
            Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2]))
            d0,th0=run_pred(c0,lg,k,Pk); t0=time.perf_counter(); dT,thT=run_pred(cp,lg,k,Pk); tt.append(time.perf_counter()-t0)
            At=angs(lg['th'][k+J+1],lg['de'][k+J+1]); A0=angs(th0,d0); AT=angs(thT,dT)
            eT=At-A0; em=At-AT; D=AT-A0; tr['ident']=max(tr['ident'],np.abs(eT-(em+D)).max())
            bt=max(bt,np.abs(eT).max()); bm=max(bm,np.abs(em).max()); bd=max(bd,np.abs(D).max())
        tr['true'].append(bt); tr['model'].append(bm); tr['D'].append(bd)
    out[l]=dict(eps=float(eps),ident=float(tr['ident']),true_scores=[float(x) for x in tr['true']],model_scores=[float(x) for x in tr['model']],D_scores=[float(x) for x in tr['D']],twin_run_seconds_mean=float(np.mean(tt)))
    m=np.array(tr['model']); print(l,'ident %.1e'%tr['ident'],'eps %.4f | e_model score: median %.4f max %.4f (frac<=eps %.2f) | D max %.4f | true score max %.4f | twin run %.1f ms'%(eps,np.median(m),m.max(),(m<=eps).mean(),max(tr['D']),max(tr['true']),1e3*np.mean(tt)),flush=True)
json.dump(out,open(ROOT/'extensions/contingency/results'/f'diagX_{name}_{sys.argv[2].replace(",","_")}.json','w'),indent=1)
