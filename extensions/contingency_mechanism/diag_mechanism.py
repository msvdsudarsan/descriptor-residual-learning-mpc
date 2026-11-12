"""Per-outage mechanism diagnostics for the predictor-twin margin (case39, calibration seed 5001, original test seeds).
Reproduces, line by line, the margin m_T = max_i max_w ||e_nom + D_T||_inf and the held-out true scores (same seeds as run_contingency.py:
calibration 5001, test 9101+l+1000), and records for every non-islanding line:
  cal_scores (20), margin, test true/model/D scores (20), the per-trajectory 'gain' g = max_w(||e+D||_inf - ||D||_inf) on calibration and test,
  the component (line angle / rotor-to-bus angle) at which the margin and the largest test score are attained.
Read-only with respect to all earlier files. Output: results/mechanism_<case>.json   Usage (repo root): PYTHONPATH=code python3 extensions/contingency_mechanism/diag_mechanism.py case39"""
import sys, json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step
import scipy.sparse as sp, scipy.sparse.csgraph as cg
name=sys.argv[1]; LAMP=0.08; N=10; J=4; NTEST=20
c0,th_eq,de_eq,coef=nominal_fit(name); nl=len(c0['lf'])
def angs(th,de): return np.r_[th[c0['lf']]-th[c0['lt']], de-th[c0['gbus']]]
def run_pred(c,lg,k,Pk):
    de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
    for j in range(J+1):
        for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
    return de,th
def conn_ok(l):
    keep=np.arange(nl)!=l; A=sp.coo_matrix((np.ones(nl-1),(c0['lf'][keep],c0['lt'][keep])),shape=(c0['n'],c0['n'])); return cg.connected_components(A,directed=False)[0]==1
cal=make_test_runs(c0,th_eq,de_eq,20,5001,LAMP)
CW=[]   # nominal calibration windows: (e_nom, A0, k, i)
for i,(load,lg) in enumerate(cal):
    for k in range(0,len(lg['t'])-1-N,4):
        Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); d0,th0=run_pred(c0,lg,k,Pk)
        A0=angs(th0,d0); CW.append((angs(lg['th'][k+J+1],lg['de'][k+J+1])-A0,A0,k,i,Pk,lg))
eps=max(np.abs(w[0]).max() for w in CW)
out={'eps':float(eps),'lines':{}}
for l in range(nl):
    if not conn_ok(l): continue
    keep=np.arange(nl)!=l; cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    # calibration side: D_T on the nominal calibration windows
    s=np.zeros(20); g=np.zeros(20); Dc=np.zeros(20); comp=[None]*20
    for (e,A0,k,i,Pk,lg) in CW:
        dT,thT=run_pred(cp,lg,k,Pk); D=angs(thT,dT)-A0; v=np.abs(e+D)
        if v.max()>s[i]: s[i]=v.max(); comp[i]=int(v.argmax())
        g[i]=max(g[i],v.max()-np.abs(D).max()); Dc[i]=max(Dc[i],np.abs(D).max())
    m=float(s.max()); im=int(s.argmax())
    # test side (held-out plant WITH the outage, original seeds)
    runs=make_test_runs(cp,th_eq,de_eq,NTEST,9101+l+1000,LAMP); tt=np.zeros(NTEST); tm=np.zeros(NTEST); td=np.zeros(NTEST); tg=np.zeros(NTEST); tcomp=[None]*NTEST
    for i,(load,lg) in enumerate(runs):
        for k in range(0,len(lg['t'])-1-N,4):
            Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); d0,th0=run_pred(c0,lg,k,Pk); dT,thT=run_pred(cp,lg,k,Pk)
            At=angs(lg['th'][k+J+1],lg['de'][k+J+1]); A0=angs(th0,d0); AT=angs(thT,dT); eT=At-A0; em=At-AT; D=AT-A0
            if np.abs(eT).max()>tt[i]: tt[i]=np.abs(eT).max(); tcomp[i]=int(np.abs(eT).argmax())
            tm[i]=max(tm[i],np.abs(em).max()); td[i]=max(td[i],np.abs(D).max()); tg[i]=max(tg[i],np.abs(eT).max()-np.abs(D).max())
    out['lines'][l]=dict(margin=m,cal_scores=s.tolist(),cal_gain=g.tolist(),cal_Dmax=Dc.tolist(),cal_argmax_comp=comp,cal_margin_traj=im,
        test_true=tt.tolist(),test_model=tm.tolist(),test_D=td.tolist(),test_gain=tg.tolist(),test_argmax_comp=tcomp,cov=float((tt<=m).mean()))
    print(l,'margin %.4f cov %.2f  cal gain med %.4f | test gain med %.4f'%(m,(tt<=m).mean(),np.median(g),np.median(tg)),flush=True)
(ROOT/'extensions/contingency_mechanism/results').mkdir(exist_ok=True)
json.dump(out,open(ROOT/'extensions/contingency_mechanism/results'/f'mechanism_{name}.json','w'),indent=1)
