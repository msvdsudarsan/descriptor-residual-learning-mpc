"""Direct per-line post-contingency calibration experiment (protocol: PROTOCOL_DIRECT_CALIBRATION.md).
Own implementation of a comparable procedure; not the code of any cited paper.
Generates, for each non-islanding case39 outage, the trajectory-level scores of the nominal-topology predictor on
  cal  : 40 new calibration trajectories (seed 40001+line)
  test0: the 20 original held-out trajectories (seed 9101+line+1000)
  test1: the 20 replication trajectories (seed 9101+10000+line+1000)
Usage (repository root): PYTHONPATH=code python3 extensions/contingency/direct_percontingency.py <start> <stride>"""
import sys, os, json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step
import scipy.sparse as sp, scipy.sparse.csgraph as cg
start=int(sys.argv[1]) if len(sys.argv)>1 else 0; stride=int(sys.argv[2]) if len(sys.argv)>2 else 1
OUT=Path(os.environ.get('DIRECT_OUT',str(ROOT/'extensions/contingency/results/direct_cal'))); OUT.mkdir(parents=True,exist_ok=True)
LAMP=0.08; N=10; J=4; NCAL=40; NTEST=20
c0,th_eq,de_eq,coef=nominal_fit('case39'); nl=len(c0['lf'])
def angs(th,de): return np.r_[th[c0['lf']]-th[c0['lt']], de-th[c0['gbus']]]
def run_pred(lg,k,Pk):   # identical to run_contingency.run_pred with the nominal model c0
    de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
    for j in range(J+1):
        for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c0),c0['M'],c0['Dphys'],c0['Pm'],c0['kappa'],c0['fric'])
    return de,th
def score(runs):          # identical to run_contingency.score
    sc=[]
    for load,lg in runs:
        best=0.0
        for k in range(0,len(lg['t'])-1-N,4):
            Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); de,th=run_pred(lg,k,Pk)
            best=max(best,np.abs(angs(lg['th'][k+J+1],lg['de'][k+J+1])-angs(th,de)).max())
        sc.append(best)
    return np.array(sc)
def conn_ok(l):
    keep=np.arange(nl)!=l; A=sp.coo_matrix((np.ones(nl-1),(c0['lf'][keep],c0['lt'][keep])),shape=(c0['n'],c0['n'])); return cg.connected_components(A,directed=False)[0]==1
lines=[l for l in range(nl) if conn_ok(l)][start::stride]
for l in lines:
    f=OUT/('line_%02d.json'%l)
    if f.exists(): continue
    keep=np.arange(nl)!=l; cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    sd={'cal':40001+l,'test0':9101+0+l+1000,'test1':9101+10000+l+1000}
    res={}
    ok=True
    for key,n in (('cal',NCAL),('test0',NTEST),('test1',NTEST)):
        runs=make_test_runs(cp,th_eq,de_eq,n,sd[key],LAMP)
        if not all(np.isfinite(lg['de']).all() and np.abs(lg['de']).max()<20 for _,lg in runs): ok=False
        res[key]=dict(seed=sd[key],score=score(runs).tolist(),load0=list(runs[0][0]))
    res['line']=l; res['valid']=ok
    json.dump(res,open(f,'w'),indent=1); print(l,ok,'cal med %.3f test0 med %.3f'%(np.median(res['cal']['score']),np.median(res['test0']['score'])),flush=True)
