"""Plant-free screening statistic: static topology shift evaluated at the POST-outage equilibrium versus at the nominal equilibrium.
K_nom(l)=||A(d_eq;T)-A(d_eq;T0)||_inf (what the calibration windows see), K_post(l)=||A(d_T;T)-A(d_T;T0)||_inf with d_T the post-outage rotor equilibrium.
Needs only the reduced model (no outage plant data).  Output: results/screen_static_case39.csv.  Usage (repo root): PYTHONPATH=code python3 extensions/contingency_mechanism/screen_static.py"""
import sys, json, csv, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
from network_benchmark import equilibrium, solve_theta
from network_experiments import args_of
import scipy.sparse as sp, scipy.sparse.csgraph as cg
c0,th_eq,de_eq,coef=nominal_fit('case39'); nl=len(c0['lf']); n=c0['n']
def A(c,th,de,lf=c0['lf'],lt=c0['lt']): return np.r_[th[lf]-th[lt], de-th[c['gbus']]]
mech=json.load(open(ROOT/'extensions/contingency_mechanism/results/mechanism_case39.json')); eps=mech['eps']
rows=[]
for l in range(nl):
    keep=np.arange(nl)!=l
    Asp=sp.coo_matrix((np.ones(nl-1),(c0['lf'][keep],c0['lt'][keep])),shape=(n,n))
    if cg.connected_components(Asp,directed=False)[0]!=1: continue
    cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    thT,deT,res=equilibrium(cp,guess=np.r_[th_eq,de_eq])
    th0T=solve_theta(thT,deT,c0['Pd'],*args_of(c0),1e-13,60)[0]                     # nominal topology at the post-outage rotor angles
    th0e=solve_theta(th_eq,de_eq,c0['Pd'],*args_of(c0),1e-13,60)[0]
    thTe=solve_theta(th_eq,de_eq,c0['Pd'],*args_of(cp),1e-13,60)[0]                  # post-outage topology at the nominal rotor angles
    Knom=np.abs(A(c0,thTe,de_eq)-A(c0,th0e,de_eq)).max(); Kpost=np.abs(A(c0,thT,deT)-A(c0,th0T,deT)).max()
    mm=mech['lines'][str(l)]['margin']
    rows.append(dict(line=l,K_nom=Knom,K_post=Kpost,margin=mm,slack_pf=(mm-Kpost)/eps,dK=(Kpost-Knom)/eps,eq_res=res,rot_shift=np.abs(deT-de_eq).max()))
out=ROOT/'extensions/contingency_mechanism/results/screen_static_case39.csv'
with open(out,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('wrote',out,len(rows))
