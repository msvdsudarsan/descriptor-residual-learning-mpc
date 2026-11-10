"""Check: d theta/d delta = J^{-1} E C is entrywise >=0 with unit row sums (so its inf-norm is exactly 1) inside the angle limits."""
import sys, numpy as np
import pathlib; ROOT=pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code')); sys.path.insert(0,str(ROOT/'tests'))
from test_props_independent import *
def run(name,N,seed):
    c, th_eq, de_eq, coef = nominal_fit(name); rng=np.random.default_rng(seed); ng=c['ng']; cnt=0
    worst_row=0; minent=1; mx_inf=0; tries=0; kmax=0; rmax=0
    while cnt<N and tries<20000:
        tries+=1; sd=rng.choice([0.05,0.15,0.3])
        de=de_eq+rng.normal(0,sd,ng); th,_=solve_theta(th_eq,de,c['Pd']*rng.uniform(0.7,1.4),*args_of(c),1e-13,60)
        if not inside(c,de,th): continue
        cnt+=1; J,_,E=mats(c,de,th); C=np.diag(c['bg']*np.cos(de-th[c['gbus']])); S=np.linalg.solve(J,E@C)
        worst_row=max(worst_row,np.abs(S.sum(1)-1).max()); minent=min(minent,S.min()); mx_inf=max(mx_inf,np.abs(S).sum(1).max())
        # line-angle sensitivity to delta: row of (e_j - e_k) S
        kmax=max(kmax,max(np.abs(S[c['lf'][l]]-S[c['lt'][l]]).sum() for l in range(len(c['lf']))))
        # rotor-to-bus angle delta_i - theta_{gbus(i)}: row e_i - S[gbus(i)]
        rmax=max(rmax,max(np.abs(np.eye(ng)[i]-S[c['gbus'][i]]).sum() for i in range(ng)))
    print(name,'states',cnt,'max|rowsum-1|',worst_row,'min entry',minent,'max inf-norm',mx_inf,'max line-angle gain (l1 of row diff)',kmax,'max rotor-to-bus gain (l1)',rmax,'half:',kmax/2,rmax/2)
run('case9',60,1); run('case39',40,2)
