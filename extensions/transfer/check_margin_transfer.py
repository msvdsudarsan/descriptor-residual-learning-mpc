"""Experiment E1: error of every algebraic angle quantity (line angles, rotor-to-bus angles) of the projected predictor A2 against
span(rotor-angle error) = max_i - min_i of (delta_pred - delta_true).  Theory (Lemma 1): the error caused by the rotor-angle error alone is
<= span; the rest is the load term (predictor holds Pd frozen).  Plant = true model; predictor = A2 with fitted residual."""
import sys, numpy as np
import pathlib; ROOT=pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import *
from network_experiments import make_test_runs, TS, NSUB, MODES, args_of
def run(name, nrun, seed, lamp_hi=0.08, N=10, stride=4):
    c, th_eq, de_eq, coef = nominal_fit(name); runs = make_test_runs(c, th_eq, de_eq, nrun, seed, lamp_hi)
    mode, use_res = MODES['A2']; ng=c['ng']; lf,lt,gb=c['lf'],c['lt'],c['gbus']
    rows=[]; outside=0; tot=0
    for load, lg in runs:
        K=len(lg['t'])-1
        for k in range(0,K-N,stride):
            Pd_k=c['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2]))
            de,w,th=lg['de'][k].copy(),lg['w'][k].copy(),lg['th'][k].copy()
            for j in range(N):
                u=lg['u'][k+j]
                for _ in range(NSUB):
                    de,w,th=pred_step(de,w,th,u,TS/NSUB,Pd_k,mode,use_res,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
                de_t=lg['de'][k+j+1]; th_t=lg['th'][k+j+1]
                dd=de-de_t; span=dd.max()-dd.min()
                # theta with true rotor angles but frozen load (isolates the load term)
                th_a,_=solve_theta(th_t,de_t,Pd_k,*args_of(c),1e-13,60)
                th_p,_=solve_theta(th,de,Pd_k,*args_of(c),1e-13,60)           # predictor's algebraic state, re-solved exactly
                def angs(d,t): return np.r_[t[lf]-t[lt], d-t[gb]]
                tot_err=np.abs(angs(de,th)-angs(de_t,th_t)).max()
                rot_err=np.abs(angs(de,th_p)-angs(de_t,th_a)).max()           # error due to rotor-angle error only
                load_err=np.abs(angs(de_t,th_a)-angs(de_t,th_t)).max()
                ok=all(np.all(np.abs(t[lf]-t[lt])<=np.radians(30)) for t in (th_p,th_a))
                outside+= (not ok); tot+=1
                rows.append((j+1,span,rot_err,load_err,tot_err,ok))
    R=np.array(rows,dtype=float); inreg=R[:,5]==1
    viol=(R[:,2]>R[:,1]*(1+1e-9)+1e-9)&inreg
    tri=(R[:,4]>R[:,1]+R[:,3]+1e-9)&inreg
    print(name,'samples',len(R),'outside 30deg region (excluded from the check):',int((~inreg).sum()))
    print('  Lemma check, rotor-angle part: violations of err<=span:',int(viol.sum()),'; max err/span = %.3f'%np.max(R[inreg,2]/np.maximum(R[inreg,1],1e-15)))
    print('  total err <= span + load term: violations',int(tri.sum()),'; median err/(span+load) = %.3f'%np.median(R[inreg,4]/(R[inreg,1]+R[inreg,3]+1e-15)))
    print('  median span %.4g  median total angle error %.4g  median load term %.4g (rad)'%(np.median(R[:,1]),np.median(R[:,4]),np.median(R[:,3])))
    for j in (1,5,10):
        m=R[:,0]==j; print('  step %d: 95%% quantile span %.4g, total err %.4g, load term %.4g'%(j,*np.quantile(R[m,1],0.95).reshape(1),np.quantile(R[m,4],0.95),np.quantile(R[m,3],0.95)))
    return R
if __name__=='__main__':
    run('case9',8,2101); run('case39',5,2102)
