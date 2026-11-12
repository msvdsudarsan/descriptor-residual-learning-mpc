"""Sharper Lipschitz bound from the stochastic-matrix lemma, inf-norm of the Jacobian of the scaled reduced field (delta, T*omega), T=1 s."""
import sys, numpy as np
import pathlib; ROOT=pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code')); sys.path.insert(0,str(ROOT/'tests'))
from test_props_independent import *
def run(name,N,seed):
    c, th_eq, de_eq, coef = nominal_fit(name); rng=np.random.default_rng(seed); ng=c['ng']; cnt=0; h=1e-6
    Lnew=max(1.0, np.max(2*c['bg']/c['M']+c['Dphys']/c['M']))   # inf-norm bound; delta-row: 1 ; omega-row: 2 bg/M (stiffness) + D/M
    k=max(np.bincount(c['gbus'])); _,B,E=mats(c,de_eq,th_eq); lamB=np.linalg.eigvalsh(B)[0]
    kap=k*c['bg'].max()/(np.cos(G)*lamB); Lold=1.0+np.max(c['Dphys']/c['M'])+np.max(c['bg']/c['M'])*(1+kap)
    worst_inf=0; worst_2=0
    while cnt<N:
        de=de_eq+rng.normal(0,0.15,ng); th,_=solve_theta(th_eq,de,c['Pd'],*args_of(c),1e-13,40)
        if not inside(c,de,th): continue
        cnt+=1
        def F(x):
            d,w=x[:ng],x[ng:]; t,_=solve_theta(th,d,c['Pd'],*args_of(c),1e-14,40)
            return np.r_[w, swing_acc(d,w,np.zeros(ng),t,c['gbus'],c['bg'],c['M'],c['Dphys'],c['Pm'],False,c['kappa'],c['fric'])]
        x0=np.r_[de,rng.normal(0,0.3,ng)]
        JF=np.column_stack([(F(x0+h*np.eye(2*ng)[:,i])-F(x0-h*np.eye(2*ng)[:,i]))/(2*h) for i in range(2*ng)])
        worst_inf=max(worst_inf,np.abs(JF).sum(1).max()); worst_2=max(worst_2,np.linalg.norm(JF,2))
    print(name,'old L_f(2-norm)=%.0f'%Lold,'| observed ||JF||_2 max=%.1f'%worst_2,'| new inf-norm bound=%.1f'%Lnew,'| observed ||JF||_inf max=%.1f'%worst_inf,'| ratio new/observed=%.2f'%(Lnew/worst_inf))
run('case9',40,3); run('case39',25,4)
