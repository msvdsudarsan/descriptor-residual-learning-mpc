"""Additional independent closed-loop realizations (new seed 7001) for the control claim; C0, A0, A2 (analytic gradient). Same setup as network_analytic_gradient.closed_loop."""
import sys, json, numpy as np
sys.path.insert(0,(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))+'/code'))
from network_analytic_gradient import *
name=sys.argv[1]; nreal=int(sys.argv[2])
cs,th_eq,de_eq,coef=stress_setup(name); eps_s=RES_STRESS_ENV(name); rel_ref=de_eq-np.sum(cs['M']*de_eq)/np.sum(cs['M'])
Tcl=6.0 if name=='case9' else 5.0
th0w,_=solve_theta(th_eq,de_eq,cs['Pd'],*args_of(cs),1e-12,30)
for nm,an in (('A0',False),('A2',True)): make_net_mpc(cs,nm,coef,rel_ref,eps_s,analytic=an)(0.0,de_eq,np.zeros(cs['ng']),th0w,np.zeros(cs['ng']),cs['Pd'])
rng=np.random.default_rng(7001); rows=[]
for i in range(nreal):
    load=(rng.uniform(0.07,0.11),rng.uniform(2.5,4.0),rng.uniform(0,6.28)); de0=de_eq+rng.normal(0,0.03,cs['ng']); w0=rng.normal(0,0.1,cs['ng'])
    th0,_=solve_theta(th_eq,de0,cs['Pd'],*args_of(cs),1e-12,30)
    for nm in ('C0','A0','A2'):
        ctrl=static_ctrl(cs) if nm=='C0' else make_net_mpc(cs,nm,coef,rel_ref,eps_s,analytic=(nm=='A2'))
        m=net_metrics(cs,simulate_plant(cs,th0,de0,w0,ctrl,Tcl,load)); rows.append(dict(run=i,controller=nm,**{k:float(v) for k,v in m.items()}))
    print(name,i,{r['controller']:round(r['w_rms'],4) for r in rows[-3:]},flush=True)
    json.dump(rows,open(f'control_more_{name}.json','w'))
