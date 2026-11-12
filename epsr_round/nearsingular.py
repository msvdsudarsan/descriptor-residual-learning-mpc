"""Synthetic stress of the outage certificate near an island. A degree-2 load bus b without generator is connected by lines l1, l2.
Outage of l1 with l2 weakened to eps*b_l2 (and the load of bus b scaled by eps so that the angles stay moderate): as eps->0 the outage leaves bus b almost isolated and the post-outage Jacobian approaches singularity.
Reported per eps: leverage, lambda_min(M), eta, h, certificate radius, and the true post-outage change from Newton (converged or not). Nothing else in the network is changed."""
import sys, json, numpy as np
R=(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))); sys.path.insert(0,R+'/code'); sys.path.insert(0,(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from network_experiments import args_of, build_case, equilibrium
from network_benchmark import alg_res_jac, solve_theta
from kantorovich_energy import certify
out={}
for name in ('case9','case39'):
    c0=build_case(name); n=c0['n']; lf,lt,lb=c0['lf'],c0['lt'],c0['lb']; gen=set(c0['gbus'].tolist())
    deg=np.zeros(n,int)
    for j,k in zip(lf,lt): deg[j]+=1; deg[k]+=1
    cands=[b for b in range(n) if deg[b]==2 and b not in gen and c0['Pd'][b]>0]
    print(name,'candidate degree-2 load buses',cands)
    rows=[]
    for b in cands[:2]:
        ls=[i for i in range(len(lb)) if lf[i]==b or lt[i]==b]; l1,l2=ls
        for eps in (1,0.5,0.2,0.1,0.05,0.02,0.01,0.005):
            c=dict(c0); c['lb']=lb.copy(); c['lb'][l2]*=eps; c['Pd']=c0['Pd'].copy(); c['Pd'][b]*=eps
            try: th,de,_=equilibrium(c)
            except Exception as e: rows.append(dict(bus=int(b),eps=eps,fail=repr(e))); continue
            z=certify(c,th,de,l1); Mi=z.pop('Mi'); M=np.linalg.inv(Mi); lam=np.linalg.eigvalsh(M)[0]
            keep=np.arange(len(lb))!=l1
            thT,nit=solve_theta(th,de,c['Pd'],lf[keep],lt[keep],c['lb'][keep],c['gbus'],c['bg'],1e-13,60)
            gres=np.abs(alg_res_jac(thT,de,c['Pd'],lf[keep],lt[keep],c['lb'][keep],c['gbus'],c['bg'],True)[0]).max()
            d=thT-th; rows.append(dict(bus=int(b),l1=int(l1),l2=int(l2),eps=eps,lam_min_M=float(lam),eta=z['eta'],K=z['K'],h=z['h'],cert=bool(z['h']<=.5),rho=z['rho'],exact_E=float(np.sqrt(d@M@d)),converged=bool(gres<1e-9),newton_it=int(nit)))
            print(name,'bus',b,'eps',eps,'lam %.3g eta %.3f K %.3f h %.3f cert %s exactE %.3f conv %s'%(lam,z['eta'],z['K'],z['h'],z['h']<=.5,np.sqrt(d@M@d),gres<1e-9))
    out[name]=rows
json.dump(out,open('nearsingular.json','w'),indent=1)
