"""Numerical verification of (a) the twin-discrepancy bound  ||D_T - K_T(dhat0)||_inf <= span(dhat_T - dhat_0)  and
(b) the energy-norm outage certificate applied at each window state:  ||K_T(dhat0)||_inf <= sqrt(r_maxT) * rho(h).  case39 / case9, all non-bridge single-line outages."""
import sys, json, numpy as np
R=(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))); sys.path.insert(0,R+'/code'); sys.path.insert(0,(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step, solve_theta
from kantorovich_energy import certify
import scipy.sparse as sp, scipy.sparse.csgraph as cg
name=sys.argv[1]; NTRAJ=int(sys.argv[2]); STRIDE=int(sys.argv[3]); SEED=int(sys.argv[4]) if len(sys.argv)>4 else 5001
c0,th_eq,de_eq,coef=nominal_fit(name); nl=len(c0['lf']); N=10; J=4
def angs(th,de): return np.r_[th[c0['lf']]-th[c0['lt']], de-th[c0['gbus']]]
def run_pred(c,lg,k,Pk):
    de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
    for j in range(J+1):
        for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
    return de,th
cal=make_test_runs(c0,th_eq,de_eq,NTRAJ,SEED,0.08)
res=[]
for l in range(nl):
    keep=np.arange(nl)!=l
    A=sp.coo_matrix((np.ones(nl-1),(c0['lf'][keep],c0['lt'][keep])),shape=(c0['n'],c0['n']))
    if cg.connected_components(A,directed=False)[0]!=1: continue
    cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    mx=dict(line=l,mincos=9.0,lhsD=0.0,rhsD=0.0,viol=0,KT=0.0,cert=0.0,viol_cert=0,hmax=0.0,n=0)
    for load,lg in cal:
        for k in range(0,len(lg['t'])-1-N,STRIDE):
            Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2]))
            d0,t0=run_pred(c0,lg,k,Pk); dT,tT=run_pred(cp,lg,k,Pk)
            D=angs(tT,dT)-angs(t0,d0)
            thT0=solve_theta(t0,d0,Pk,*args_of(cp),1e-13,60)[0]       # post-outage algebraic solution at predicted nominal rotor angles
            KT=angs(thT0,d0)-angs(t0,d0)
            # hypothesis check of Prop 5/7: cos of every branch angle of T stays > 0 on the segment dhat_0 -> dhat_T
            thp=t0
            for sv in (0.0,0.25,0.5,0.75,1.0):
                ds=d0+sv*(dT-d0); thp=solve_theta(thp,ds,Pk,*args_of(cp),1e-13,60)[0]
                ang=np.r_[thp[cp['lf']]-thp[cp['lt']], ds-thp[c0['gbus']]]; mx['mincos']=min(mx['mincos'],float(np.cos(ang).min()))
            sp_=np.ptp(dT-d0); lhs=np.abs(D-KT).max()
            z=certify(c0,t0,d0,l,Pd=Pk); Mi=z.pop('Mi')
            # line/rotor-bus angle output bound: sqrt(max_k a_k' M^-1 a_k) * rho
            n=c0['n']; ra=[]
            for j,kk in zip(cp['lf'],cp['lt']):
                a=np.zeros(n); a[j]=1; a[kk]=-1; ra.append(a@Mi@a)
            ra+= [Mi[j,j] for j in c0['gbus']]
            cert=np.sqrt(max(ra))*z['rho'] if z['rho'] is not None else np.inf
            # NB: the removed line itself is also an output of A; bound it with a' M^-1 a for that line
            a=np.zeros(n); a[c0['lf'][l]]=1; a[c0['lt'][l]]=-1; cert=max(cert,np.sqrt(a@Mi@a)*z['rho']) if z['rho'] is not None else np.inf
            mx['n']+=1; mx['lhsD']=max(mx['lhsD'],lhs); mx['rhsD']=max(mx['rhsD'],sp_); mx['viol']+= int(lhs>sp_+1e-12)
            mx['KT']=max(mx['KT'],np.abs(KT).max()); mx['cert']=max(mx['cert'],cert); mx['viol_cert']+=int(np.abs(KT).max()>cert+1e-12); mx['hmax']=max(mx['hmax'],z['h'])
    res.append(mx); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in mx.items()},flush=True)
json.dump(res,open(f'verify_twin_bound_{name}_seed{SEED}.json','w'),indent=1)
print('min cos over all segments',min(r['mincos'] for r in res))
print('violations of twin bound',sum(r['viol'] for r in res),'of',sum(r['n'] for r in res),'; violations of certificate',sum(r['viol_cert'] for r in res))
