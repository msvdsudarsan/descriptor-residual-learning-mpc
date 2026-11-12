"""Three-bus worked example for the descriptor swing model (lossless, unit voltages).
Buses 1,2 carry generators (b_i=1/x'd, x'd=0.3), bus 3 is a load bus. Lines (1,3),(2,3),(1,2).
All numbers printed here are produced by this script."""
import numpy as np
np.set_printoptions(precision=5, suppress=True)
bg=np.array([1/0.3,1/0.3]); sig=[0,1]
lines=[(0,2,10.0),(1,2,10.0),(0,1,5.0)]       # (j,k,b_l)
Pm=np.array([0.6,0.4]); Pd=np.array([0.,0.,1.0])
def g(theta,delta,bl):
    r=-Pd.copy()
    for i,s in enumerate(sig): r[s]+=bg[i]*np.sin(delta[i]-theta[s])
    for (j,k,b),bb in zip(lines,bl):
        f=bb*np.sin(theta[j]-theta[k]); r[j]-=f; r[k]+=f
    return r
def Jac(theta,delta,bl):
    J=np.zeros((3,3))
    for i,s in enumerate(sig): J[s,s]+=bg[i]*np.cos(delta[i]-theta[s])
    for (j,k,b),bb in zip(lines,bl):
        c=bb*np.cos(theta[j]-theta[k]); J[j,j]+=c;J[k,k]+=c;J[j,k]-=c;J[k,j]-=c
    return J
def solve(delta,bl,th0=None,tol=1e-14):
    th=np.zeros(3) if th0 is None else th0.copy(); it=0
    while np.linalg.norm(g(th,delta,bl),np.inf)>tol and it<50:
        th=th+np.linalg.solve(Jac(th,delta,bl),g(th,delta,bl)); it+=1
    return th,it
bl0=np.array([l[2] for l in lines])
# equilibrium: find delta with Pm = bg sin(delta-theta_s), g=0
d=np.array([0.3,0.3])
for _ in range(50):
    th,_=solve(d,bl0)
    res=bg*np.sin(d-th[sig])-Pm
    d=d-res/ (bg*np.cos(d-th[sig]))*0.5
th,_=solve(d,bl0)
print("equilibrium delta",d,"theta",th,"swing residual",np.abs(bg*np.sin(d-th[sig])-Pm).max())
J=Jac(th,d,bl0); print("J=\n",J,"\neig(J)",np.linalg.eigvalsh(J))
B=np.zeros((3,3))
for (j,k,b) in lines:
    a=np.zeros(3);a[j]=1;a[k]=-1;B+=b*np.outer(a,a)
for i,s in enumerate(sig): B[s,s]+=bg[i]
gam=max(max(abs(th[j]-th[k]) for j,k,_ in lines),max(abs(d-th[sig])))
print("gamma",gam,"lambda_min(B)",np.linalg.eigvalsh(B)[0],"bound cos(gamma)*lmin(B)",np.cos(gam)*np.linalg.eigvalsh(B)[0],"lmin(J)",np.linalg.eigvalsh(J)[0])
G=np.zeros((3,2))
for i,s in enumerate(sig): G[s,i]=bg[i]*np.cos(d[i]-th[s])
S=np.linalg.solve(J,G); print("S=dtheta/ddelta=\n",S,"\nrow sums",S.sum(1),"min entry",S.min())
# Prop 5 check: rotor-angle error -> algebraic-angle error
err=np.array([0.05,-0.02]); dh=d+err
th_h,it=solve(dh,bl0,th); print("angle error",th_h-th,"span of rotor error",err.max()-err.min(),"inf-norm of algebraic error",np.abs(th_h-th).max(),"newton iters",it)
# Newton projection from a stale algebraic guess
th_bad=th+np.array([0.08,-0.05,0.1]); print("residual before",np.abs(g(th_bad,dh,bl0)).max())
t=th_bad.copy()
for k in range(1,5):
    t=t+np.linalg.solve(Jac(t,dh,bl0),g(t,dh,bl0)); print(" Newton iter",k,"residual",np.abs(g(t,dh,bl0)).max())
# outage of line 0 (1,3): homotopy identity with delta fixed
l=0; j,k,b=lines[l]
def th_s(s):
    bl=bl0.copy(); bl[l]=(1-s)*b; return solve(d,bl,th)[0]
th_out=th_s(1.0)
print("post-outage theta",th_out,"change",th_out-th)
ss,w=np.polynomial.legendre.leggauss(40); ss=(ss+1)/2; w=w/2
tot=np.zeros(3)
for s,wi in zip(ss,w):
    ts=th_s(s); bl=bl0.copy(); bl[l]=(1-s)*b
    ej=np.zeros(3);ej[j]=1;ej[k]-=1
    tot+=wi*np.linalg.solve(Jac(ts,d,bl),ej)*b*np.sin(ts[j]-ts[k])
print("integral",tot,"max deviation",np.abs(tot-(th_out-th)).max())
print("one-step LODF-like first-order change",np.linalg.solve(J,(np.eye(3)[j]-np.eye(3)[k]))*b*np.sin(th[j]-th[k]),"(first-order, not exact)")
