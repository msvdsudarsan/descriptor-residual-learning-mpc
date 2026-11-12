"""Energy-norm Newton-Kantorovich certificate for the post-outage algebraic solution (fixed rotor angles).
M = J_T(theta0) (post-outage Jacobian at the pre-outage angles), ||x||_M = sqrt(x' M x), eta = ||M^{-1} g_T(theta0)||_M,
K = sqrt(r_max)/cos(gamma0), r_max = max over branches (lines of T and generator branches) of a' M^{-1} a, gamma0 = max |angle| over branches at theta0.
Claim: ||M^{-1}(J(x)-J(y))||_M <= K ||x-y||_M, hence h=K*eta<=1/2 gives existence, uniqueness in a ball and ||theta*-theta0||_M <= (1-sqrt(1-2h))/K."""
import sys, json, numpy as np
R=(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))); sys.path.insert(0,R+'/code')
from network_validity_gate import nominal_fit
from network_benchmark import alg_res_jac, solve_theta
from network_experiments import args_of
import scipy.sparse as sp, scipy.sparse.csgraph as cg
def certify(c,th,de,l,Pd=None):
    n=c['n']; lf,lt,lb,gb,bg=args_of(c); Pd=c['Pd'] if Pd is None else Pd; nl=len(lb)
    keep=np.arange(nl)!=l; f,t,b=lf[keep],lt[keep],lb[keep]
    g,M=alg_res_jac(th,de,Pd,f,t,b,gb,bg,True); M=np.array(M)
    Mi=np.linalg.inv(M); eta=np.sqrt(g@Mi@g)
    ra=[]; cg_=[]
    for j,k in zip(f,t):
        a=np.zeros(n); a[j]=1; a[k]=-1; ra.append(a@Mi@a); cg_.append(abs(th[j]-th[k]))
    for i,j in enumerate(gb): ra.append(Mi[j,j]); cg_.append(abs(de[i]-th[j]))
    gam=max(cg_); rmax=max(ra); K=np.sqrt(rmax)/np.cos(gam); h=K*eta
    rho=(1-np.sqrt(1-2*h))/K if h<=.5 else None
    return dict(eta=float(eta),K=float(K),h=float(h),gamma0=float(gam),rho=None if rho is None else float(rho),Mi=Mi)
def run(name):
    c,th,de=nominal_fit(name)[:3]; n=c['n']; lf,lt,lb,gb,bg=args_of(c); nl=len(lb)
    cov={}
    for s in (0,1):
        try:
            for r in json.load(open(R+f'/extensions/contingency/results/c2_{name}_full_{s}.json'))['rows']: cov[r['line']]=r
        except FileNotFoundError: pass
    out=[]
    for l in range(nl):
        keep=np.arange(nl)!=l
        A=sp.coo_matrix((np.ones(nl-1),(lf[keep],lt[keep])),shape=(n,n))
        if cg.connected_components(A,directed=False)[0]!=1: continue
        z=certify(c,th,de,l); Mi=z.pop('Mi')
        thT=solve_theta(th,de,c['Pd'],lf[keep],lt[keep],lb[keep],gb,bg,1e-13,60)[0]
        M=np.linalg.inv(Mi); d=thT-th; exact_E=float(np.sqrt(d@M@d))
        r=cov.get(l,{}); tw=r.get('twin',{})
        z.update(line=l,exact_E=exact_E,exact_inf=float(np.abs(d).max()),cert=bool(z['h']<=.5),twin_cov=tw.get('cov'),ratio=(r['obs_max']/tw['margin'] if tw else None))
        out.append(z)
    return out
if __name__=='__main__':
    for name in ('case9','case39'):
        o=run(name); json.dump(o,open(f'kantorovich_{name}.json','w'),indent=1)
        print(name,'certified',sum(r['cert'] for r in o),'of',len(o))
        for r in o: print('%3d eta %.4f K %.3f h %.3f cert %s rho %s exactE %.4f cov %s ratio %s'%(r['line'],r['eta'],r['K'],r['h'],r['cert'],None if r['rho'] is None else round(r['rho'],4),r['exact_E'],r['twin_cov'],None if r['ratio'] is None else round(r['ratio'],3)))
