"""Static topology diagnostics per single-line outage (lossless swing-algebraic model, fixed rotor angles at equilibrium).
For every non-bridge line l: leverage, post-outage regularity, Newton-Kantorovich certificate, exact change, and the stored twin coverage."""
import sys, json, numpy as np
R=(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))); sys.path.insert(0,R+'/code')
from network_validity_gate import nominal_fit
from network_benchmark import alg_res_jac, solve_theta
from network_experiments import args_of
import scipy.sparse as sp, scipy.sparse.csgraph as cg
def run(name):
    c,th,de=nominal_fit(name)[:3]; n=c['n']; nl=len(c['lb']); lf,lt,lb,gb,bg=args_of(c)
    g0,J0=alg_res_jac(th,de,c['Pd'],lf,lt,lb,gb,bg,True); print(name,'equil residual',np.abs(g0).max())
    J0=np.array(J0); res=json.load(open(R+f'/extensions/contingency/results/c2_{name}_full_0.json'))['rows']; cov={r['line']:r for r in res}
    out=[]
    # L_inf Lipschitz constant of the Jacobian in theta (infinity norms), valid for any topology that is a subgraph
    rowb=np.zeros(n)
    for j,k,b in zip(lf,lt,lb): rowb[j]+=b; rowb[k]+=b
    gen=np.zeros(n); gen[gb]+=bg
    L=2*np.max(2*rowb+gen)
    for l in range(nl):
        keep=np.arange(nl)!=l
        A=sp.coo_matrix((np.ones(nl-1),(lf[keep],lt[keep])),shape=(n,n))
        if cg.connected_components(A,directed=False)[0]!=1: continue
        j,k,b=lf[l],lt[l],lb[l]; a=np.zeros(n); a[j]=1; a[k]=-1; phi=th[j]-th[k]
        cl=b*np.cos(phi); ell=cl*a@np.linalg.solve(J0,a)         # leverage (outage regularity score)
        JT=J0-cl*np.outer(a,a); lamT=np.linalg.eigvalsh(JT)[0]
        gT=-(b*np.sin(phi))*a   # residual of post-outage network at nominal theta: g_T(th0)=g0 + b sin(phi) a -> sign via numeric check below
        gTn=alg_res_jac(th,de,c['Pd'],lf[keep],lt[keep],lb[keep],gb,bg,True)[0]
        step=np.linalg.solve(JT,gTn); eta=np.abs(step).max(); beta=np.abs(np.linalg.inv(JT)).sum(1).max(); h=beta*eta*L
        thT,nit=solve_theta(th,de,c['Pd'],lf[keep],lt[keep],lb[keep],gb,bg,1e-13,60); act=np.abs(thT-th).max()
        r=cov.get(l,{}); tw=r.get('twin',{}); 
        out.append(dict(line=l,lev=float(ell),lamT=float(lamT),lam0=float(np.linalg.eigvalsh(J0)[0]),eta=float(eta),h=float(h),exact=float(act),
                        bound2eta=float(2*eta) if h<=.5 else None,Kmax=r.get('K'),twin_cov=tw.get('cov'),margin=tw.get('margin'),obs_max=r.get('obs_max')))
    return out,L
if __name__=='__main__':
    for name in ('case9','case39'):
        o,L=run(name); json.dump(o,open(f'diag_topology_{name}.json','w'),indent=1)
        print(name,'L_inf',round(L,2),'n lines',len(o))
        print('line  lev   lamT   eta     h      exact  twin_cov obs_max/margin')
        for r in o: print('%3d %6.3f %6.3f %7.4f %7.3f %7.4f  %s  %s'%(r['line'],r['lev'],r['lamT'],r['eta'],r['h'],r['exact'],r['twin_cov'], (round(r['obs_max']/r['margin'],3) if r['margin'] else None)))
