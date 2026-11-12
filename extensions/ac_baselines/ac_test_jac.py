"""Verification: power-flow equilibrium, analytic Jacobian dg/dz and dg/ddelta vs central finite differences."""
import numpy as np, sys
from ac_model import *
for name in ('case9','case39'):
  for (npw,nqw) in ((0,0),(2,2)):
    c = build_ac(name, npw=npw, nqw=nqw); z0, de0, pfres = nominal_point(c); net = net_of(c)
    g,_ = ac_res_jac(z0, de0, 1.0, net, False)
    n=c['n']; pe = c['bgE']*z0[n+c['gbus']]*np.sin(de0-z0[c['gbus']])
    rng=np.random.default_rng(0); z = z0 + rng.normal(0,0.03,2*n); de = de0+rng.normal(0,0.1,c['ng'])
    g,J = ac_res_jac(z, de, 1.1, net, True); Jfd=np.zeros_like(J); h=1e-6
    for k in range(2*n):
        e=np.zeros(2*n); e[k]=h
        Jfd[:,k]=(ac_res_jac(z+e,de,1.1,net,False)[0]-ac_res_jac(z-e,de,1.1,net,False)[0])/(2*h)
    D = dg_ddelta(z,de,net); Dfd=np.zeros_like(D)
    for i in range(c['ng']):
        e=np.zeros(c['ng']); e[i]=h
        Dfd[:,i]=(ac_res_jac(z,de+e,1.1,net,False)[0]-ac_res_jac(z,de-e,1.1,net,False)[0])/(2*h)
    print(name,'np,nq',npw,nqw,'PF resid',pfres,'|g(eq)|max',np.abs(g).max() if False else np.abs(ac_res_jac(z0,de0,1.0,net,False)[0]).max(),
          '|Pm-Pe|max',np.abs(c['Pm']-pe).max(),'Vmin/max',z0[n:].min(),z0[n:].max(),'E range',c['E'].min(),c['E'].max(),
          '| max|J-Jfd|',np.abs(J-Jfd).max(),'rel',np.abs(J-Jfd).max()/np.abs(J).max(),'| max|dg/dd - fd|',np.abs(D-Dfd).max(),
          'sigma_min(J) at eq',np.linalg.svd(ac_res_jac(z0,de0,1.0,net,True)[1],compute_uv=False)[-1],'losses pu',(c['Pm'].sum()-c['Pd'].sum()), 'Pm sum', c['Pm'].sum())
