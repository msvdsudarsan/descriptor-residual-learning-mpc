import sys,time
sys.path.insert(0,'code')
from network_validity_gate import *
for nm in ('case9','case39'):
    c0,th,de,coef=nominal_fit(nm)
    t=time.time(); runs=make_shift_runs(c0,th,de,None,2,1,0.08,True,nm); print(nm,'gen2',time.time()-t)
    load,lg,pm=runs[0]; rng=np.random.default_rng(1)
    t=time.time(); lam=regularity(c0,th,de)[0]; s=gate_series(c0,coef,pm,lg,1,1,0,lam,rng)[3]; print('gate_series',time.time()-t, len(lg['t']))
