import sys,json; sys.path.insert(0,'code')
import numpy as np
from network_regime_map import *
out=[]
for nm in ('case9','case39'):
    c0=build_case(nm); th,de,_=equilibrium(c0); tr=make_training(c0,th,de,30,1001,1e-3); vl=make_training(c0,th,de,8,1002,1e-3)
    c=fit_net(c0,tr,vl,th)[0]; st=np.load(RES/f'coef_nominal_{nm}.npy'); d=float(np.max(np.abs(c-st))); s=float(np.max(np.abs(st)))
    out.append(f'{nm}: max abs diff refit vs stored coef_nominal = {d:.3e}; scale max|stored| = {s:.3f}; relative = {d/s:.3e}; supplementary diagnostic tolerance 1e-9 (not a protocol criterion) -> {"PASS_WITH_NUMERICAL_ROUNDING" if d<1e-9 else "CHECK"}')
out.append('Seed 9400 note: the H1 assertion drew matched random numbers from default_rng(9400) for M0/M1/M2 and compared them. No test trajectory was simulated and no test metric computed. This deviates from the literal wording "test seed not touched".')
open('results/regime_map/V_stage_disclosures.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
