"""Cost of the predictor-twin margin: wall-clock time per twin run and per candidate contingency (calibration windows x twin runs),
versus the nominal-only epsilon and the static shift.  Writes results/cost_benchmark.json.  Usage (repo root): PYTHONPATH=code python3 extensions/contingency/cost_benchmark.py
Timings are machine dependent (2-core cloud container, single thread); only the scaling (linear in the number of candidates) is a claim."""
import sys, json, time, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step
LAMP=0.08; N=10; J=4; out={}
for name in ('case9','case39'):
    c0,th_eq,de_eq,coef=nominal_fit(name); nl=len(c0['lf'])
    def run_pred(c,lg,k,Pk):
        de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
        for j in range(J+1):
            for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
        return de,th
    cal=make_test_runs(c0,th_eq,de_eq,20,5001,LAMP)
    wins=[(load,lg,k) for load,lg in cal for k in range(0,len(lg['t'])-1-N,4)]
    run_pred(c0,wins[0][1],wins[0][2],c0['Pd'])  # warm-up (JIT)
    t=time.perf_counter()
    for load,lg,k in wins: run_pred(c0,lg,k,c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])))
    t_nom=(time.perf_counter()-t)/len(wins)
    keep=np.arange(nl)!=0; cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    run_pred(cp,wins[0][1],wins[0][2],c0['Pd'])
    t=time.perf_counter()
    for load,lg,k in wins: run_pred(cp,lg,k,c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])))
    t_tw=(time.perf_counter()-t)/len(wins)
    out[name]=dict(n_lines=nl,calibration_windows=len(wins),ms_per_nominal_prediction=1e3*t_nom,ms_per_twin_run=1e3*t_tw,
      seconds_per_candidate=t_tw*len(wins),seconds_all_single_line_candidates=t_tw*len(wins)*nl,
      note='margin for one candidate = one twin run per calibration window; cost is linear in the number of candidates; the nominal epsilon needs no twin run')
    print(name,json.dumps(out[name]))
json.dump(out,open(ROOT/'extensions/contingency/results/cost_benchmark.json','w'),indent=1)
