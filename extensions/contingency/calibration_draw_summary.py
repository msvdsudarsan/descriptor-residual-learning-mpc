"""Compare a new calibration draw (files c2cal<seed>_case*_full_*.json) with the original calibration (seed 5001) on the same held-out test trajectories.
Writes results/calibration_draw_<seed>.csv and prints the summary used in the manuscript.  Usage (repo root): python3 extensions/contingency/calibration_draw_summary.py 7001"""
import sys, json, glob, csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/'extensions/contingency/results'; seed=sys.argv[1]
def rows(pat):
    r={}
    for f in sorted(glob.glob(str(RES/pat))):
        for x in json.load(open(f))['rows']: r[int(x['line'])]=x
    return r
out=[]
for case in ('case9','case39'):
    new=rows(f'c2cal{seed}_{case}_full_*.json'); old=rows(f'c2_{case}_full_*.json')
    if not new: continue
    ls=sorted(new); print(f'{case}: {len(ls)} outages, new calibration eps {json.load(open(sorted(glob.glob(str(RES/f"c2cal{seed}_{case}_full_*.json")))[0]))["eps"]:.4f}')
    for m in ('nominal','Kstatic','twin'):
        c=np.array([new[l][m]['cov'] for l in ls]); co=np.array([old[l][m]['cov'] for l in ls])
        print(f'  {m}: new mean {c.mean():.3f} (>=0.9: {(c>=0.9).sum()}, <0.8: {(c<0.8).sum()}) | original calibration on the same lines: mean {co.mean():.3f} (>=0.9: {(co>=0.9).sum()}, <0.8: {(co<0.8).sum()})')
    mr=np.array([new[l]['twin']['margin']/old[l]['twin']['margin'] for l in ls]); print(f'  twin margin new/original: min {mr.min():.3f} median {np.median(mr):.3f} max {mr.max():.3f}')
    print('  twin coverage below 0.8 (new calibration):',[(l,new[l]['twin']['cov']) for l in ls if new[l]['twin']['cov']<0.8],'| original:',[(l,old[l]['twin']['cov']) for l in ls if old[l]['twin']['cov']<0.8])
    for l in ls: out.append([case,l,old[l]['twin']['margin'],new[l]['twin']['margin'],old[l]['twin']['cov'],new[l]['twin']['cov'],old[l]['Kstatic']['cov'],new[l]['Kstatic']['cov'],old[l]['obs_max'],new[l]['obs_max']])
with open(RES/f'calibration_draw_{seed}.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['case','line','twin_margin_cal5001','twin_margin_new','twin_cov_cal5001','twin_cov_new','Kstatic_cov_cal5001','Kstatic_cov_new','obs_max_cal5001','obs_max_new']); w.writerows(out)
