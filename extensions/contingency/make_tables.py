"""Regenerate the contingency tables of the extension from the raw per-contingency JSON files in results/.
Writes: results/summary_per_contingency.csv, results/summary_by_set.csv, results/tables_by_set.tex, paper/values_contingency.json (values quoted in the manuscript).
Usage (repository root): python3 extensions/contingency/make_tables.py"""
import json, glob, csv, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/'extensions/contingency/results'
METHODS=('nominal','Kstatic','Kwin','twin','lin')
def load(pat):
    rows=[]
    for f in sorted(glob.glob(str(RES/pat))):
        for r in json.load(open(f))['rows']: r['_file']=Path(f).name; rows.append(r)
    return rows
SETS=[]   # (key, label, rows)
for cs in ('case9','case39'):
    p=load(f'c2_{cs}_partial_*.json')
    for f in (0.1,0.2,0.4): SETS.append((f'{cs}_partial_{int(f*100)}',f'{cs} line weakened by {int(f*100)}%',[r for r in p if abs(r['f']-f)<1e-9]))
    SETS.append((f'{cs}_full',f'{cs} full outage',load(f'c2_{cs}_full_*.json')))
    SETS.append((f'{cs}_full_rep',f'{cs} full outage, replication (new seeds)',load(f'c2rep_{cs}_full_*.json')))
    SETS.append((f'{cs}_full_mis',f'{cs} full outage, 5% plant line-parameter mismatch',load(f'c2repmis_{cs}_full_*.json')))
def boot(d,seed=0):
    rng=np.random.default_rng(seed); d=np.asarray(d,float); bs=[rng.choice(d,len(d)).mean() for _ in range(4000)]
    return float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
vals={}; lines=[]
with open(RES/'summary_per_contingency.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['set','line','fraction_weakened','full_outage','K_static','obs_max_score']+[f'{m}_{x}' for m in METHODS for x in ('margin','coverage','ci_lo','ci_hi')]+['source_file'])
    for key,lab,rows in SETS:
        for r in rows: w.writerow([key,r['line'],r['f'],int(r['full']),f"{r['K']:.6g}",f"{r['obs_max']:.6g}"]+[v for m in METHODS for v in (f"{r[m]['margin']:.6g}",r[m]['cov'],f"{r[m]['ci'][0]:.4f}",f"{r[m]['ci'][1]:.4f}")]+[r['_file']])
with open(RES/'summary_by_set.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['set','label','cases']+[f'{m}_{x}' for m in METHODS for x in ('mean_cov','n_cov_ge_0.9','n_cov_lt_0.8','median_margin_over_obsmax','n_margin_gt_2x_obsmax')]+['twin_minus_Kstatic_mean','twin_minus_Kstatic_ci_lo','twin_minus_Kstatic_ci_hi'])
    for key,lab,rows in SETS:
        if not rows: continue
        row=[key,lab,len(rows)]; v={'n':len(rows)}
        for m in METHODS:
            cov=np.array([r[m]['cov'] for r in rows]); ratio=np.array([r[m]['margin']/r['obs_max'] for r in rows])
            row+= [f'{cov.mean():.4f}',int((cov>=0.9).sum()),int((cov<0.8).sum()),f'{np.median(ratio):.3f}',int((ratio>2).sum())]
            v[m]=dict(mean=float(cov.mean()),ge09=int((cov>=0.9).sum()),lt08=int((cov<0.8).sum()),med_ratio=float(np.median(ratio)),over2=int((ratio>2).sum()))
        d=np.array([r['twin']['cov']-r['Kstatic']['cov'] for r in rows]); lo,hi=boot(d)
        row+=[f'{d.mean():.4f}',f'{lo:.4f}',f'{hi:.4f}']; v['diff']=dict(mean=float(d.mean()),lo=lo,hi=hi); vals[key]=v; w.writerow(row)
# missed failures list
miss={}
for key,lab,rows in SETS:
    miss[key]=[dict(line=r['line'],f=r['f'],twin_cov=r['twin']['cov'],margin_over_obsmax=r['twin']['margin']/r['obs_max']) for r in rows if r['twin']['cov']<0.8]
vals['twin_missed']=miss
(ROOT/'paper').mkdir(exist_ok=True); json.dump(vals,open(ROOT/'paper/values_contingency.json','w'),indent=1)
# LaTeX table (main text): full outages and partial 40%
def cell(v,m): x=v[m]; return f"{x['mean']:.2f} / {x['ge09']} / {x['lt08']}"
tex=['\\begin{tabular}{@{}lrccccc@{}}','\\toprule','set & cases & nominal & $\\varepsilon{+}K_{\\rm static}$ & $K_T$ windows & twin & first-order\\\\','\\midrule']
for key,lab,rows in SETS:
    if not rows: continue
    v=vals[key]; nm=lab.replace('%','\\%').replace('case9','case9').replace('case39','case39')
    tex.append(f"{nm} & {v['n']} & "+' & '.join(cell(v,m) for m in METHODS)+'\\\\')
tex+=['\\bottomrule','\\end{tabular}']
open(RES/'tables_by_set.tex','w').write('\n'.join(tex)+'\n')
print('wrote', RES/'summary_per_contingency.csv', RES/'summary_by_set.csv', RES/'tables_by_set.tex', ROOT/'paper/values_contingency.json')
