"""Analysis of the direct per-line calibration experiment, exactly as fixed in PROTOCOL_DIRECT_CALIBRATION.md.
Usage: python3 extensions/contingency/analyze_direct.py   (writes results/direct_cal_summary.json, direct_cal_per_line.csv, direct_cal_table.tex)"""
import json, math, csv, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
R=Path(__file__).resolve().parent/'results'; ALPHA=0.10; NSUB=200; BOOT=2000
rng=np.random.default_rng(20261105)
st={}   # stored twin / nominal / static margins
for pre in ('c2','c2rep'):
    for s in (0,1):
        for r in json.load(open(R/f'{pre}_case39_full_{s}.json'))['rows']: st.setdefault(r['line'],{})[pre]=r
D={}
for f in sorted((R/'direct_cal').glob('line_*.json')): d=json.load(open(f)); D[d['line']]=d
lines=sorted(D); assert set(lines)==set(st), (set(st)-set(lines))
assert all(D[l]['valid'] for l in lines)
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return max(0,c-h),min(1,c+h)
# leakage / reproduction checks
chk=dict(max_abs_diff_test0_vs_stored_obs_max=0.0,cov_mismatch=0,seed_overlap=False,load_overlap=0)
for l in lines:
    a=D[l]; t0=np.array(a['test0']['score']); chk['max_abs_diff_test0_vs_stored_obs_max']=max(chk['max_abs_diff_test0_vs_stored_obs_max'],abs(t0.max()-st[l]['c2']['obs_max']))
    for m in ('nominal','Kstatic','twin'):
        if int((t0<=st[l]['c2'][m]['margin']).sum())!=round(st[l]['c2'][m]['cov']*20): chk['cov_mismatch']+=1
    if len({a['cal']['seed'],a['test0']['seed'],a['test1']['seed']})<3: chk['seed_overlap']=True
    if a['cal']['load0']==a['test0']['load0'] or a['cal']['load0']==a['test1']['load0']: chk['load_overlap']+=1
def qk(s,k): return np.sort(s)[k-1]
rules={}
for n in (10,20,40):
    rules[f'q{n}']=(n,math.ceil((n+1)*(1-ALPHA)))   # split-conformal quantile
    rules[f'max{n}']=(n,n)                            # max rule
rows=[]; per={}
for l in lines:
    cal=np.array(D[l]['cal']['score']); te=np.r_[D[l]['test0']['score'],D[l]['test1']['score']]
    m={'nominal':st[l]['c2']['nominal']['margin'],'Kstatic':st[l]['c2']['Kstatic']['margin'],'twin':st[l]['c2']['twin']['margin']}
    assert abs(st[l]['c2']['twin']['margin']-st[l]['c2rep']['twin']['margin'])<1e-12
    cov={};
    for k,v in m.items(): cov[k]=float((te<=v).mean())
    dr={}
    for name,(n,k) in rules.items():
        if k>n: continue
        sub=[rng.choice(40,n,replace=False) for _ in range(NSUB)]
        mg=np.array([qk(cal[s],k) for s in sub]); cv=np.array([(te<=x).mean() for x in mg])
        dr[name]=dict(margin_mean=float(mg.mean()),cov_mean=float(cv.mean()),cov_p10=float(np.quantile(cv,.1)),cov_min=float(cv.min()))
    # full-pool (n=40) reference
    per[l]=dict(line=l,test_med=float(np.median(te)),test_max=float(te.max()),cal_med=float(np.median(cal)),m=m,cov=cov,direct=dr,
                lo_hi={k:wilson(round(cov[k]*40),40) for k in cov},ratio_twin_q20=m['twin']/dr['q20']['margin_mean'])
    rows.append(per[l])
L=np.array(lines)
def boot_pool(vals):    # vals per line, mean over lines (equal 40 trials each); bootstrap over lines
    v=np.array(vals); b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(BOOT)]; return float(v.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
S={'checks':chk,'n_lines':len(lines)}
for key,get in (('twin',lambda p:p['cov']['twin']),('nominal',lambda p:p['cov']['nominal']),('Kstatic',lambda p:p['cov']['Kstatic'])):
    v=[get(p) for p in rows]; S[key]=dict(pooled_cov=boot_pool(v),below90=int(sum(x<0.9 for x in v)),below80=int(sum(x<0.8 for x in v)),lines_below80=[int(p['line']) for p in rows if get(p)<0.8])
for name in rules:
    if rules[name][1]>rules[name][0]: continue
    v=[p['direct'][name]['cov_mean'] for p in rows]; w=[p['direct'][name]['margin_mean'] for p in rows]
    S[name]=dict(pooled_cov=boot_pool(v),below90=int(sum(x<0.9 for x in v)),below80=int(sum(x<0.8 for x in v)),mean_p10_cov=float(np.mean([p['direct'][name]['cov_p10'] for p in rows])),
                 lines_below80=[int(p['line']) for p in rows if p['direct'][name]['cov_mean']<0.8],gm_margin_ratio_twin_over_direct=float(np.exp(np.mean([math.log(p['m']['twin']/p['direct'][name]['margin_mean']) for p in rows]))))
lr=np.array([math.log(p['m']['twin']/p['direct']['q20']['margin_mean']) for p in rows]); b=[np.exp(lr[rng.integers(0,len(lr),len(lr))].mean()) for _ in range(BOOT)]
S['paired_q20']=dict(gm_ratio=float(np.exp(lr.mean())),ci=[float(np.quantile(b,.025)),float(np.quantile(b,.975))],wilcoxon_logratio_p=float(wilcoxon(lr).pvalue),
   n_twin_wider=int((lr>0).sum()),n_twin_narrower=int((lr<0).sum()),
   cov_diff_mean=float(np.mean([p['cov']['twin']-p['direct']['q20']['cov_mean'] for p in rows])),
   wilcoxon_covdiff_p=float(wilcoxon([p['cov']['twin']-p['direct']['q20']['cov_mean'] for p in rows]).pvalue))
# margin relative to median test score
S['margin_over_median_test']=dict(twin=float(np.exp(np.mean([math.log(p['m']['twin']/p['test_med']) for p in rows]))),q20=float(np.exp(np.mean([math.log(p['direct']['q20']['margin_mean']/p['test_med']) for p in rows]))))
json.dump(dict(summary=S,per_line=rows),open(R/'direct_cal_summary.json','w'),indent=1)
with open(R/'direct_cal_per_line.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['line','test_median_err','test_max_err','twin_margin','twin_cov','twin_cov_ci','direct_q20_margin','direct_q20_cov_mean','direct_q20_cov_p10','direct_q40_cov_mean','nominal_cov','Kstatic_cov','ratio_twin_over_direct_q20'])
    for p in rows: w.writerow([p['line'],'%.4f'%p['test_med'],'%.4f'%p['test_max'],'%.4f'%p['m']['twin'],'%.3f'%p['cov']['twin'],'[%.2f,%.2f]'%tuple(p['lo_hi']['twin']),'%.4f'%p['direct']['q20']['margin_mean'],'%.3f'%p['direct']['q20']['cov_mean'],'%.3f'%p['direct']['q20']['cov_p10'],'%.3f'%p['direct']['q40']['cov_mean'],'%.3f'%p['cov']['nominal'],'%.3f'%p['cov']['Kstatic'],'%.2f'%p['ratio_twin_q20']])
print(json.dumps(S,indent=1))
# LaTeX table for the supplement (35 rows)
T=['% generated by extensions/contingency/analyze_direct.py','\\begin{longtable}{rcccccccc}','\\toprule',
   'line & median error & twin margin & twin cov. & direct margin & direct cov. & direct cov., 10th pct. & ratio twin/direct & nominal / static cov.\\\\','\\midrule','\\endhead']
for p in rows:
    T.append('%d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.2f & %.2f / %.2f\\\\'%(p['line'],p['test_med'],p['m']['twin'],p['cov']['twin'],p['direct']['q20']['margin_mean'],p['direct']['q20']['cov_mean'],p['direct']['q20']['cov_p10'],p['ratio_twin_q20'],p['cov']['nominal'],p['cov']['Kstatic']))
T+=['\\bottomrule','\\end{longtable}']; (R/'direct_cal_table.tex').write_text('\n'.join(T)+'\n')
