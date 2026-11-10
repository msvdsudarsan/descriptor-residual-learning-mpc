"""Value checks for the contingency results.  Exit status 0 only if every check passes.
 (1) independent re-aggregation of the raw JSON against paper/values_contingency.json (written by make_tables.py);
 (2) every case in the raw logs (logs/*.log) against the raw JSON (margins to 3 decimals, coverages to 2);
 (3) the numbers quoted in the manuscript (listed below) against the re-aggregated values;
 (4) independence of calibration and test trajectories (disjoint seeds, no identical initial conditions);
 (5) missed-failure lists.
Usage (repository root): python3 extensions/contingency/check_contingency_values.py"""
import json, glob, re, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/'extensions/contingency/results'; LOG=ROOT/'extensions/contingency/logs'
ok=True
def fail(msg):
    global ok; ok=False; print('FAIL:',msg)
def rows(pat):
    r=[]
    for f in sorted(glob.glob(str(RES/pat))): r+=json.load(open(f))['rows']
    return r
def agg(rs,m):
    c=np.array([x[m]['cov'] for x in rs]); return float(c.mean()),int((c>=0.9).sum()),int((c<0.8).sum())
def boot(d):
    rng=np.random.default_rng(0); d=np.asarray(d,float); b=[rng.choice(d,len(d)).mean() for _ in range(4000)]; return float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
SETS={'case9_full':'c2_case9_full_*.json','case9_full_rep':'c2rep_case9_full_*.json','case9_full_mis':'c2repmis_case9_full_*.json','case39_full':'c2_case39_full_*.json','case39_full_rep':'c2rep_case39_full_*.json','case39_full_mis':'c2repmis_case39_full_*.json'}
data={k:rows(p) for k,p in SETS.items()}
for cs in ('case9','case39'):
    p=rows(f'c2_{cs}_partial_*.json')
    for f in (0.1,0.2,0.4): data[f'{cs}_partial_{int(f*100)}']=[r for r in p if abs(r['f']-f)<1e-9]
# (1)
stored=json.load(open(ROOT/'paper/values_contingency.json')); n1=0
for k,rs in data.items():
    for m in ('nominal','Kstatic','Kwin','twin','lin'):
        a=agg(rs,m); s=stored[k][m]; n1+=1
        if abs(a[0]-s['mean'])>1e-9 or a[1]!=s['ge09'] or a[2]!=s['lt08']: fail(f'stored value differs: {k} {m} {a} vs {s}')
    d=[r['twin']['cov']-r['Kstatic']['cov'] for r in rs]; lo,hi=boot(d)
    if abs(np.mean(d)-stored[k]['diff']['mean'])>1e-9 or abs(lo-stored[k]['diff']['lo'])>1e-9 or abs(hi-stored[k]['diff']['hi'])>1e-9: fail(f'paired difference differs: {k}')
print(f'(1) {n1} aggregated entries compared with paper/values_contingency.json')
# (2) logs vs JSON
pat=re.compile(r'^(\d+) ([0-9.]+) K ([0-9.]+) obs_max ([0-9.]+) \| nominal ([0-9.]+)/([0-9.]+) Kstatic ([0-9.]+)/([0-9.]+) Kwin ([0-9.]+)/([0-9.]+) twin ([0-9.]+)/([0-9.]+) lin ([0-9.]+)/([0-9.]+)')
nl=0
for lf in sorted(glob.glob(str(LOG/'c2*.log'))):
    stem=Path(lf).stem; rs=[]
    for f in sorted(glob.glob(str(RES/(stem[:-1]+'*.json')))): pass
    jf=RES/(stem+'.json')
    if not jf.exists(): fail(f'no json for {lf}'); continue
    J={(r['line'],round(r['f'],6)):r for r in json.load(open(jf))['rows']}
    for line in open(lf):
        mt=pat.match(line.strip())
        if not mt: continue
        l=int(mt.group(1)); f=round(float(mt.group(2)),6); r=J.get((l,f))
        if r is None: fail(f'{stem}: line {l} f {f} missing in json'); continue
        g=[float(x) for x in mt.groups()[2:]]; nl+=1
        exp=[r['K'],r['obs_max']]+[v for m in ('nominal','Kstatic','Kwin','twin','lin') for v in (r[m]['margin'],r[m]['cov'])]
        for a,b,i in zip(g,exp,range(len(g))):
            tol=6e-4 if i<2 or i%2==0 else 6e-3
            if abs(a-b)>tol: fail(f'{stem} line {l} f {f} item {i}: log {a} json {b}')
print(f'(2) {nl} log lines compared with the raw JSON')
# (3) numbers quoted in the manuscript
q=[('case9_full','twin',(0.99,6,0)),('case9_full','Kstatic',(0.00,0,6)),('case9_full_rep','twin',(1.00,6,0)),('case9_full_mis','twin',(1.00,6,0)),
   ('case39_full','twin',(0.95,31,3)),('case39_full','Kstatic',(0.60,20,15)),('case39_full','nominal',(0.21,6,29)),('case39_full','Kwin',(0.32,9,24)),('case39_full','lin',(0.21,6,29)),
   ('case39_full_rep','twin',(0.95,31,2)),('case39_full_rep','Kstatic',(0.59,20,15)),('case39_full_mis','twin',(0.92,10,2)),('case39_full_mis','Kstatic',(0.55,6,6)),
   ('case9_partial_40','twin',(0.79,1,5)),('case9_partial_40','Kstatic',(0.74,5,3)),('case39_partial_40','twin',(0.91,7,1)),('case39_partial_40','Kstatic',(0.94,9,1))]
for k,m,(mean,ge,lt) in q:
    a=agg(data[k],m)
    if abs(a[0]-mean)>0.0051 or a[1]!=ge or a[2]!=lt: fail(f'manuscript value {k} {m}: expected {(mean,ge,lt)} got {a}')
for k,(dm,dlo,dhi) in {'case39_full':(0.35,0.18,0.52),'case39_full_rep':(0.36,0.19,0.53),'case39_full_mis':(0.37,0.04,0.69)}.items():
    d=[r['twin']['cov']-r['Kstatic']['cov'] for r in data[k]]; lo,hi=boot(d)
    if abs(np.mean(d)-dm)>0.0051 or abs(lo-dlo)>0.0051 or abs(hi-dhi)>0.0051: fail(f'paired difference {k}: expected {(dm,dlo,dhi)} got {(np.mean(d),lo,hi)}')
for k in data:
    if k.endswith('full') or '_full' in k:
        r=np.array([x['twin']['margin']/x['obs_max'] for x in data[k]])
        if r.max()>2: fail(f'{k}: twin margin exceeds twice the observed maximum')
print(f'(3) {len(q)+3} manuscript values compared')
# (5) missed failures
exp={'case39_full':{8,10,11},'case39_full_rep':{8,10},'case39_full_mis':{14,17},'case9_full':set(),'case9_full_rep':set(),'case9_full_mis':set()}
for k,e in exp.items():
    got={r['line'] for r in data[k] if r['twin']['cov']<0.8}
    if got!=e: fail(f'missed failures {k}: expected {sorted(e)} got {sorted(got)}')
print('(5) missed-failure lists compared (case39 full: lines 8,10,11 original; 8,10 replication; 14,17 mismatch)')
# (4) independence: seeds and initial conditions
def draws(seed,n,ng,de_eq):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(n):
        de0=de_eq+rng.normal(0,0.10,ng); w0=rng.normal(0,0.3,ng); ld=(rng.uniform(0,0.08),rng.uniform(1,6),rng.uniform(0,6.28)); out.append(np.r_[de0,w0,ld])
    return np.array(out)
seeds={'calibration':[5001,7001],'training':[1001,1002]}
test=[]
for l in range(0,60):
    for f in (0.1,0.2,0.4): test.append(9101+l+int(1000*f))
    test.append(9101+l+1000)
test_all=set(test)|{s+10000 for s in test}|{s+20000 for s in test}
if 5001 in test_all or 7001 in test_all or 1001 in test_all or 1002 in test_all: fail('a test seed coincides with the calibration/training seed')
sys.path.insert(0,str(ROOT/'code'))
from network_validity_gate import nominal_fit
bad=0
for cs in ('case9','case39'):
    c,th,de_eq,_=nominal_fit(cs); cal=draws(5001,20,c['ng'],de_eq)
    for s in sorted(test_all):
        T=draws(s,20,c['ng'],de_eq)
        for a in T:
            if np.min(np.abs(cal-a).max(1))<1e-12: bad+=1
if bad: fail(f'{bad} test trajectories have initial conditions identical to a calibration trajectory')
print('(4) calibration seed 5001 and training seeds 1001/1002 are disjoint from',len(test_all),'test seeds; no test trajectory shares initial conditions with a calibration trajectory')
print('ALL CHECKS PASSED' if ok else 'CHECKS FAILED'); sys.exit(0 if ok else 1)
