"""Compare a fresh rerun (directory given) against the stored raw JSON of the same name.  Exit 0 only if every margin and coverage agrees to 1e-9."""
import sys, json, glob
from pathlib import Path
new=Path(sys.argv[1]); old=Path(__file__).resolve().parent/'results'; bad=0; n=0
for f in sorted(new.glob('c2*.json')):
    a=json.load(open(f))['rows']
    # the stored raw files of the large sets were produced in two interleaved parts (_0, _1); merge them by line index before comparing
    parts=[old/f.name]+([old/f.name.replace('_0.json','_1.json')] if f.name.endswith('_0.json') and (old/f.name.replace('_0.json','_1.json')).exists() else [])
    bd={}
    for p in parts:
        for r in json.load(open(p))['rows']: bd[(r['line'],r['f'])]=r
    # every stored row must be reproduced; the rerun may contain additional rows (the stored 5% mismatch set holds 12 of the 35 outages)
    a=[r for r in a if (r['line'],r['f']) in bd]; b=[bd[(r['line'],r['f'])] for r in a]
    if len(b)!=len(bd): print('FAIL stored rows not reproduced',f.name,len(b),len(bd)); bad+=1; continue
    for x,y in zip(a,b):
        for m in ('nominal','Kstatic','Kwin','twin','lin'):
            for q in x[m]:
                n+=1
                if isinstance(x[m][q],(int,float)) and abs(x[m][q]-y[m][q])>1e-9: print('FAIL',f.name,x.get('line'),m,q,x[m][q],y[m][q]); bad+=1
print(f'compared {n} numbers; mismatches {bad}'); sys.exit(1 if bad else 0)
