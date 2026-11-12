import csv,sys,collections,json,numpy as np
from scipy import stats
nm=sys.argv[1]; R='results/regime_map/'; r=list(csv.DictReader(open(R+f'{nm}_prediction_runs.csv'))); pick=json.load(open(R+f'{nm}_prediction_meta.json'))['selection']
d=collections.defaultdict(dict)
for x in r: d[(x['regime'],x['setting'])].setdefault(int(x['run']),{})[x['model']]=np.sqrt(float(x['sumsq'])/float(x['n']))
rows=[];M=0.02
for (k,s),runs in d.items():
    ps=pick[k+s]['pick']; rat=lambda m:np.array([runs[i][m]/runs[i]['A0'] for i in sorted(runs)])
    for a,b in dict.fromkeys((('P2',ps),('P2','P1c'),('P3','P2'))):
        diff=rat(a)-rat(b); n=len(diff); mu=diff.mean(); h=stats.t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n); lo,hi=mu-h,mu+h
        v='equivalent' if (lo>-M and hi<M) else ('first better' if hi<-M else ('second better' if lo>M else 'INCONCLUSIVE'))
        rows.append((nm,k,s,f'{a}-{b}',n,round(mu,4),round(lo,4),round(hi,4),v))
w=csv.writer(open(R+f'{nm}_prediction_paired.csv','w')); w.writerow(['case','regime','setting','difference','n','mean','ci_lo','ci_hi','verdict(margin 0.02)']); w.writerows(rows)
for x in rows: print(*x)
