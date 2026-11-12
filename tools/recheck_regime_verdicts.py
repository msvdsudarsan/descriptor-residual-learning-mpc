"""Independent recomputation of every regime-map verdict from the raw run-level CSVs (unrounded), compared with the stored paired CSVs."""
import csv,collections,numpy as np,json
from scipy import stats
R='results/regime_map/'; out=[]; bad=0; tally=collections.Counter()
def verdict(lo,hi,m): return 'equivalent' if lo>-m and hi<m else 'first better' if hi<-m else 'second better' if lo>m else 'INCONCLUSIVE'
for nm in ('case9','case39'):
    r=list(csv.DictReader(open(R+f'{nm}_closed_loop_runs.csv'))); d=collections.defaultdict(lambda:collections.defaultdict(dict))
    for x in r: d[(x['regime'],x['authority'])][x['controller']][int(x['run'])]=x
    f=lambda cs,c,k:np.array([float(cs[c][i][k]) for i in sorted(cs[c])])
    st={(x['regime'],x['authority'],x['difference']):x['verdict'] for x in csv.DictReader(open(R+f'{nm}_closed_loop_paired.csv'))}
    for (g,a),cs in d.items():
        m=0.02*f(cs,'P1*','w_rms').mean()
        for x,y in (('P2','P1*'),('P2','C0eq'),('P2','C0'),('P3','P2'),('P2','A0')):
            df=f(cs,x,'w_rms')-f(cs,y,'w_rms'); n=len(df); h=stats.t.ppf(.975,n-1)*df.std(ddof=1)/np.sqrt(n); lo,hi=df.mean()-h,df.mean()+h; v=verdict(lo,hi,m)
            ok=(v==st[(g,a,f'{x}-{y}')]); bad+=not ok; out.append((nm,'CL',g,a,f'{x}-{y}',n,df.mean(),lo,hi,m,v,st[(g,a,f'{x}-{y}')],ok)); tally[(nm,'CL',f'{x}-{y}',v)]+=1
    r=list(csv.DictReader(open(R+f'{nm}_prediction_runs.csv'))); sel=json.load(open(R+f'{nm}_prediction_meta.json'))['selection']; dd=collections.defaultdict(dict)
    for x in r: dd[(x['regime'],x['setting'])].setdefault(int(x['run']),{})[x['model']]=np.sqrt(float(x['sumsq'])/float(x['n']))
    st={(x['regime'],x['setting'],x['difference']):x['verdict(margin 0.02)'] for x in csv.DictReader(open(R+f'{nm}_prediction_paired.csv'))}
    for (g,s),runs in dd.items():
        rat=lambda mm:np.array([runs[i][mm]/runs[i]['A0'] for i in sorted(runs)])
        for x,y in (('P2',sel[g+s]['pick']),('P3','P2')):
            df=rat(x)-rat(y); n=len(df); h=stats.t.ppf(.975,n-1)*df.std(ddof=1)/np.sqrt(n); lo,hi=df.mean()-h,df.mean()+h; v=verdict(lo,hi,0.02)
            k=(g,s,f'{x}-{y}' if y!='P1c' else 'P2-P1c'); ok=(v==st[k]); bad+=not ok; out.append((nm,'PRED',g,s,f'{x}-{y}',n,df.mean(),lo,hi,0.02,v,st[k],ok)); tally[(nm,'PRED',f'{x}-{y}',v)]+=1
w=csv.writer(open(R+'verdict_recheck.csv','w')); w.writerow(['case','stage','regime','authority_or_setting','difference','n','mean','ci_lo','ci_hi','margin','recomputed_verdict','stored_verdict','match']); w.writerows([[*o[:6],*(f'{v:.8g}' for v in o[6:10]),*o[10:]] for o in out])
print('comparisons',len(out),'mismatches vs stored',bad)
for k,v in sorted(tally.items()): print(k,v)
print('M1 x1 rows:'); [print(o[0],o[4],f'CI[{o[7]:.3e},{o[8]:.3e}] margin {o[9]:.4f}',o[10]) for o in out if o[1]=='CL' and o[2]=='M1' and o[3]=='1' and o[4] in('P2-C0','P2-A0')]
