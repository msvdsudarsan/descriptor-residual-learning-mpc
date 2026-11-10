import json,glob,numpy as np,sys
from scipy import stats
files=sorted(glob.glob('results_setA/setA*case39*.json')); print(files)
runs=[]  # each: dict line-> row
for f in files:
    d=json.load(open(f)); runs.append(dict(f=f,eps=d['eps'],rows={r['line']:r for r in d['rows']}))
lines=sorted(set.intersection(*[set(r['rows']) for r in runs]))
print('draws',len(runs),'lines',len(lines))
def wil(k,n,z=1.96):
    p=k/n; dd=1+z*z/n; c=(p+z*z/(2*n))/dd; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd; return c-h,c+h
tab=[]
for l in lines:
    k=sum(int((np.array(r['rows'][l]['obs'])<=r['rows'][l]['twin']['margin']).sum()) for r in runs); n=20*len(runs)
    cov_each=[r['rows'][l]['twin']['cov'] for r in runs]
    ratio=np.mean([r['rows'][l]['obs_max']/r['rows'][l]['twin']['margin'] for r in runs])
    dm=np.mean([max(r['rows'][l]['disc'])/r['rows'][l]['twin']['margin'] for r in runs])
    tab.append(dict(line=l,cov=k/n,lo=wil(k,n)[0],hi=wil(k,n)[1],cov_each=cov_each,ratio=ratio,disc_over_margin=dm))
tot_k=sum(t['cov']*20*len(runs) for t in tab); tot_n=20*len(runs)*len(tab)
M=np.array([[r['rows'][l]['twin']['cov'] for l in lines] for r in runs]); rng=np.random.default_rng(1)
bs=[M[np.ix_(rng.integers(0,len(runs),len(runs)),rng.integers(0,len(lines),len(lines)))].mean() for _ in range(5000)]
print('pooled twin coverage %.3f (n=%d); draw means'%(tot_k/tot_n,tot_n),[round(x,3) for x in M.mean(1)],'cluster bootstrap over draws and lines',np.percentile(bs,[2.5,97.5]))
print('line cov lo hi each ratio disc/margin')
for t in tab: print('%3d %.3f [%.2f,%.2f] %s ratio %.3f disc/m %.3f %s'%(t['line'],t['cov'],t['lo'],t['hi'],[round(x,2) for x in t['cov_each']],t['ratio'],t['disc_over_margin'],'<-- low' if t['cov']<0.9 else ''))
print('lines with pooled cov<0.9:',[t['line'] for t in tab if t['cov']<0.9],' with upper CI<0.95:',[t['line'] for t in tab if t['hi']<0.95])
print('fraction of lines with mean cov>=0.9: %.2f'%np.mean([t['cov']>=0.9 for t in tab]))
json.dump(tab,open('expA_summary.json','w'),indent=1)
# per-line features
try:
    kd={r['line']:r for r in json.load(open('kantorovich_case39.json'))}
    X=[];Y=[]
    for t in tab:
        z=kd[t['line']]; X.append([z['h'],z['exact_E'],z['eta']]); Y.append(t['ratio'])
    X=np.array(X);Y=np.array(Y)
    for i,nm in enumerate(['h','exact_E','eta']): print('Spearman ratio vs',nm,stats.spearmanr(X[:,i],Y))
except Exception as e: print(e)
