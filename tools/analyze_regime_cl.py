import csv,json,sys,collections,numpy as np
from scipy import stats
nm=sys.argv[1]; R='results/regime_map/'; r=list(csv.DictReader(open(R+f'{nm}_closed_loop_runs.csv'))); out=[]
d=collections.defaultdict(lambda:collections.defaultdict(dict))
for x in r: d[(x['regime'],x['authority'])][x['controller']][int(x['run'])]=x
f=lambda cs,c,k:np.array([float(cs[c][i][k]) for i in sorted(cs[c])])
w=csv.writer(open(R+f'{nm}_closed_loop_summary.csv','w')); w.writerow(['regime','authority','controller','w_rms','peak_deg','input_energy','saturation_fraction','viol_integral'])
for (g,a),cs in d.items():
    for c in cs: w.writerow([g,a,c]+[round(f(cs,c,k).mean(),5) for k in('w_rms','peak_deg','input_energy','saturation_fraction','viol_integral')])
w=csv.writer(open(R+f'{nm}_closed_loop_paired.csv','w')); w.writerow(['regime','authority','difference','n','mean','ci_lo','ci_hi','margin(2% mean P1* w_rms)','verdict','energy_ratio'])
for (g,a),cs in d.items():
    mg=0.02*f(cs,'P1*','w_rms').mean()
    for x,y in (('P2','P1*'),('P2','C0eq'),('P2','C0'),('P3','P2'),('P2','A0')):
        df=f(cs,x,'w_rms')-f(cs,y,'w_rms'); n=len(df); h=stats.t.ppf(.975,n-1)*df.std(ddof=1)/np.sqrt(n); lo,hi=df.mean()-h,df.mean()+h
        v='equivalent' if lo>-mg and hi<mg else 'first better' if hi<-mg else 'second better' if lo>mg else 'INCONCLUSIVE'
        w.writerow([g,a,f'{x}-{y}',n,round(df.mean(),5),round(lo,5),round(hi,5),round(mg,5),v,round(f(cs,x,'input_energy').mean()/f(cs,y,'input_energy').mean(),4)])
