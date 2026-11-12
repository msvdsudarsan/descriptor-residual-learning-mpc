"""D4 test: sample efficiency of parametric (damping+const), library (sparse, as in the paper) and hybrid (parametric first, sparse library on the remainder).
Prediction only, case9, one-second state RMSE relative to A0, in-family and saliency x1.5 (out-of-family), 3 training seeds per size."""
import sys, numpy as np
import pathlib; ROOT=pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from network_param_baseline import restricted
from network_baselines_loaddyn import stack, scenario_runs
from network_validity_gate import *
name=sys.argv[1] if len(sys.argv)>1 else 'case9'
c, th_eq, de_eq, coef_nom = nominal_fit(name); ng=c['ng']
val=make_training(c,th_eq,de_eq,8,1002,1e-3); Va=[weak_form_net(c,t,th_eq) for t in val]; THv,Yv=stack(Va)
nrun=8 if name=='case9' else 5
scen={'in-family':make_test_runs(c,th_eq,de_eq,nrun,9100,0.06,T=8.0),'saliency x1.5':scenario_runs(c,th_eq,de_eq,'saliency x1.5',nrun,9200)}
r0={k:np.sqrt(np.mean(np.vstack(predict_windows(c,r,'A0',coef_nom,stride=4)[0])**2)) for k,r in scen.items()}
def hybrid(TH,Y):
    cp=restricted(TH,Y,THv,Yv,ng,[6,11]); coef=cp.copy()
    for i in range(ng):
        y2=Y[:,i]-TH[:,i,:]@cp[i,:NF]; y2v=Yv[:,i]-THv[:,i,:]@cp[i,:NF]; cands=[]
        for rg in (1e-6,1e-4,1e-2):
            for tq in (0.01,0.03,0.1,0.3):
                cc=stlsq(TH[:,i,:],y2,ridge=rg,thresh=tq); e=np.sqrt(np.mean((y2v-THv[:,i,:]@cc)**2)); cands.append((e,int(np.sum(cc!=0)),cc))
        eb=min(x[0] for x in cands); e0=np.sqrt(np.mean(y2v**2))
        if eb<0.95*e0:                                    # accept the library correction only if it lowers the validation error by 5%
            cc=min([x for x in cands if x[0]<=1.05*eb],key=lambda x:(x[1],x[0]))[2]; coef[i,:NF]+=cc
    return coef
rows=[]
for ntr in (2,3,5,8,16,30):
    for sd in (1001,1101,1201):
        tr=make_training(c,th_eq,de_eq,ntr,sd,1e-3); TH,Y=stack([weak_form_net(c,t,th_eq) for t in tr])
        cf={'library':fit_net(c,tr,val,th_eq)[0],'parametric':restricted(TH,Y,THv,Yv,ng,[6,11]),'hybrid':hybrid(TH,Y)}
        for k,co in cf.items():
            for s,r in scen.items():
                e=np.vstack(predict_windows(c,r,'A2',co,stride=4)[0]); rows.append((ntr,sd,k,s,float(np.sqrt(np.mean(e**2))/r0[s])))
    print('done ntr',ntr,flush=True)
import collections
agg=collections.defaultdict(list)
for ntr,sd,k,s,v in rows: agg[(ntr,s,k)].append(v)
with open(f'.','w') as f:
    f.write('ntrain,scenario,method,mean_ratio_to_A0,min,max\n')
    for (ntr,s,k),v in sorted(agg.items()): f.write(f'{ntr},{s},{k},{np.mean(v):.4f},{np.min(v):.4f},{np.max(v):.4f}\n')
print(open(f'.').read())
