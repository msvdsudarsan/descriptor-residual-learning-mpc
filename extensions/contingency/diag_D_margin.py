import sys, json, numpy as np
sys.path.insert(0,'code')
from network_validity_gate import nominal_fit
from network_experiments import make_test_runs, TS, NSUB, args_of
from network_benchmark import pred_step
name='case39'; LAMP=0.08; N=10; J=4
c0,th_eq,de_eq,coef=nominal_fit(name); nl=len(c0['lf'])
def angs(th,de): return np.r_[th[c0['lf']]-th[c0['lt']], de-th[c0['gbus']]]
def run_pred(c,lg,k,Pk):
    de,w,th=lg['de'][k],lg['w'][k],lg['th'][k]
    for j in range(J+1):
        for _ in range(NSUB): de,w,th=pred_step(de,w,th,lg['u'][k+j],TS/NSUB,Pk,1,1,coef,*args_of(c),c['M'],c['Dphys'],c['Pm'],c['kappa'],c['fric'])
    return de,th
cal=make_test_runs(c0,th_eq,de_eq,20,5001,LAMP)
D=json.load(open('diagX_case39_8_10_11_0_12_24_34_44_5_14.json'))
for l in (8,10,11,5,14,12):
    keep=np.arange(nl)!=l; cp=dict(c0); cp['lf'],cp['lt'],cp['lb']=c0['lf'][keep],c0['lt'][keep],c0['lb'][keep]
    cs=[];nsc=[];jt=[]
    for load,lg in cal:
        b=0.0;bn=0.0;bj=0.0
        for k in range(0,len(lg['t'])-1-N,4):
            Pk=c0['Pd']*(1+load[0]*np.sin(load[1]*lg['t'][k]+load[2])); d0,t0=run_pred(c0,lg,k,Pk); dT,tT=run_pred(cp,lg,k,Pk)
            e=angs(lg['th'][k+J+1],lg['de'][k+J+1])-angs(t0,d0); Dv=angs(tT,dT)-angs(t0,d0)
            b=max(b,np.abs(Dv).max()); bn=max(bn,np.abs(e).max()); bj=max(bj,np.abs(e+Dv).max())
        cs.append(b); nsc.append(bn); jt.append(bj)
    t=np.array(D[str(l)]['D_scores']); tr=np.array(D[str(l)]['true_scores'])
    print(l,'D score: calibration windows median %.4f max %.4f | test windows median %.4f max %.4f (ratio of medians %.3f) | margin(max of ||e_nom+D||) %.4f | true test scores: median %.4f max %.4f, frac<=margin %.2f'%(np.median(cs),max(cs),np.median(t),t.max(),np.median(t)/np.median(cs),max(jt),np.median(tr),tr.max(),(tr<=max(jt)).mean()),flush=True)
