"""Checks the values quoted in the manuscript/supplement about the direct per-line calibration against results/direct_cal_summary.json
(and, if a directory is given, that a fresh regeneration of the raw scores equals the stored scores exactly).
Usage: python3 extensions/contingency/check_direct_values.py [fresh_dir]"""
import json, sys
from pathlib import Path
R=Path(__file__).resolve().parent/'results'; S=json.load(open(R/'direct_cal_summary.json'))['summary']; bad=0
def chk(name,val,exp,tol):
    global bad
    ok=abs(val-exp)<=tol; bad+=not ok; print(('ok   ' if ok else 'DIFF ')+name,'%.4g'%val,'expected',exp)
chk('twin pooled coverage',S['twin']['pooled_cov'][0],0.95,0.005); chk('twin CI lo',S['twin']['pooled_cov'][1],0.92,0.005); chk('twin CI hi',S['twin']['pooled_cov'][2],0.98,0.005)
chk('direct q20 pooled',S['q20']['pooled_cov'][0],0.90,0.005); chk('direct q20 CI lo',S['q20']['pooled_cov'][1],0.88,0.005); chk('direct q20 CI hi',S['q20']['pooled_cov'][2],0.92,0.005)
chk('direct max20 pooled',S['max20']['pooled_cov'][0],0.945,0.0005)
chk('GM ratio twin/direct q20',S['paired_q20']['gm_ratio'],1.08,0.005); chk('ratio CI lo',S['paired_q20']['ci'][0],1.05,0.005); chk('ratio CI hi',S['paired_q20']['ci'][1],1.11,0.005)
chk('twin wider on',S['paired_q20']['n_twin_wider'],30,0); chk('twin below 0.9',S['twin']['below90'],4,0); chk('twin below 0.8',S['twin']['below80'],2,0)
chk('direct q20 lines below 0.8',len(S['q20']['lines_below80']),2,0); chk('nominal pooled',S['nominal']['pooled_cov'][0],0.20,0.005); chk('static pooled',S['Kstatic']['pooled_cov'][0],0.60,0.005)
chk('leakage checks: cov_mismatch',S['checks']['cov_mismatch'],0,0); chk('seed overlap',float(S['checks']['seed_overlap']),0,0); chk('load overlap',S['checks']['load_overlap'],0,0)
# supplement Table (pooled summary)
for k,exp in (('q10',(0.91,0.89,0.92,12,2,1.07)),('q40',(0.90,0.88,0.92,11,1,1.09)),('max20',(0.945,0.932,0.957,3,0,1.04)),('max40',(0.97,0.96,0.98,2,0,1.02))):
    r=S[k]; chk(k+' cov',r['pooled_cov'][0],exp[0],0.0051); chk(k+' lo',r['pooled_cov'][1],exp[1],0.0051); chk(k+' hi',r['pooled_cov'][2],exp[2],0.0051)
    chk(k+' <0.9',r['below90'],exp[3],0); chk(k+' <0.8',r['below80'],exp[4],0); chk(k+' ratio',r['gm_margin_ratio_twin_over_direct'],exp[5],0.0051)
chk('nominal lo',S['nominal']['pooled_cov'][1],0.08,0.0051); chk('nominal hi',S['nominal']['pooled_cov'][2],0.33,0.0051); chk('nominal <0.9',S['nominal']['below90'],29,0)
chk('static lo',S['Kstatic']['pooled_cov'][1],0.43,0.0051); chk('static hi',S['Kstatic']['pooled_cov'][2],0.74,0.0051); chk('static <0.9',S['Kstatic']['below90'],15,0)
chk('margin/median twin',S['margin_over_median_test']['twin'],1.23,0.005); chk('margin/median direct',S['margin_over_median_test']['q20'],1.14,0.005)
chk('wilcoxon width p <1e-6',float(S['paired_q20']['wilcoxon_logratio_p']<1e-6),1,0); chk('wilcoxon cov p',S['paired_q20']['wilcoxon_covdiff_p'],3e-4,0.5e-4)
if len(sys.argv)>1:
    n=0
    for f in sorted((R/'direct_cal').glob('line_*.json')):
        a=json.load(open(f)); b=json.load(open(Path(sys.argv[1])/f.name))
        for k in ('cal','test0','test1'):
            n+=1
            if max(abs(x-y) for x,y in zip(a[k]['score'],b[k]['score']))>1e-9: bad+=1; print('DIFF scores',f.name,k)
    print('regenerated score vectors compared:',n)
print('CHECK_DIRECT',  'FAILED' if bad else 'PASSED'); sys.exit(1 if bad else 0)
