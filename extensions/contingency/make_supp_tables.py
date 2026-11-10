"""Supplement tables (per-contingency coverage and missed failures) from results/summary_per_contingency.csv -> results/supp_per_contingency.tex.  Usage (repo root): python3 extensions/contingency/make_supp_tables.py"""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; R=list(csv.DictReader(open(ROOT/'extensions/contingency/results/summary_per_contingency.csv')))
def get(s): return {int(x['line']):x for x in R if x['set']==s}
def f(x,k,d=2): return '--' if x is None else ('%.'+str(d)+'f')%float(x[k])
out=[]
for case,names in (('case9',('case9_full','case9_full_rep','case9_full_mis')),('case39',('case39_full','case39_full_rep','case39_full_mis'))):
    o,rp,ms=[get(n) for n in names]
    out.append('\\begin{longtable}{@{}rrrrrrrrr@{}}\n\\caption{%s full outages: per-line results. Coverage = fraction of 20 held-out trajectories whose score is below the margin; margins and observed maximum scores in rad; twin 95\\%% Wilson intervals in \\texttt{summary\\_per\\_contingency.csv}. Rows marked with an asterisk have twin coverage below 0.8 in at least one run (missed failures). ``--\'\' = not run.}\\\\\n\\toprule\nline & $K_{\\rm static}$ & max score & twin margin & nominal & $\\varepsilon{+}K_{\\rm static}$ & twin & twin repl. & twin 5\\%% mism.\\\\\n\\midrule\\endfirsthead\n\\toprule line & $K_{\\rm static}$ & max score & twin margin & nominal & $\\varepsilon{+}K_{\\rm static}$ & twin & twin repl. & twin mism.\\\\\\midrule\\endhead\n'%case)
    for l in sorted(o):
        miss=any(float(d[l]['twin_coverage'])<0.8 for d in (o,rp,ms) if l in d)
        out.append('%d%s & %s & %s & %s & %s & %s & %s & %s & %s\\\\\n'%(l,'$^*$' if miss else '',f(o[l],'K_static',3),f(o[l],'obs_max_score',3),f(o[l],'twin_margin',3),f(o[l],'nominal_coverage'),f(o[l],'Kstatic_coverage'),f(o[l],'twin_coverage'),f(rp.get(l),'twin_coverage') if l in rp else '--',f(ms.get(l),'twin_coverage') if l in ms else '--'))
    out.append('\\bottomrule\n\\end{longtable}\n')
(ROOT/'extensions/contingency/results/supp_per_contingency.tex').write_text(''.join(out)); print('ok')
