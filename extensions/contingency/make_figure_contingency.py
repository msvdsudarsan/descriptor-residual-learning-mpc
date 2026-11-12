"""Figure for the contingency section: per-line coverage on the 35 case39 full-outage candidates (original and replication runs) for the nominal margin,
eps+K_static and the predictor-twin margin, from results/summary_per_contingency.csv.  Usage (repo root): python3 extensions/contingency/make_figure_contingency.py [out.pdf]"""
import csv, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; R=list(csv.DictReader(open(ROOT/'extensions/contingency/results/summary_per_contingency.csv')))
def get(s): return {int(x['line']):x for x in R if x['set']==s}
o,rp=get('case39_full'),get('case39_full_rep'); lines=sorted(o)
fig,ax=plt.subplots(figsize=(7.4,3.1)); x=np.arange(len(lines))
ax.axhline(20/21,color='0.55',lw=0.8,ls='--'); ax.text(len(lines)-0.5,20/21+0.02,'20/21',ha='right',fontsize=7,color='0.35')
ax.plot(x,[float(o[l]['nominal_coverage']) for l in lines],'s',ms=3.2,color='#b0b0b0',label=r'nominal $\varepsilon$')
ax.plot(x,[float(o[l]['Kstatic_coverage']) for l in lines],'^',ms=4,color='#d95f02',label=r'$\varepsilon+K_{\rm static}$')
ax.plot(x,[float(o[l]['twin_coverage']) for l in lines],'o',ms=4.5,color='#1b6ca8',label='twin margin, original')
ax.plot(x+0.18,[float(rp[l]['twin_coverage']) for l in lines],'o',ms=3.5,mfc='none',color='#1b6ca8',label='twin margin, replication')
for l in (8,10,11):
    i=lines.index(l); ax.annotate('line %d'%l,(i,float(o[l]['twin_coverage'])),xytext=(i,float(o[l]['twin_coverage'])-0.17),ha='center',fontsize=7,arrowprops=dict(arrowstyle='-',lw=0.5))
ax.set_xticks(x); ax.set_xticklabels(lines,fontsize=6.5); ax.set_xlabel('outaged line (case39, sorted)',fontsize=8); ax.set_ylabel('coverage of 20 held-out runs',fontsize=8)
ax.set_ylim(-0.05,1.08); ax.tick_params(axis='y',labelsize=7); ax.legend(fontsize=6.5,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(0.5,1.0))
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv)>1 else ROOT/'figures/Figure_contingency.pdf')
