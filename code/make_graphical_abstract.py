# Writes the graphical abstract (Graphical_abstract_EPSR.pdf/.png in the current directory); copied to figures/graphical_abstract.*. Not part of run_all.sh.
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'opentype/inter/Inter-' in f: fm.fontManager.addfont(f)
plt.rcParams['font.family']='Inter'; plt.rcParams['pdf.fonttype']=42
W,H=2000,666
fig=plt.figure(figsize=(13.28,4.425)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
ink='#1f2d3d'; edge='#2f4a66'
ax.text(1000,38,'Physics-informed residual learning of descriptor power-network models: prediction, contingency margins and a control test',ha='center',va='center',fontsize=14.6,fontweight='bold',color=ink)
boxes=[(30,602,'#dce8f4','1  Model',['Swing dynamics and power balance kept','0 = g(x, z, u): algebraic angles','Only the missing differential term is','learned (weak-form sparse regression)','','Newton projection returns the bus angles','to the constraint manifold at every stage']),
(714,1287,'#e4f0e0','2  Results (lossless case9, case39)',['Mean per-run state-RMSE ratio','(10 horizons): 0.17 to 0.24 of physics-only','Algebraic residual at solver tolerance','Row-stochastic angle sensitivity:','algebraic error <= rotor-angle span','Analytic gradient: case39 control step','588 ms → 24 ms','Case39: 3.6-4.2% gain vs static law;','no consistent gain over recalibrated physics']),
(1400,1968,'#f9e8d8','3  Outage certificate and margin',['Outage certificate: Newton-Kantorovich,','held on all 41 non-islanding outages','Predictor rerun under outage topology:','pooled coverage 0.945, 4 calibration draws','Misses lines 8, 10, 11, 37; unresolved','Coverage conditional on exchangeability','No independent AC, real-grid or','stability validation'])]
for x0,x1,c,h,lines in boxes:
    ax.add_patch(FancyBboxPatch((x0,117),x1-x0,481,boxstyle='round,pad=0,rounding_size=14',fc=c,ec=edge,lw=2))
    ax.text((x0+x1)/2,157,h,ha='center',va='center',fontsize=14.2,fontweight='bold',color=ink)
    for i,t in enumerate(lines): ax.text(x0+21,212+43*i,t,ha='left',va='center',fontsize=(12.1 if x0==714 else 12.6),color=ink)
for xa,xb in [(608,705),(1294,1390)]:
    ax.add_patch(FancyArrowPatch((xa,357),(xb,357),arrowstyle='-|>',mutation_scale=34,lw=3.2,color=edge,shrinkA=0,shrinkB=0))
fig.savefig('Graphical_abstract_EPSR.pdf'); fig.savefig('Graphical_abstract_EPSR.png',dpi=200)
