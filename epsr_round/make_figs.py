import json,glob,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
OUT='./'
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
# Fig A: certificate tightness + near-singular
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
for c,m,col in [('case9','s','#d95f02'),('case39','o','#1b6ca8')]:
    d=json.load(open(f'kantorovich_{c}.json'))
    ax[0].scatter([x['exact_E'] for x in d],[x['rho'] for x in d],marker=m,s=18,color=col,label=c)
mx=0.7; ax[0].plot([0,mx],[0,mx],'k--',lw=.8); ax[0].set_xlabel(r'exact displacement $\|\theta^*-\theta_0\|_M$'); ax[0].set_ylabel(r'certified radius $\rho_*$'); ax[0].legend(frameon=False)
ns=json.load(open('nearsingular.json'))
for (c,sty),col in zip([('case9','-o'),('case39','-s')],['#d95f02','#1b6ca8']):
    v=ns[c]; half=len(v)//2
    for part,ls in [(v[:half],'-'),(v[half:],'--')]:
        ax[1].plot([x['lam_min_M'] for x in part],[x['h'] for x in part],ls,marker='o',ms=3,color=col,label=c if ls=='-' else None)
ax[1].axhline(.5,color='k',lw=.8,ls=':'); ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\lambda_{\min}(M)$'); ax[1].set_ylabel(r'$h=K\eta$'); ax[1].text(0.03,.52,'certificate fails above 1/2',fontsize=7)
ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(OUT+'Figure_certificate.pdf'); plt.close()

# Fig B
tab={t['line']:t for t in json.load(open('expA_summary.json'))}
on={}
for f in ['setBrep_case39','setBrepcal5002_case39']:
    for r in json.load(open(f'results_setB/{f}_full_0.json'))['rows']: on.setdefault(r['line'],[]).append(r['online_traj_cov'])
L=sorted(tab); x=np.arange(len(L))
fig,ax=plt.subplots(figsize=(7.2,2.9))
tw=[tab[l]['cov'] for l in L]; lo=[tab[l]['cov']-tab[l]['lo'] for l in L]; hi=[tab[l]['hi']-tab[l]['cov'] for l in L]
ax.errorbar(x-0.12,tw,yerr=[lo,hi],fmt='o',ms=3.5,color='#1b6ca8',lw=.8,label='twin margin, 4 calibration draws (pooled, 95% Wilson)')
ax.plot(x+0.12,[np.mean(on[l]) for l in L],'s',ms=3.5,color='#d95f02',label='online margin, 2 calibration draws')
ax.axhline(20/21,color='k',ls='--',lw=.8); ax.set_xticks(x); ax.set_xticklabels(L,fontsize=6,rotation=90); ax.set_xlabel('removed line'); ax.set_ylabel('coverage'); ax.set_ylim(0.3,1.04)
ax.legend(frameon=False,fontsize=7,loc='lower right'); plt.tight_layout(); plt.savefig(OUT+'Figure_draws.pdf'); plt.close()
print('ok')
