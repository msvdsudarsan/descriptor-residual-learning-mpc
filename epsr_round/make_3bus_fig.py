import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
fig,ax=plt.subplots(figsize=(4.6,2.8)); ax.set_xlim(0,10); ax.set_ylim(-0.6,7); ax.axis('off')
B={1:(1.8,4.2),2:(8.2,4.2),3:(5,1.6)}
for k,(x,y) in B.items(): ax.plot([x-0.5,x+0.5],[y,y],'k',lw=3); ax.text(x-0.9 if k<3 else x+0.9,y-0.45 if k<3 else y+0.15,f'Bus {k}',ha='center',fontsize=9)
for (a,b,lab,dy) in [(1,2,'b=5',0.25),(1,3,'b=10',0),(2,3,'b=10',0)]:
    (x1,y1),(x2,y2)=B[a],B[b]; ax.plot([x1,x2],[y1,y2],'#38546e',lw=1.4); ax.text((x1+x2)/2+(0 if a==1 and b==2 else (-0.7 if a==1 else 0.7)),(y1+y2)/2+dy+(0.15 if a==1 and b==2 else 0),lab,fontsize=8,ha='center')
for k,m in [(1,'0.6'),(2,'0.4')]:
    x,y=B[k]; ax.plot([x,x],[y,y+0.9],'k',lw=1); ax.add_patch(Circle((x,y+1.25),0.35,fill=False,lw=1.2)); ax.text(x,y+1.25,'G',ha='center',va='center',fontsize=9); ax.text(x+0.6,y+1.25,f'$P_m$={m}',fontsize=8,va='center')
ax.annotate('',xy=(5,0.6),xytext=(5,1.6),arrowprops=dict(arrowstyle='-|>',lw=1.4)); ax.text(5.35,0.8,'load 1.0',fontsize=8)
ax.text(5,-0.4,'generator susceptance $1/x_d\'=3.33$; all values per unit',ha='center',fontsize=7)
plt.tight_layout(); plt.savefig('supp/Figure_3bus.pdf')
