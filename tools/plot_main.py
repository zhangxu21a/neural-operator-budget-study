import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();O=a.out
if O.exists():raise SystemExit('Choose a fresh output directory.')
O.mkdir(parents=True);P=Path(__file__).resolve().parents[1]/'research/publication_track/mechanism_review_v4'
def load(name):
 rows=list(csv.DictReader((P/name).open()))
 for r in rows:
  for k,v in r.items():
   if k in ['study','run','architecture','arm']:continue
   r[k]=float(v)
 return rows
rows=load('endpoint_ledger.csv');observations=load('assessment_observations.csv')
# Primary figure: latest balanced cohort only, all seeds and rates retained.
R=[r for r in rows if r['study']=='matched_rate_v3' and r['cap']==600]
keys=[(.002,'coarse32'),(.002,'phys32'),(.008,'coarse32'),(.008,'phys32')]
labels=['Coarse\n0.002','Corrected\n0.002','Coarse\n0.008','Corrected\n0.008']
colors={'coarse32':'#285F87','phys32':'#B87324'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
a=axs[0,0]
bottom=np.zeros(4)
for metric,label,col,hatch in [('label_seconds','Numerical labels','#B87324',''),('training_seconds','Training','#285F87',''),('other_seconds','Other charged time','#B9B9B9','//')]:
 vals=[np.mean([r[metric] for r in R if (r['lr'],r['arm'])==key]) for key in keys]
 a.bar(range(4),vals,bottom=bottom,color=col,label=label,hatch=hatch,width=.65)
 for i,(v,b) in enumerate(zip(vals,bottom)):
  if v>35:a.text(i,b+v/2,f'{v:.0f}',ha='center',va='center',color='white',fontsize=10)
 bottom+=vals
for i,key in enumerate(keys):
 rs=sorted([r for r in R if (r['lr'],r['arm'])==key],key=lambda r:r['seed'])
 a.scatter(np.array([i]*5)+np.linspace(-.12,.12,5),[r['ready_seconds'] for r in rs],c='black',s=10,zorder=4)
a.axhline(600,color='#444444',ls=':',lw=1);a.set_ylim(0,690);a.set_xticks(range(4),labels);a.set_ylabel('Charged wall time (s)');a.set_title('(a) Cost ledger at the 600 s cap',loc='left');a.legend(fontsize=8,ncol=3,loc='upper center',frameon=False)
a=axs[0,1]
for i,key in enumerate(keys):
 rs=sorted([r for r in R if (r['lr'],r['arm'])==key],key=lambda r:r['seed']);xs=i+np.linspace(-.15,.15,5)
 for x,r in zip(xs,rs):
  a.plot([x,x],[r['selected_updates'],r['completed_updates']],color='#999999',lw=1)
  a.scatter(x,r['completed_updates'],marker='o',color=colors[key[1]],s=22)
  a.scatter(x,r['selected_updates'],marker='s',facecolors='white',edgecolors=colors[key[1]],s=26,zorder=3)
a.scatter([],[],c='#444444',marker='o',label='Completed');a.scatter([],[],facecolors='white',edgecolors='#444444',marker='s',label='Selected checkpoint');a.legend(fontsize=8,frameon=False,loc='upper left');a.set_ylim(0,1700);a.set_xticks(range(4),labels);a.set_ylabel('Optimizer updates');a.set_title('(b) Spent updates vs retained model',loc='left');a.grid(axis='y',alpha=.15)
for ax,lr in zip(axs[1],[.002,.008]):
 for seed in sorted(set(r['seed'] for r in R)):
  pair=[next(r for r in R if r['seed']==seed and r['lr']==lr and r['arm']==arm) for arm in ['coarse32','phys32']]
  ax.plot([r['selected_updates'] for r in pair],[r['id_pct'] for r in pair],color='#B8B8B8',lw=.8,zorder=1)
 for arm,marker in [('coarse32','o'),('phys32','s')]:
  rs=[r for r in R if r['lr']==lr and r['arm']==arm];ax.scatter([r['selected_updates'] for r in rs],[r['id_pct'] for r in rs],c=colors[arm],marker=marker,s=42,label='Coarse' if arm=='coarse32' else 'Corrected',zorder=3)
 ax.set(xlim=(0,1600),ylim=(11,20),xlabel='Updates of validation-selected checkpoint',ylabel='ID relative L2 error (%)');ax.set_title(f'({"c" if lr==.002 else "d"}) Initial learning rate {lr}',loc='left');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
fig.savefig(O/'cost_opportunity_error.pdf');fig.savefig(O/'cost_opportunity_error.png',dpi=180);plt.close(fig)
# Schedule diagnostic: function evaluated at recorded assessment completion, NOT actual step LR.
fig,axs=plt.subplots(1,2,figsize=(11.6,3.8),layout='constrained')
for ax,lr in zip(axs,[.002,.008]):
 for arm in ['coarse32','phys32']:
  for i,seed in enumerate(sorted(set(r['seed'] for r in R))):
   qs=[q for q in observations if q['study']=='matched_rate_v3' and q['lr']==lr and q['arm']==arm and q['seed']==seed and q['steps']>0]
   ax.plot([q['steps'] for q in qs],[q['schedule_at_observation']/lr for q in qs],color=colors[arm],alpha=.45,lw=1,ls='-' if arm=='coarse32' else '--',label=(('Coarse' if arm=='coarse32' else 'Corrected') if i==0 else None))
 ax.set(xlim=(0,1600),ylim=(0,1),xlabel='Completed updates at assessment',ylabel='Schedule value / initial learning rate');ax.set_title(f'Initial learning rate {lr}');ax.legend(frameon=False);ax.grid(alpha=.15)
fig.savefig(O/'schedule_diagnostic.pdf');fig.savefig(O/'schedule_diagnostic.png',dpi=180);plt.close(fig)
