import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
R=Path(__file__).resolve().parents[1]/'research/publication_track'
s=json.loads((R/'fixed_updates_v12/summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
for ax,split in zip(axs,['test','ood']):
 for k,rate in enumerate([.002,.008]):
  cs=[next(x for x in s['contrasts'] if x['type']=='corrected_minus_coarse' and x['model']=='last' and x['split']==split and x['rate']==rate and x['updates']==u) for u in [800,1400]]
  for i in range(5): ax.plot([k*3,k*3+1],[c['all_differences_pp'][i] for c in cs],color=plt.cm.tab10(i),marker='o',lw=1,alpha=.8,label=str([3101,3202,3303,3404,3505][i]) if k==0 else None)
  ax.plot([k*3,k*3+1],[c['mean_pp'] for c in cs],color='black',marker='D',lw=2,label='Mean' if k==0 else None)
 ax.axhline(0,color='gray',ls='--',lw=1); ax.set_xticks([0,1,3,4],['800','1400','800','1400']); ax.set_xlabel('Updates:  rate 0.002                  rate 0.008'); ax.set_title('ID' if split=='test' else 'OOD'); ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Corrected − coarse error (percentage points)'); fig.legend(*axs[0].get_legend_handles_labels(), title='Seed',fontsize=8,loc='lower center',ncol=6,bbox_to_anchor=(0.5,-0.12))
fig.tight_layout(); fig.savefig(a.out/'fixed_updates_v12.pdf',bbox_inches='tight'); fig.savefig(a.out/'fixed_updates_v12.png',dpi=160,bbox_inches='tight')
