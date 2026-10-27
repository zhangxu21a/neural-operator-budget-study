import bootstrap
import scoring
import os,sys,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_config');sys.path.append('/Volumes/MacShare/Something/research_q1_20261024/new_directions/python_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=json.loads((ROOT/'results_summary.json').read_text())['groups'];fig,axs=plt.subplots(2,2,figsize=(16,9),layout='constrained')
labels={'data':'Adaptive data','gradient':'Gradient','uniform':'Uniform','field':'Field error','pretrain':'Pretrain HF','hf':'High fidelity','phys_only':'Physics only','author_mlmc':'Author MLMC'}
colors=dict(zip(labels,['#4987a5','#d96840','#93a4b4','#c5a24c','#718c5e','#66758b','#4c956c','#9275a6']))
for ai,arch in enumerate(['selected','matched']):
 for bi,budget in enumerate([300,1200]):
  ax=axs[ai,bi];items=[(k.split('_',2)[2],v) for k,v in s.items() if k.startswith(f'{arch}_{budget}_')];x=np.arange(len(items))
  ax.bar(x,[100*v['test_mean'] for _,v in items],yerr=[100*(v['test_sd'] or 0) for _,v in items],capsize=4,color=[colors[m] for m,v in items])
  ax.set_xticks(x,[labels[m].replace(' ','\n') for m,v in items]);ax.set_ylabel('Fresh test relative L2 (%)');ax.set_title(f'{arch.upper()} — {budget//60} minutes, 3 seeds');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
fig.suptitle('Darcy: acquisition mechanism and learning-rate controls',fontsize=14)
fig.savefig(ROOT/'results.png',dpi=180);fig.savefig(ROOT/'results.pdf');plt.close(fig)
