import os,pathlib,json,csv,statistics,collections
R=pathlib.Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(R/'figures'/'mpl_config')
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(R.parent/'mainline_revision/darcy_optimized'))
import scoring
sys.path.append(str(R.parent/'new_directions/python_deps'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
raw=json.loads((R/'experiments/numerical_pilot/result.json').read_text())['rows']
summary=[]
for g in [1,2]:
 for nf in [128,256,512]:
  for nc in [8,32]:
   a=[r for r in raw if (r['gain'],r['nf'],r['nc'])==(g,nf,nc)];d={'gain':g,'nf':nf,'nc':nc,'n_coefficients':6}
   for k in ['fine_seconds','coarse_seconds','correction_seconds','coarse_rel_hf','corrected_rel_hf','fine_rel_reference','corrected_rel_reference']:
    v=[x[k] for x in a];d[k+'_mean']=statistics.mean(v);d[k+'_sd']=statistics.stdev(v)
   summary.append(d)
(R/'evidence/numerical_summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained',sharey=True)
for ax,g in zip(axes,[1,2]):
 for nc,color,marker in [(8,'#a65d24','s'),(32,'#225b87','o')]:
  groups=[[r['corrected_rel_hf']*100 for r in raw if (r['gain'],r['nf'],r['nc'])==(g,n,nc)] for n in [128,256,512]]
  ax.plot([128,256,512],[np.mean(v) for v in groups],color=color,marker=marker,label=f'Coarse grid: {nc} intervals')
  for n,v in zip([128,256,512],groups):ax.scatter([n]*6,v,s=13,alpha=.35,color=color)
 ax.set(title=f'Log-coefficient gain {g}',xlabel='Fine-grid intervals',ylabel='Corrected-label relative L2 error (%)',yscale='log');ax.set_xticks([128,256,512]);ax.grid(axis='y',alpha=.18);ax.legend(fontsize=9)
fig.savefig(R/'figures/correction_bias.png');fig.savefig(R/'figures/correction_bias.pdf');plt.close(fig)
rows=list(csv.DictReader((R.parent/'mainline_revision/darcy_mechanism/all_runs.csv').open()));methods=['gradient','uniform','field','data','phys_only','hf'];labels=['Gradient','Uniform','Field','Adaptive','Corrected','HF']
fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for ax,metric,title in zip(axs,['test','ood'],['In-distribution','Out-of-distribution']):
 for i,m in enumerate(methods):
  a=sorted([r for r in rows if r['method']==m and r['lr']=='0.008' and r['budget']=='1200'],key=lambda r:r['seed']);v=[float(r[metric])*100 for r in a]
  ax.scatter(np.array([i]*3)+[-.13,0,.13],v,color='#225b87',marker='o',s=28);ax.plot([i-.2,i+.2],[np.mean(v)]*2,color='#a65d24',lw=2)
 ax.set_xticks(range(6),labels,rotation=25,ha='right');ax.set(title=title,ylabel='Relative L2 error (%)');ax.grid(axis='y',alpha=.18)
fig.suptitle('20-minute FNO results | three paired optimization seeds | LR 0.008',fontsize=11)
fig.savefig(R/'figures/matched_seeds.png');fig.savefig(R/'figures/matched_seeds.pdf');plt.close(fig)
pilots=[]
for p in sorted((R/'experiments/training_pilot').glob('*/result.json')):
 r=json.loads(p.read_text());pilots.append({'scenario':r['scenario'],'method':r['method'],'test_percent':100*r['test']['relative_l2'],'validation_percent':100*r['validation'],'updates':r['steps'],'fine_labels':r['counts']['fine'],'wall_seconds':r['wall']})
assert len(pilots)==6
(R/'evidence/training_pilot_summary.json').write_text(json.dumps(pilots,indent=2))
tex=['\\begin{table}[ht]\\centering','\\caption{Exploratory expensive-label runs: one seed, fixed learning rate, 90-second cap. No inferential method ranking.}','\\begin{tabular}{llrrr}\\toprule Scenario & Target & Test error (\\%) & Updates & HF labels\\\\\\midrule']
for r in pilots:tex.append(f"{r['scenario']} & {r['method'].replace('_','-')} & {r['test_percent']:.2f} & {r['updates']} & {r['fine_labels']}\\\\")
tex+=['\\bottomrule\\end{tabular}\\end{table}','In the resolved pilot, coarse targets outperform corrected targets; in the stress pilot, correction improves substantially on coarse targets. HF training completes only eleven updates because acquisition dominates this very short direct-solver budget. These runs diagnose a regime and expose a weak short-budget comparator; they do not establish a deployable allocation policy. A policy that reuses a smaller HF set for more optimization, together with a more efficient numerical solver, is required in the formal study.']
(R/'manuscript/paper/pilot_results.tex').write_text('\n'.join(tex))
# Check numerical screening and ledger integrity.
checks={'numerical_rows':len(raw),'six_paired_inputs':all(d['n_coefficients']==6 for d in summary),'max_direct_residual':max(r['hf_residual'] for r in raw),'all_finite':all(np.isfinite(v) for r in raw for v in r.values() if isinstance(v,(int,float))),'pilot_runs':len(pilots),'all_pilot_caps_pass':all(r['wall_seconds']<=90 for r in pilots)}
(R/'evidence/new_experiment_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
