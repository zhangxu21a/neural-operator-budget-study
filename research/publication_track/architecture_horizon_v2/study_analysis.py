"""Explicit-path, new-output-only analysis; all predeclared runs retained."""
from pathlib import Path
import json,csv,statistics,hashlib,math,os,sys,shutil
ROOT=Path(__file__).resolve().parent
LABELS={'coarse32':'Coarse 32','phys32':'Corrected 32','hf_online':'HF online','hf32':'HF reuse 32','hf128':'HF reuse 128'}
def main(engine):
 p=json.loads((ROOT/'protocol.json').read_text());selected=json.loads((ROOT/'selected.json').read_text());rows=[];runs=[]
 for arch,arm,seed in p['formal_jobs']:
  path=ROOT/'formal'/f'{arch}_{arm}_s{seed}';r=json.loads((path/'result.json').read_text());ev=json.loads((path/'evaluation.json').read_text());runs.append(r)
  for cap in p['checkpoints'][arch]:
   m=r['milestones'][str(cap)]
   rows.append({'architecture':arch,'arm':arm,'seed':seed,'budget':cap,'lr':r['lr'],'test':ev[str(cap)]['test']['relative_l2'],'ood':ev[str(cap)]['ood']['relative_l2'],'validation':m['validation'],'steps':m['steps_completed'],'selected_steps':m['selected_steps'],'hf':m['counts']['hf'],'decision_ready':m['decision_ready'],'nonfinite':r['nonfinite_loss'],'labels_seconds':sum(m['timings'][k] for k in ['hf','coarse8','coarse32','phys8','phys32']),**{k+'_seconds':v for k,v in m['timings'].items()}})
 assert len(rows)==130 and len(runs)==45
 with (ROOT/'all_runs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups={}
 for arch,arms in p['arms'].items():
  for cap in p['checkpoints'][arch]:
   for arm in arms:
    rr=[r for r in rows if r['architecture']==arch and r['arm']==arm and r['budget']==cap];assert len(rr)==5
    d={'n':5,'lr':selected[arch][arm]}
    for metric in ['test','ood','hf','steps','labels_seconds','training_seconds','decision_ready']:
     v=[r[metric] for r in rr];d[metric+'_mean']=statistics.mean(v);d[metric+'_sd']=statistics.stdev(v)
    groups[f'{arch}_{cap}_{arm}']=d
 pairs=[]
 for arch,arms in p['arms'].items():
  for cap in p['checkpoints'][arch]:
   for left,right in [('phys32','coarse32'),('hf128','hf_online'),('phys32','hf128')]+([('hf32','hf_online')] if arch=='deeponet' else []):
    for metric in ['test','ood']:
     a={r['seed']:100*r[metric] for r in rows if r['architecture']==arch and r['budget']==cap and r['arm']==left};b={r['seed']:100*r[metric] for r in rows if r['architecture']==arch and r['budget']==cap and r['arm']==right};ds=[a[k]-b[k] for k in sorted(a)]
     mean=statistics.mean(ds);sd=statistics.stdev(ds);half=2.7764451051977987*sd/math.sqrt(5);wins=sum(x<0 for x in ds);losses=sum(x>0 for x in ds);n=wins+losses
     signp=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
     pairs.append({'architecture':arch,'budget':cap,'left':left,'right':right,'metric':metric,'difference_pp':ds,'mean_pp':mean,'t95_pp':[mean-half,mean+half],'left_wins':wins,'ties':5-n,'sign_p':signp})
 horizon=[]
 for arm in p['arms']['fno']:
  a={r['seed']:100*r['test'] for r in rows if r['architecture']=='fno' and r['budget']==180 and r['arm']==arm};b={r['seed']:100*r['test'] for r in rows if r['architecture']=='fno' and r['budget']==600 and r['arm']==arm}
  horizon.append({'arm':arm,'id_600_minus_180_pp':[b[k]-a[k] for k in sorted(a)],'mean_pp':statistics.mean(b[k]-a[k] for k in a),'scope':'Paired endpoints of same five trajectories, not independent runs.'})
 tuning=[json.loads(x.read_text()) for x in (ROOT/'tuning').glob('*/result.json')]
 sourcechecks={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in p['sources'].items()}
 audit={'formal_runs':len(runs),'tuning_runs':len(tuning),'budget_points':len(rows),'all_sources_match':all(sourcechecks.values()),'source_checks':sourcechecks,'all_caps_pass':all(r['decision_ready']<=r['budget'] for r in rows) and all(r['wall']<=r['budget'] for r in runs+tuning),'nonfinite_formal':sum(r['nonfinite_loss'] for r in runs),'nonfinite_tuning':sum(r['nonfinite_loss'] for r in tuning),'all_initial_complete':all(r['initial_complete'] for r in runs+tuning),'max_hf_residual':max(r['max_hf_residual'] for r in runs+tuning),'tuning_allocated_per_arm':{f'{a}_{m}':sum(r['budget'] for r in tuning if r['architecture']==a and r['arm']==m) for a,arms in p['arms'].items() for m in arms}}
 assert audit['all_sources_match'] and audit['all_caps_pass'] and audit['all_initial_complete'] and len(tuning)==27
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'summary.json').write_text(json.dumps({'groups':groups,'pairs':pairs,'horizon':horizon},indent=2))
 md=['# 第二架构与长预算复核','', '27次主调参、45条正式轨迹、每方法5个独立训练池/优化种子；共130个相关预算点。DeepONet五种方法180秒；FNO四种方法600秒，记录120/180/360/600秒。新参考集与新训练池，不能与v1混为重复。','', '|架构|方法|预算秒|ID误差%（均值±SD）|OOD误差%|HF标签|更新次数|','|---|---|---:|---:|---:|---:|---:|']
 tex=[r'\section{Prospective architecture and horizon replication}',r'The follow-up uses new independent training pools and new validation/ID/OOD reference sets (32/128/64 inputs), preserving the gain-two channel family, 512-interval fine solve and 129-by-129 observation grid. All inputs use the same 129-grid log-coefficient sensors. The second architecture is the archived convolutional-branch DeepONet with rank 32; this is not a parameter-count-matched architecture ranking. Both architectures receive three 60-second learning-rate trials per method, using validation only. The FNO grid is 0.002/0.008/0.032; the DeepONet grid is 0.0005/0.002/0.008, chosen from its historical search range before the new results. This is equal allocated search time, not exhaustive architecture optimization.',r'Five new data-pool/optimization seeds are paired across methods and architectures. DeepONet runs five methods to 180 seconds, with a 120-second checkpoint. FNO runs four methods from scratch to 600 seconds, with 120/180/360-second checkpoints. It does not resume archived best-model checkpoints without optimizer state. Learning rates retain the common wall-time decay, with no retuning at long endpoints. Thus this is a fixed-policy horizon test, not a claim of converged performance. Shared reference construction and offline testing are separately recorded; labels, initialization, training inputs, validation and checkpoints remain inside each run timer.']
 for arch,cap in [('deeponet',180),('fno',180),('fno',600)]:
  tex +=[r'\begin{table}[ht]\centering\small',f'\\caption{{{arch.upper()} at the {cap}-second cap: five independent training units. Errors are percentages, mean $\\pm$ sample SD.}}',r'\begin{tabular}{lrrrr}\toprule Method & ID & OOD & HF labels & Updates\\\midrule']
  for arm in p['arms'][arch]:
   g=groups[f'{arch}_{cap}_{arm}'];md.append(f"|{arch}|{LABELS[arm]}|{cap}|{100*g['test_mean']:.3f} ± {100*g['test_sd']:.3f}|{100*g['ood_mean']:.3f}|{g['hf_mean']:.1f}|{g['steps_mean']:.1f}|")
   tex.append(f"{LABELS[arm]} & ${100*g['test_mean']:.3f}\\pm{100*g['test_sd']:.3f}$ & ${100*g['ood_mean']:.3f}\\pm{100*g['ood_sd']:.3f}$ & {g['hf_mean']:.1f} & {g['steps_mean']:.1f}\\\\")
  tex +=[r'\bottomrule\end{tabular}\end{table}']
 tex +=[r'\paragraph{Predeclared paired comparisons.} Differences below are ID percentage points, first minus second; negative values favor the first method. The paired intervals in the machine-readable release are exploratory and not multiplicity-adjusted. Five non-tied pairs cannot attain a two-sided exact sign-test $p<0.05$.']
 for x in pairs:
  if x['metric']!='test' or (x['architecture']=='deeponet' and x['budget']!=180) or (x['architecture']=='fno' and x['budget'] not in [180,600]):continue
  tex.append(f"{x['architecture'].upper()} at {x['budget']} s, {LABELS[x['left']]} versus {LABELS[x['right']]}: {x['mean_pp']:.3f} percentage points, {x['left_wins']}/5 pairs favoring the first method.\\par")
 md+=['','## 审计与限制',json.dumps({k:v for k,v in audit.items() if k!='source_checks'},ensure_ascii=False,indent=2),'','长预算检查点不增加独立重复数；新旧评估集不同，不能直接归因跨轮绝对误差差值。两架构容量不同；学习率搜索区间不同但分配时间相同；仅一个调参训练池。所有不利种子和非有限损失标记均保留。']
 (ROOT/'results_report.md').write_text('\n'.join(md)+'\n')
 os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_config');sys.path.append(str(ROOT.parents[1]/'new_directions/python_deps'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');colors=['#225b87','#c07826','#3a8065','#8a568b']
 for i,arm in enumerate(p['arms']['deeponet']):
  vals=[100*r['test'] for r in rows if r['architecture']=='deeponet' and r['arm']==arm and r['budget']==180]
  axes[0].scatter([i+(j-2)*.08 for j in range(5)],vals,s=20,color='#225b87');axes[0].plot([i-.22,i+.22],[statistics.mean(vals)]*2,color='#c07826')
 axes[0].set_xticks(range(5),[LABELS[a] for a in p['arms']['deeponet']],rotation=30,ha='right');axes[0].set_title('DeepONet, 180 s');axes[0].set_ylabel('ID relative L2 error (%)')
 for color,arm in zip(colors,p['arms']['fno']):
  caps=[180,360,600]
  for seed in p['seeds']:
   ys=[next(100*r['test'] for r in rows if r['architecture']=='fno' and r['arm']==arm and r['budget']==cap and r['seed']==seed) for cap in caps];axes[1].plot(caps,ys,color=color,alpha=.2,lw=.8)
  axes[1].plot(caps,[100*groups[f'fno_{c}_{arm}']['test_mean'] for c in caps],marker='o',label=LABELS[arm],color=color)
 axes[1].set(title='FNO: dependent horizon checkpoints',xlabel='Construction budget cap (s)',ylabel='ID relative L2 error (%)');axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
 fig.savefig(ROOT/'architecture_horizon.png',dpi=180);fig.savefig(ROOT/'architecture_horizon.pdf');plt.close(fig)
 tex +=[r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{images/architecture_horizon.pdf}\caption{Left: all five DeepONet training units. Right: FNO means (bold) and individual trajectories (faint). Checkpoints on each trajectory are dependent. Vertical scales differ.}\end{figure}',r'These results remain limited to one PDE family. They test transfer to one additional architecture and a longer finite construction horizon, not cross-PDE universality, complete convergence, or numerical-solver optimality. No test-driven parameter reselection, seed deletion or failed-run replacement is performed.']
 (ROOT/'architecture_horizon_results.tex').write_text('\n\n'.join(tex)+'\n')
 # Append evidence to the working paper; do not silently replace earlier findings.
 paper=ROOT.parent/'manuscript/paper';shutil.copy2(ROOT/'architecture_horizon_results.tex',paper/'architecture_horizon_results.tex');shutil.copy2(ROOT/'architecture_horizon.pdf',paper/'images/architecture_horizon.pdf')
 f=paper/'main.tex';m=f.read_text()
 if '\\input{architecture_horizon_results.tex}' not in m:m=m.replace('\\section{Discussion and limitations}','\\input{architecture_horizon_results.tex}\n\n\\section{Discussion and limitations}')
 f.write_text(m)
 (ROOT/'analysis_complete.json').write_text(json.dumps({'complete':True,'manuscript_section_added':str(f),'pending':'Review interpretation, update abstract/conclusion and compile final manuscript.'},indent=2))
