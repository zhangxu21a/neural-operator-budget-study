from pathlib import Path
import json,csv,statistics,math,hashlib,os,sys,shutil
R=Path(__file__).resolve().parent
LABELS={'coarse32':'Coarse 32','phys32':'Corrected 32'}
def main(e):
 p=json.loads((R/'protocol.json').read_text());rows=[];runs=[]
 for seed,lr,arm in p['jobs']:
  d=R/'formal'/f'{arm}_lr{lr}_s{seed}';r=json.loads((d/'result.json').read_text());ev=json.loads((d/'evaluation.json').read_text());runs.append(r)
  assert r['lr']==lr and r['seed']==seed and r['arm']==arm
  for cap in p['checkpoints']:
   m=r['milestones'][str(cap)]
   rows.append({'seed':seed,'lr':lr,'arm':arm,'budget':cap,'id':ev[str(cap)]['test']['relative_l2'],'ood':ev[str(cap)]['ood']['relative_l2'],'validation':m['validation'],'steps':m['steps_completed'],'hf':m['counts']['hf'],'ready':m['decision_ready'],'label_seconds':sum(m['timings'][k] for k in ['hf','coarse8','coarse32','phys8','phys32']),'training_seconds':m['timings']['training']})
 assert len(rows)==80 and len(runs)==20
 with (R/'all_runs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups={};contrasts=[];interactions=[]
 def stats(ds):
  mean=statistics.mean(ds);sd=statistics.stdev(ds);half=2.7764451051977987*sd/math.sqrt(5);wins=sum(d<0 for d in ds);loss=sum(d>0 for d in ds);n=wins+loss
  return {'differences_pp':ds,'mean_pp':mean,'sd_pp':sd,'t95_pp':[mean-half,mean+half],'negative_pairs':wins,'ties':5-n,'sign_p':min(1.,2*sum(math.comb(n,k) for k in range(min(wins,loss)+1))/2**n) if n else 1.}
 for cap in p['checkpoints']:
  for lr in p['rates']:
   for arm in p['arms']:
    rr=[r for r in rows if r['budget']==cap and r['lr']==lr and r['arm']==arm];assert len(rr)==5
    g={'n':5}
    for metric in ['id','ood','steps','hf','ready','label_seconds','training_seconds']:
     v=[r[metric] for r in rr];g[metric+'_mean']=statistics.mean(v);g[metric+'_sd']=statistics.stdev(v)
    groups[f'{cap}_{lr}_{arm}']=g
   for metric in ['id','ood']:
    a={r['seed']:100*r[metric] for r in rows if r['budget']==cap and r['lr']==lr and r['arm']=='phys32'};b={r['seed']:100*r[metric] for r in rows if r['budget']==cap and r['lr']==lr and r['arm']=='coarse32'}
    contrasts.append({'budget':cap,'lr':lr,'metric':metric,'definition':'corrected minus coarse',**stats([a[s]-b[s] for s in p['seeds']])})
  for metric in ['id','ood']:
   lo=next(x for x in contrasts if x['budget']==cap and x['lr']==.002 and x['metric']==metric);hi=next(x for x in contrasts if x['budget']==cap and x['lr']==.008 and x['metric']==metric)
   interactions.append({'budget':cap,'metric':metric,'definition':'(corrected-coarse at .008) minus (corrected-coarse at .002)',**stats([a-b for a,b in zip(hi['differences_pp'],lo['differences_pp'])])})
 checks={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in p['sources'].items()}
 audit={'formal_runs':20,'budget_points':80,'seeds_per_cell':5,'new_tuning_trials':0,'all_frozen_sources_match':all(checks.values()),'source_checks':checks,'all_caps_pass':all(r['ready']<=r['budget'] for r in rows) and all(r['wall']<=r['budget'] for r in runs),'nonfinite_loss_runs':sum(r['nonfinite_loss'] for r in runs),'all_initial_complete':all(r['initial_complete'] for r in runs),'max_hf_residual':max(r['max_hf_residual'] for r in runs)}
 assert audit['all_frozen_sources_match'] and audit['all_caps_pass'] and audit['all_initial_complete']
 e.atomic_json(R/'audit.json',audit);e.atomic_json(R/'summary.json',{'groups':groups,'contrasts':contrasts,'interactions':interactions})
 md=['# FNO匹配学习率对照','', '两档固定学习率×两种标签×五个新独立训练池/优化种子，共20条600秒轨迹、80个相关预算点。无新增调参，不选择有利学习率。负差值表示校正误差更低。','', '|学习率|预算|粗标签ID%|校正ID%|配对差/百分点|校正胜出|','|---:|---:|---:|---:|---:|---:|']
 tex=[r'\section{Matched-learning-rate follow-up}',r'The prior FNO comparison selected learning rates 0.002 for coarse targets and 0.008 for corrected targets. We therefore fix both rates in a complete two-by-two comparison of coarse32 and corrected32. Each cell has five new paired training pools and optimization seeds, with a fresh validation/ID/OOD reference set (32/128/64 inputs). Twenty trajectories run from scratch to 600 seconds and contain dependent 120/180/360-second checkpoints. All numerical targets, architectures, optimizer schedules, initialization labels, sensor/output grids, and accounting rules use the frozen follow-up engine. There is no new hyperparameter search or test-based rate selection; both rate-specific outcomes are reported. The existing DeepONet coarse/corrected comparison already matched rate 0.008 and is not rerun.',r'\begin{table}[ht]\centering\small',r'\caption{Matched-rate FNO results. ID errors are percentages (mean $\pm$ sample SD). Differences are paired corrected-minus-coarse percentage points; negative values favor correction.}',r'\begin{tabular}{rrllrr}\toprule LR & Cap (s) & Coarse ID & Corrected ID & Difference & Wins\\\midrule']
 for lr in p['rates']:
  for cap in [180,360,600]:
   a=groups[f'{cap}_{lr}_coarse32'];b=groups[f'{cap}_{lr}_phys32'];c=next(x for x in contrasts if x['budget']==cap and x['lr']==lr and x['metric']=='id')
   md.append(f"|{lr}|{cap}|{100*a['id_mean']:.3f}|{100*b['id_mean']:.3f}|{c['mean_pp']:.3f}|{c['negative_pairs']}/5|")
   tex.append(f"{lr} & {cap} & ${100*a['id_mean']:.3f}\\pm{100*a['id_sd']:.3f}$ & ${100*b['id_mean']:.3f}\\pm{100*b['id_sd']:.3f}$ & {c['mean_pp']:.3f} & {c['negative_pairs']}/5\\\\")
 tex+=[r'\bottomrule\end{tabular}\end{table}',r'\paragraph{Paired effects and uncertainty.} The following approximate paired Student-$t$ intervals use five training units and are not multiplicity-adjusted. Five non-tied pairs have minimum exact two-sided sign-test $p=0.0625$.']
 for c in contrasts:
  if c['budget'] not in [180,600]:continue
  lo,hi=c['t95_pp'];tex.append(f"Rate {c['lr']}, {c['budget']} s, {c['metric'].upper()}: corrected-minus-coarse {c['mean_pp']:.3f} percentage points, interval [{lo:.3f}, {hi:.3f}], correction favored in {c['negative_pairs']}/5 pairs.\\par")
 tex.append(r'\paragraph{Rate sensitivity.} The difference between the two rate-specific correction contrasts is reported without choosing a preferred rate after testing.')
 for c in interactions:
  if c['budget'] not in [180,600] or c['metric']!='id':continue
  lo,hi=c['t95_pp'];tex.append(f"At {c['budget']} s, the ID difference-of-differences (rate 0.008 minus 0.002) is {c['mean_pp']:.3f} percentage points, approximate interval [{lo:.3f}, {hi:.3f}].")
 tex.append(r'Matching the rate removes this particular configuration difference but does not isolate correction bias from its numerical cost and the resulting number of updates. These remain budget-policy comparisons, not a pure causal estimate of label fidelity. New evaluation inputs prevent pooling these runs with earlier cohorts. All adverse outcomes remain in the release.')
 md+=['','## 审计',json.dumps({k:v for k,v in audit.items() if k!='source_checks'},ensure_ascii=False,indent=2),'','全部逐种子ID/OOD结果见all_runs.csv；所有预算点、配对区间及学习率交互对比见summary.json。配对区间为小样本近似且未校正多重比较，不把相关检查点当独立重复。']
 (R/'results_report.md').write_text('\n'.join(md)+'\n')
 os.environ['MPLCONFIGDIR']=str(R/'mpl_config');sys.path.append(str(R.parents[1]/'new_directions/python_deps'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
 for ax,lr in zip(axes,p['rates']):
  caps=[180,360,600]
  for arm,color in [('coarse32','#225b87'),('phys32','#c07826')]:
   for seed in p['seeds']:
    vals=[next(100*r['id'] for r in rows if r['lr']==lr and r['arm']==arm and r['seed']==seed and r['budget']==cap) for cap in caps];ax.plot(caps,vals,color=color,alpha=.22,lw=.8)
   ax.plot(caps,[100*groups[f'{c}_{lr}_{arm}']['id_mean'] for c in caps],marker='o',label=LABELS[arm],color=color)
  ax.set(title=f'Fixed initial learning rate {lr}',xlabel='Construction budget cap (s)');ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
 axes[0].set_ylabel('ID relative L2 error (%)');fig.savefig(R/'matched_rate.png',dpi=180);fig.savefig(R/'matched_rate.pdf');plt.close(fig)
 tex +=[r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{images/matched_rate.pdf}\caption{Matched-rate FNO comparisons. Bold curves are means; faint curves are the five individual trajectories. Both panels share a vertical scale. Within-trajectory checkpoints are dependent.}\end{figure}']
 (R/'matched_rate_results.tex').write_text('\n\n'.join(tex)+'\n')
 paper=R.parent/'manuscript/paper';shutil.copy2(R/'matched_rate_results.tex',paper/'matched_rate_results.tex');shutil.copy2(R/'matched_rate.pdf',paper/'images/matched_rate.pdf')
 f=paper/'main.tex';s=f.read_text()
 if '\\input{matched_rate_results.tex}' not in s:s=s.replace('\\section{Discussion and limitations}','\\input{matched_rate_results.tex}\n\n\\section{Discussion and limitations}')
 f.write_text(s);e.atomic_json(R/'analysis_complete.json',{'complete':True,'pending':'Interpretation review, abstract/conclusion update, and compilation.'})
