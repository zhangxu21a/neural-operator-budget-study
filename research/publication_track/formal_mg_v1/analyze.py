"""Frozen analysis: all runs retained, paired training units, no test-based selection."""
import pathlib,json,csv,statistics,math,hashlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent
LABELS={'coarse8':'Coarse 8','coarse32':'Coarse 32','phys8':'Corrected 8','phys32':'Corrected 32','hf_online':'HF online','hf32':'HF reuse 32','hf128':'HF reuse 128'}
def main():
 import core
 import numpy as np
 from scipy.stats import t as student_t
 protocol=json.loads((ROOT/'protocol.json').read_text());selected=json.loads((ROOT/'selected.json').read_text());rows=[]
 for path in sorted((ROOT/'formal').iterdir()):
  if not path.is_dir():continue
  r=json.loads((path/'result.json').read_text());ev=json.loads((path/'evaluation.json').read_text())
  for cap in [120,180]:
   m=r['milestones'][str(cap)]
   rows.append({'arm':r['arm'],'seed':r['seed'],'budget':cap,'lr':r['lr'],'test':ev[str(cap)]['test']['relative_l2'],'ood':ev[str(cap)]['ood']['relative_l2'],'validation':m['validation'],'steps':m['steps_completed'],'selected_steps':m['selected_steps'],'hf':m['counts']['hf'],'decision_ready':m['decision_ready'],'nonfinite':r['nonfinite_loss'],**{k+'_seconds':v for k,v in m['timings'].items()}})
 assert len(rows)==70
 with (ROOT/'all_runs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 groups={}
 for cap in [120,180]:
  for a in protocol['arms']:
   rr=[r for r in rows if r['arm']==a and r['budget']==cap];assert len(rr)==5
   d={'n':5,'lr':selected[a]}
   for metric in ['test','ood','validation','steps','hf','hf_seconds','training_seconds','decision_ready']:
    v=[r[metric] for r in rr];d[metric+'_mean']=statistics.mean(v);d[metric+'_sd']=statistics.stdev(v)
   groups[f'{cap}_{a}']=d
 pairs=[]
 comparisons=[('phys32','coarse32'),('phys8','coarse8'),('phys32','phys8'),('hf32','hf_online'),('hf128','hf_online'),('phys32','hf32'),('phys32','hf128')]
 for cap in [120,180]:
  for left,right in comparisons:
   for metric in ['test','ood']:
    aa={r['seed']:r[metric] for r in rows if r['arm']==left and r['budget']==cap};bb={r['seed']:r[metric] for r in rows if r['arm']==right and r['budget']==cap};ds=[100*(aa[s]-bb[s]) for s in sorted(aa)];mean=statistics.mean(ds);sd=statistics.stdev(ds);half=student_t.ppf(.975,4)*sd/math.sqrt(5)
    wins=sum(d<0 for d in ds);p=min(1.,2*sum(math.comb(5,k) for k in range(min(wins,5-wins)+1))/32)
    pairs.append({'left':left,'right':right,'budget':cap,'metric':metric,'n':5,'per_seed_difference_pp':ds,'mean_difference_pp':mean,'sd_difference_pp':sd,'t95_interval_pp':[mean-half,mean+half],'left_wins':wins,'two_sided_sign_p':p,'caveat':'Exploratory comparisons; no multiplicity correction; t interval approximate at n=5; shared evaluation set; checkpoint endpoints correlated.'})
 sourcechecks={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in protocol['sources'].items()}
 runs=[json.loads(p.read_text()) for p in (ROOT/'formal').glob('*/result.json')]
 tuning=[json.loads(p.read_text()) for p in (ROOT/'tuning').glob('*/result.json')]
 audit={'formal_runs':len(runs),'tuning_runs':len(tuning),'budget_points':len(rows),'independent_training_units_per_arm':5,'source_hashes_match':sourcechecks,'all_sources_match':all(sourcechecks.values()),'all_caps_pass':all(r['decision_ready']<=r['budget'] for r in rows) and all(r['wall']<=r['budget'] for r in runs+tuning),'nonfinite_formal':sum(r['nonfinite_loss'] for r in runs),'nonfinite_tuning':sum(r['nonfinite_loss'] for r in tuning),'all_formal_initial_complete':all(r['initial_complete'] for r in runs),'max_hf_residual':max(r['max_hf_residual'] for r in runs+tuning),'max_hf_iterations':max(r['max_hf_iterations'] for r in runs+tuning),'equal_tuning_budget':{a:sum(r['budget'] for r in tuning if r['arm']==a) for a in protocol['arms']}}
 assert audit['all_sources_match'] and audit['all_caps_pass'] and audit['all_formal_initial_complete']
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'summary.json').write_text(json.dumps({'groups':groups,'paired':pairs},indent=2))
 md=['# 更强求解器下的正式短预算比较','', '21次60秒调参；35条180秒正式轨迹，五个独立训练池/优化种子，每条同时记录120秒预算点。每方法总调参180秒；不合并旧实验或单种子筛查。', '', '|方法|学习率|120秒ID%|180秒ID%（均值±种子SD）|180秒OOD%|HF标签|训练步|','|---|---:|---:|---:|---:|---:|---:|']
 tex=[r'\section{Replicated construction-budget comparison with efficient HF solves}',r'This prospective experiment fixes coefficient gain two and a 512-interval numerical grid, with a common 129-by-129 output. Coarse spaces have eight or 32 intervals. HF methods are independent of this choice and are not counted twice. Five independent channel-data pools and optimization seeds are paired across seven arms. All methods use the same 32 initial HF labels; the 128-label reuse arm purchases an additional 96 labels. The reuse arms then optimize only on their fixed HF set. A pool of 8192 inputs limits saturation. HF labels use MG-preconditioned CG at relative residual tolerance $10^{-8}$, including setup and geometry preparation within each run.',r'Each arm receives three 60-second trials at learning rates 0.002, 0.008, and 0.032; validation selects one configuration, giving 180 seconds of tuning per arm. Formal trajectories last 180 seconds and contain a dependent 120-second checkpoint. All runs execute serially with one CPU thread. New validation/ID/OOD sets contain 32/128/64 inputs; OOD Gaussian coefficients are multiplied by 1.25. These are short-construction-budget experiments, not evidence of long-horizon convergence. Data pools and testing protocol differ from the earlier experiment.',r'\begin{table}[ht]\centering\small',r'\caption{New MG-based experiment at 180 seconds: five independent training units. Errors are percentages (mean $\pm$ sample SD).}',r'\begin{tabular}{lrrrr}\toprule Method & LR & ID error & OOD error & HF labels\\\midrule']
 for a in protocol['arms']:
  x=groups['180_'+a];short=groups['120_'+a]
  md.append(f"|{LABELS[a]}|{x['lr']}|{100*short['test_mean']:.3f}|{100*x['test_mean']:.3f} ± {100*x['test_sd']:.3f}|{100*x['ood_mean']:.3f}|{x['hf_mean']:.1f}|{x['steps_mean']:.1f}|")
  tex.append(f"{LABELS[a]} & {x['lr']} & ${100*x['test_mean']:.3f}\\pm{100*x['test_sd']:.3f}$ & ${100*x['ood_mean']:.3f}\\pm{100*x['ood_sd']:.3f}$ & {x['hf_mean']:.1f}\\\\")
 tex +=[r'\bottomrule\end{tabular}\end{table}',r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{images/formal_mg.pdf}\caption{All five paired training units at the 180-second endpoint. Points show runs, horizontal segments show means. Different ID/OOD vertical scales are labeled.}\end{figure}',r'\paragraph{Paired effects.} Negative differences favor the first method. Differences below are percentage points of ID error at 180 seconds, with approximate paired Student-$t$ 95\% intervals over five training units; they are not multiplicity-adjusted.']
 md +=['','## 逐种子配对结果','负值表示左侧误差更小。区间为五个训练单位上的配对t近似95%区间，未校正多重比较；同轨迹预算点相关。','', '|比较（180秒ID）|均值差/百分点|近似95%区间|左侧胜出|','|---|---:|---|---:|']
 for p in pairs:
  if p['budget']!=180 or p['metric']!='test':continue
  lo,hi=p['t95_interval_pp'];md.append(f"|{p['left']} vs {p['right']}|{p['mean_difference_pp']:.3f}|[{lo:.3f}, {hi:.3f}]|{p['left_wins']}/5|")
  tex.append(f"{LABELS[p['left']]} versus {LABELS[p['right']]}: {p['mean_difference_pp']:.3f}, interval [{lo:.3f}, {hi:.3f}], with {p['left_wins']} of five pairs favoring the first arm.\\par")
 tex +=[r'The two-sided exact sign test cannot attain $p<0.05$ with only five non-tied pairs. Error intervals are conditional on the shared evaluation set. The tuning horizon is shorter than the formal horizon; selected configurations support a policy comparison, not attribution to a single mechanism. No negative or divergent trial is discarded. The result does not establish a universal allocation rule or a cross-PDE ranking.']
 md +=['','## 审计与限制',json.dumps({k:v for k,v in audit.items() if k!='source_hashes_match'},ensure_ascii=False,indent=2),'','两预算点不是独立重复；调参时限短于正式时限；每个训练单位同时改变参数池和优化种子，不能将方差拆为单独来源。五个配对样本即使全胜，双侧精确符号检验最小p=0.0625。没有用测试结果选择配置、删种子或换算法。']
 (ROOT/'results_report.md').write_text('\n'.join(md));(ROOT/'formal_results.tex').write_text('\n'.join(tex))
 import os,sys
 os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_config');sys.path.append(str(ROOT.parents[1]/'new_directions/python_deps'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
 for ax,metric,title in zip(axes,['test','ood'],['ID','OOD']):
  for i,a in enumerate(protocol['arms']):
   rr=sorted([r for r in rows if r['arm']==a and r['budget']==180],key=lambda r:r['seed']);v=[100*r[metric] for r in rr]
   ax.scatter(np.array([i]*5)+np.linspace(-.18,.18,5),v,color='#225b87',s=22);ax.plot([i-.22,i+.22],[statistics.mean(v)]*2,color='#a65d24',lw=2)
  ax.set_xticks(range(7),[LABELS[a] for a in protocol['arms']],rotation=30,ha='right');ax.set(title=title,ylabel='Relative L2 error (%)');ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
 fig.savefig(ROOT/'formal_mg.pdf');fig.savefig(ROOT/'formal_mg.png',dpi=180);plt.close(fig)
 paper=ROOT.parent/'manuscript/paper';shutil.copy2(ROOT/'formal_mg.pdf',paper/'images/formal_mg.pdf');shutil.copy2(ROOT/'formal_results.tex',paper/'formal_results.tex')
 main=paper/'main.tex';s=main.read_text()
 if '\\input{formal_results.tex}' not in s:s=s.replace('\\section{Discussion and limitations}','\\input{formal_results.tex}\n\n\\section{Discussion and limitations}')
 main.write_text(s)
 (ROOT/'analysis_complete.json').write_text(json.dumps({'complete':True,'manuscript_updated':str(main),'pending':'Human/model interpretation and final compile check'}))
if __name__=='__main__':main()
