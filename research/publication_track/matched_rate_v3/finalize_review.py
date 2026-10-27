from pathlib import Path
import json,math,statistics,hashlib,shutil,datetime
R=Path(__file__).resolve().parent;P=R.parent;paper=P/'manuscript/paper'
p=json.loads((R/'protocol.json').read_text());s=json.loads((R/'summary.json').read_text());audit=json.loads((R/'audit.json').read_text());load=json.loads((R/'execution_audit.json').read_text());source={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in p['sources'].items()};assert all(source.values())
raw={};records=[]
for seed,lr,arm in p['jobs']:
 d=R/'formal'/f'{arm}_lr{lr}_s{seed}';r=json.loads((d/'result.json').read_text());ev=json.loads((d/'evaluation.json').read_text());records.append(r)
 assert r['train_parameter_seed']==91000+seed and r['initial_complete'] and r['budget']==600 and r['wall']<=600
 for cap in p['checkpoints']:
  assert r['milestones'][str(cap)]['decision_ready']<=cap
  for metric,split in [('id','test'),('ood','ood')]:
   values=ev[str(cap)][split]['individual'];assert len(values)==(128 if metric=='id' else 64) and all(math.isfinite(x) for x in values)
   mean=statistics.mean(values);assert abs(mean-ev[str(cap)][split]['relative_l2'])<1e-12
   raw[seed,lr,arm,cap,metric]=100*mean
for cap in p['checkpoints']:
 for lr in p['rates']:
  for arm in p['arms']:
   for metric in ['id','ood']:
    v=[raw[seed,lr,arm,cap,metric]/100 for seed in p['seeds']];g=s['groups'][f'{cap}_{lr}_{arm}'];assert abs(statistics.mean(v)-g[metric+'_mean'])<1e-12 and abs(statistics.stdev(v)-g[metric+'_sd'])<1e-12
for c in s['contrasts']:
 ds=[raw[seed,c['lr'],'phys32',c['budget'],c['metric']]-raw[seed,c['lr'],'coarse32',c['budget'],c['metric']] for seed in p['seeds']];assert all(abs(a-b)<1e-10 for a,b in zip(ds,c['differences_pp']))
for c in s['interactions']:
 ds=[(raw[seed,.008,'phys32',c['budget'],c['metric']]-raw[seed,.008,'coarse32',c['budget'],c['metric']])-(raw[seed,.002,'phys32',c['budget'],c['metric']]-raw[seed,.002,'coarse32',c['budget'],c['metric']]) for seed in p['seeds']];assert all(abs(a-b)<1e-10 for a,b in zip(ds,c['differences_pp']))
assert len(records)==20 and len(load['runs'])==20 and len(raw)==160
independent={'formal_runs':20,'budget_points':80,'individual_metric_sets_checked':160,'frozen_sources_checked':len(source),'all_frozen_sources_match':True,'means_sds_paired_differences_interactions_recomputed':True,'all_caps_pass':True,'nonfinite_loss_runs':sum(r['nonfinite_loss'] for r in records),'flagged_load_runs':load['flagged_runs'],'no_runs_removed':True}
(R/'independent_review.json').write_text(json.dumps(independent,indent=2))
archive=R/'editorial_archive';archive.mkdir(exist_ok=True)
for f in [paper/'main.tex',P/'results_report.md']:
 target=archive/('main_before_matched_review.tex' if f.suffix=='.tex' else 'publication_report_before_matched_review.md')
 if not target.exists():shutil.copy2(f,target)
f=paper/'main.tex';m=f.read_text();m=m.replace('completed strong-solver, second-architecture and longer-horizon comparisons','completed strong-solver, second-architecture, longer-horizon and matched-learning-rate comparisons')
a=m.index('\\begin{abstract}');b=m.index('\\end{abstract}',a)+len('\\end{abstract}')
m=m[:a]+r'''\begin{abstract}
Finite construction budgets couple numerical label generation with neural-operator optimization, so label count or fidelity alone can misrepresent predictive value. We report separate Darcy studies with measured runtime, efficient multigrid-preconditioned HF solves, fixed-HF-label reuse baselines, independent training pools, and retained adverse outcomes. An FNO/DeepONet replication shows that method comparisons depend on architecture and construction horizon: corrected FNO targets overtake coarse supervision at a 600-second endpoint under separately selected learning rates, whereas the short-budget DeepONet comparison favors coarse supervision. A further balanced matched-rate study runs 20 fresh FNO trajectories across two target policies, two fixed rates, and five paired training pools. At 600 seconds and rate 0.008, corrected targets attain mean ID error 13.811\% versus 15.726\% for coarse targets, with all five pairs favoring correction. At rate 0.002, the corresponding means are 13.792\% and 13.994\%, with correction favored in only three pairs and an approximate paired interval spanning zero. The estimated ID rate interaction is imprecise and also has an interval spanning zero. Thus the long-budget advantage is not solely an artifact of comparing unequal rates, but its observed magnitude and consistency are not invariant to the rate. Fixed-HF reuse remains an important strong comparator in the separate baseline studies. These conditional cost--accuracy results do not establish convergence, cross-PDE superiority, or a universal allocation rule.
\end{abstract}'''+m[b:]
m=m.replace('The first two questions are addressed by completed prospective studies. A prespecified matched-rate follow-up addresses the third using both previously selected rates, without selecting a favorable rate after testing.','All three questions are addressed by completed prospective studies. The matched-rate follow-up uses both previously selected rates on fresh training and evaluation inputs, without selecting a favorable rate after testing.')
a=m.index('\\section{Discussion and limitations}');b=m.index('\\section{Conclusion}',a)
m=m[:a]+r'''\section{Discussion and limitations}
The matched-rate study qualifies the earlier horizon result. In the architecture/horizon cohort, separately selected FNO configurations changed their mean ordering between 180 and 600 seconds. That comparison used rate 0.002 for coarse targets and 0.008 for corrected targets. The new balanced study holds the rate fixed within each target comparison. At rate 0.008 and 600 seconds, correction has lower ID and OOD errors in all five pairs: mean paired differences are $-1.915$ and $-2.295$ percentage points. The observed long-budget benefit therefore cannot be attributed solely to comparing unequal rates in the earlier cohort.

The less favorable fixed-rate result is equally important. At rate 0.002 and 600 seconds, corrected/coarse ID means are 13.792\%/13.994\%; the paired difference is $-0.202$ percentage points, with approximate 95\% interval $[-1.444,1.040]$, and only three of five pairs favor correction. This is inconclusive, not an equivalence result or proof of no effect. At 180 seconds, the ID contrasts at both rates have intervals spanning zero. The ID difference-of-differences between rates at 600 seconds is $-1.712$ percentage points with approximate interval $[-3.526,0.101]$. These estimates suggest rate-sensitive magnitude and consistency, but do not establish a precise statistical interaction. All intervals are exploratory and not multiplicity-adjusted; five same-direction non-tied pairs still yield exact two-sided sign-test $p=0.0625$.

Matching the learning rate removes one configuration difference, not the full cost--bias mechanism. In this study, corrected targets consume about 271 seconds of numerical label work and permit roughly 790--795 updates at the 600-second cap, versus about 36 seconds and 1423--1431 updates for coarse targets. These are measured policy costs, including the common initial HF purchases; other overhead remains in wall time. They are consistent with an optimization-opportunity trade-off but do not isolate correction bias, sample exposure, and update count causally. An unconditional claim that correction always helps or always hurts is unsupported.

HF128 reuse remains a strong comparator in the earlier baseline and architecture/horizon cohorts. It beats online acquisition in all five ID pairs for both architectures at 180 seconds and for FNO at 600 seconds. Its advantage over corrected FNO shrinks at the longer endpoint, with one corrected run winning and the paired interval including zero. For DeepONet, HF128 and coarse supervision have nearly equal mean ID errors, while coarse supervision has lower mean OOD error. HF32 also beats online acquisition in the DeepONet cohort, unlike the earlier FNO result. The new matched-rate cohort contains no HF128 arm; its absolute errors must not be used to rank HF128 across different evaluation sets.

The observed comparisons depend jointly on numerical cost, correction bias, cache saturation and optimization opportunity. DeepONet's cheaper updates alter the cost structure, but the 22,560-parameter DeepONet and 532,993-parameter FNO are not capacity matched. Their errors do not constitute a controlled architecture ranking. Each study uses its own training and reference inputs; cohorts are not pooled as independent replicates. Within-trajectory checkpoints establish finite-horizon comparisons, not independent repeats or a precisely located crossover time.

Equal allocated tuning time does not remove hyperparameter-selection uncertainty. The earlier prospective studies use one tuning pool, three 60-second trials per method and small shared validation sets; architecture-specific rate grids differ. The matched-rate study performs no new tuning and reports both fixed rates without test-based selection. No long-horizon retuning is performed. A 600-second endpoint is not proof of convergence, and five training units cannot support precise small-effect claims.

The scope remains one PDE family, projected fixed-grid outputs and CPU execution. The multigrid solver is a validated local implementation, not a claim of numerical optimality. The appendix's MPS probes do not establish end-to-end GPU performance. A held-out numerical regime or second PDE would be required before broad allocation guidance. These limits are not removed by more elaborate plots or selectively retaining favorable seeds.

'''+m[b:]
a=m.index('\\section{Conclusion}');b=m.index('\\section*{Reproducibility and declarations}',a)
m=m[:a]+r'''\section{Conclusion}
Strong numerical and HF-reuse baselines change the apparent value of multifidelity supervision. Across the completed studies, comparisons vary with architecture, construction horizon and optimization settings. The matched-rate FNO experiment retains a long-budget correction advantage at rate 0.008, while the rate-0.002 comparison is inconclusive and its mean gap is small. This both rules out a purely unequal-rate explanation for the observed advantage and rejects a rate-invariant superiority claim. Earlier unfavorable correction, HF32 and gradient-proxy results remain part of the evidence. Measured construction costs and complete paired reporting support conditional comparisons; they do not establish a universal winner, converged ranking or validated allocation law.

'''+m[b:]
f.write_text(m)
interpretation=r'''\subsection{Matched-rate interpretation and execution audit}
At the 600-second cap, the correction advantage is consistent across all five ID/OOD pairs at rate 0.008, but not across all ID pairs at rate 0.002. The latter small mean difference is not an equivalence finding. The imprecise ID interaction interval spans zero; neither that interval nor the unadjusted OOD interaction should be used to assert a universal rate interaction.

All twenty runs and eighty budget points passed budget and finite-value checks. All 28 frozen source hashes matched at execution and independent review. The review recomputed means, sample standard deviations, paired differences and rate interactions from all 160 per-split metric sets. No competing high-load Python process was detected by the five-second monitor; this does not certify complete host or thermal isolation. No seed, rate, adverse outcome or non-finite flag was removed or replaced. Common reference generation and post-run testing remain separately accounted expenses.
'''
(paper/'matched_rate_interpretation.tex').write_text(interpretation)
m=f.read_text()
if '\\input{matched_rate_interpretation.tex}' not in m:m=m.replace('\\input{matched_rate_results.tex}','\\input{matched_rate_results.tex}\n\\input{matched_rate_interpretation.tex}')
f.write_text(m)
notes='''# 匹配学习率实验：最终审核

20条600秒轨迹于2026-10-27 14:39（Asia/Seoul）全部完成；80个相关预算点，五个独立训练池/优化种子。两档学习率均保留，无新增调参，无测试驱动的学习率选择。

|600秒结果|粗标签ID%|校正ID%|校正胜出|配对差/百分点及近似95%区间|
|---|---:|---:|---:|---|
|学习率0.002|13.994|13.792|3/5|-0.202，[-1.444, 1.040]|
|学习率0.008|15.726|13.811|5/5|-1.915，[-3.362, -0.467]|

学习率0.008下OOD误差也在五个配对中均有利于校正，平均差为-2.295个百分点；0.002下OOD平均差-0.542，近似区间含零。180秒下两档ID配对区间均含零。

600秒ID的学习率交互估计为-1.712个百分点，近似区间[-3.526, 0.101]仍含零。因此可以说观察到的优势大小和一致性依赖所测学习率，但不能声称已经精确证实统计交互。五个全胜配对的精确双侧符号检验最小p=0.0625，所有区间均未做多重比较校正。

结论：长预算下的校正优势并非完全来自此前两种方法使用不同学习率，但它也不是在所有学习率上都稳定占优。0.002下的接近均值不能当作等价性证明。本轮无HF128臂，不能跨不同测试集用绝对误差把它与旧HF128结果重新排名。

所有预算、初始标签、有限值及28个冻结源文件检查通过。独立复核从逐样本记录重算160组ID/OOD指标集合、组均值/标准差、配对差与交互。未观测到外部高负载Python采样；不宣称完全排除整机或热状态影响。

论文摘要、讨论、结论及新结果解释已按这些限制重写，全部不利结果和旧机制结果保留。没有新增训练或删除种子。
'''
(R/'interpretation.md').write_text(notes)
f=R/'paper_storyline.md';t=f.read_text().replace('运行前整理完成；结果审核与最终摘要/讨论/结论修订待全部轨迹结束。','20条轨迹已全部完成并审核。核心结论已收窄：0.008下长预算校正优势仍在，0.002下接近且证据不足；不宣称学习率无关优势或已证实精确交互。摘要/讨论/结论已据此修订。');f.write_text(t)
f=P/'results_report.md';t=f.read_text();heading='# 最新进展：匹配学习率对照已完成'
if not t.startswith(heading):f.write_text(heading+'\n\n2026-10-27。20条正式轨迹、80个相关预算点全部完成，28个冻结源码哈希及预算检查通过，无非有限损失，无观测到的外部高负载Python采样。\n\n600秒下，0.008学习率的校正/粗标签ID误差为13.811%/15.726%，五个配对均有利于校正；0.002下为13.792%/13.994%，仅3/5配对有利于校正，配对区间含零。不能宣称校正普遍占优，也不能把接近均值写成等价。\n\n完整数据见`matched_rate_v3/results_report.md`、`summary.json`、`all_runs.csv`，审核解释见`matched_rate_v3/interpretation.md`。论文主线、摘要、讨论和结论已更新；以下保留此前各阶段报告。\n\n---\n\n'+t)
f=P/'README.md';t=f.read_text().replace('## 正在运行：匹配学习率对照','## 已完成：匹配学习率对照').replace('当前任务已配置完成检查，跑完后审核全部结果、更新论文并编译。','20条轨迹已完成且独立审核；摘要/讨论/结论已更新。0.008下长预算校正优势保留，0.002下差异不确定。详见该目录`interpretation.md`。');f.write_text(t)
f=P/'WORK_STATE.json';w=json.loads(f.read_text());w['status']='matched_rate_v3_complete_editorial_updated_pending_compile';w['active_processes']=[];w['completed']['matched_rate_v3']=independent;w['active_study']['status']='complete';w['active_study']['editorial_followup']='Interpretation complete; final compilation and release checks pending.';w['next_required']=['Submission-focused evidence and related-work review','Verify author declarations and public reproducibility archive before submission','Additional PDE/regime only if broader generalization is claimed'];f.write_text(json.dumps(w,indent=2,ensure_ascii=False))
print(json.dumps(independent,indent=2))
