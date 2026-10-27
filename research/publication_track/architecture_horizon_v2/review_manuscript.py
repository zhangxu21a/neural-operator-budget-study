from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent;P=R.parent;paper=P/'manuscript/paper';f=paper/'main.tex';s=f.read_text()
assert json.loads((R/'completion.json').read_text())['complete']
archive=R/'editorial_archive';archive.mkdir(exist_ok=True)
if not (archive/'main_before_review.tex').exists():shutil.copy2(f,archive/'main_before_review.tex')
s=s.replace('Research draft with a completed prospective, replicated strong-solver comparison. Claims remain limited to one PDE family, one architecture, and the stated CPU construction budgets.','Research draft with completed strong-solver, second-architecture and longer-horizon comparisons. Claims remain limited to one PDE family, the tested FNO/DeepONet configurations, and the stated CPU construction budgets.')
a=s.index('\\begin{abstract}');b=s.index('\\end{abstract}',a)+len('\\end{abstract}')
s=s[:a]+r'''\begin{abstract}
High-fidelity label count alone does not determine neural-operator accuracy under a finite construction budget. We measure numerical label generation, coarse-label correction, and network optimization in separately analyzed Darcy studies. An initial training-dominated experiment does not establish independent benefit for a last-layer gradient acquisition proxy. A strong-solver FNO study uses multigrid-preconditioned conjugate gradients, equal-budget tuning and five independent training pools; at 180 seconds, fixed reuse of 128 HF labels outperforms online acquisition, while one-cycle correction of 32-interval coarse targets does not improve prediction. A prospective follow-up adds 27 tuning trials and 45 trajectories with new training and evaluation inputs, covering DeepONet at 180 seconds and FNO to 600 seconds. Within the new FNO trajectories, corrected-label mean ID error improves from 19.195\% at 180 seconds to 14.222\% at 600 seconds, reversing its mean ordering against uncorrected coarse supervision (18.224\% and 15.805\%). Fixed HF128 reuse still has the lowest tested mean FNO error at 600 seconds (13.453\%). For DeepONet at 180 seconds, correction is worse than coarse supervision (40.470\% versus 31.268\%); HF128 reuse has similar ID error (31.105\%) and slightly worse OOD error than coarse supervision. The value of correction and label reuse therefore depends on construction horizon and architecture. These bounded empirical comparisons do not establish convergence, a universal allocation law, or cross-PDE superiority.
\end{abstract}'''+s[b:]
s=s.replace('A validated allocation rule across tasks remains a prospective objective.','A new-pool follow-up further shows that correction-versus-coarse ordering changes with the FNO construction horizon, while the analogous DeepONet comparison remains unfavorable to correction. A validated allocation rule across tasks remains a prospective objective.')
s=s.replace('The held-out sets contain 256 in-distribution','For the original mechanism study, the held-out sets contain 256 in-distribution')
a=s.index('\\section{Discussion and limitations}');b=s.index('\\section{Conclusion}',a)
s=s[:a]+r'''\section{Discussion and limitations}
The follow-up changes the interpretation of the short-budget negative result. Within the same new FNO trajectories, corrected 32 has mean ID error 19.195\% at 180 seconds versus 18.224\% for coarse 32, winning only two of five pairs. At 600 seconds it attains 14.222\% versus 15.805\%, winning all five pairs. The mean ordering therefore reverses with the horizon. This does not locate a precise crossover time or establish an isolated causal effect of correction: selected learning rates differ, and optimization trajectories differ. All five non-tied pairs still have minimum exact two-sided sign-test $p=0.0625$.

HF128 reuse remains a strong comparator: it beats online acquisition in all five ID pairs for both architectures at 180 seconds and for FNO at 600 seconds. Its advantage over corrected FNO shrinks at the longer endpoint; correction wins one of five pairs at 600 seconds, and the approximate paired interval includes zero. It is consequently inappropriate to claim uniform superiority over correction. For DeepONet, HF128 and coarse supervision have nearly equal mean ID errors, while coarse supervision has lower mean OOD error (35.915\% versus 37.130\%). The earlier poor HF32 result also does not transfer universally: HF32 beats online acquisition in all five DeepONet ID pairs. All of these favorable and unfavorable outcomes are retained.

The observed differences are consistent with joint effects of solve cost, correction bias, cache saturation and optimization opportunity, rather than a rule that one should always buy labels, always correct them, or always train longer. DeepONet's much cheaper updates change the relative cost structure. However, the 22,560-parameter DeepONet and 532,993-parameter FNO are not capacity matched; their accuracy difference is not a controlled architecture ranking. The follow-up uses new references and training pools, so absolute changes from the earlier experiment cannot be attributed solely to horizon or architecture. Its within-trajectory checkpoints provide the horizon comparison.

Equal allocated tuning time does not eliminate selection uncertainty. Each method receives three 60-second trials using one tuning pool and a small shared validation set; architecture-specific rate grids differ and no long-horizon retuning is performed. The 600-second endpoint is a longer finite-budget test, not proof of convergence. Five training units cannot establish precise small effects, and correlated checkpoints do not increase the independent sample size. Approximate paired intervals are exploratory and not multiplicity-adjusted.

The scope remains one PDE family, fixed-grid projected outputs and CPU execution. The multigrid baseline is a validated local implementation, not numerical optimality. The appendix's MPS probes do not establish end-to-end GPU performance. A matched-rate check would help separate selected-configuration effects from the correction mechanism; a held-out numerical regime or second PDE would be needed before claiming broad allocation guidance. Neither is supplied by adding plots or selectively retaining favorable seeds.

'''+s[b:]
a=s.index('\\section{Conclusion}');b=s.index('\\section*{Reproducibility and declarations}',a)
s=s[:a]+r'''\section{Conclusion}
The gradient proxy does not establish independent benefit in the original training-dominated experiment. Strong HF-reuse baselines alter the interpretation of multifidelity performance, and the new architecture/horizon replication shows why short-budget rankings should not be generalized. Corrected FNO supervision overtakes coarse supervision at the 600-second endpoint, whereas the DeepONet comparison at 180 seconds favors coarse supervision. HF128 reuse remains competitive across the tested settings but does not uniformly dominate every comparator and metric. Measured construction costs, explicit correction bias, architecture-dependent optimization cost and retained adverse results are necessary to assess multifidelity value. The evidence supports conditional comparisons, not a universal winner or a validated allocation law.

'''+s[b:]
f.write_text(s)
notes='''# 第二架构与长预算复核：审核结论

全部27次调参、45条正式轨迹完成；共130个相关预算点。完成时间：2026-10-27 05:55（Asia/Seoul）。预算、冻结源码、初始标签检查均通过，无非有限损失；未检测到外部高负载Python采样。未删种子或替换不利结果。

- FNO同轨迹比较出现均值排序反转：180秒校正32为19.195%，粗32为18.224%；600秒变为14.222%与15.805%。600秒五个配对均有利于校正。不能再写“校正32始终劣于粗32”。
- FNO 600秒固定128标签为13.453%，仍低于在线购数15.884%，五个ID配对方向一致。但与校正32的差距缩小，校正有1个种子胜出；不能写固定标签始终优于校正。
- DeepONet 180秒：校正32为40.470%，差于粗32的31.268%，五个ID配对方向一致。固定128为31.105%，与粗32平均ID接近；OOD上粗32更低（35.915% vs37.130%）。
- 固定32标签在DeepONet上优于在线购数，和前一轮FNO结果方向不同，不能推广“32标签一定不足”。

下一步优先完成论文证据整合与投稿前差距审查。若追加实验，首选针对校正/粗标签比较的匹配学习率验证，厘清配置选择与机制效果；扩大到第二PDE应服务于明确的外推主张，不应无目的增加训练。

本轮仍只有一个PDE家族、两种容量不同的架构及有限CPU预算。600秒不代表收敛；五个全胜配对的精确双侧符号检验p最小为0.0625，不把描述性方向一致写成普遍显著性。
'''
(R/'interpretation.md').write_text(notes)
f=P/'results_report.md';old=archive/'publication_report_before_review.md'
if not old.exists():shutil.copy2(f,old)
prefix='# 最新进展：第二架构与长预算复核已完成\n\n2026-10-27。27次调参、45条正式轨迹全部结束，完整结果见 `architecture_horizon_v2/results_report.md`，审核解释见 `architecture_horizon_v2/interpretation.md`。\n\n**关键修正：FNO校正32在600秒时超过粗32，短预算结论不能推广到长预算；DeepONet的短预算校正仍落后。** 固定128标签是强基线，但并非在全部方法、指标和种子上统一占优。论文摘要、讨论与结论已据此改写；以下保留上一阶段记录。\n\n---\n\n'
if not f.read_text().startswith('# 最新进展：'):f.write_text(prefix+f.read_text())
w=json.loads((P/'WORK_STATE.json').read_text());w['status']='architecture_horizon_v2_complete_editorial_updated_pending_compile';w['active_processes']=[];w['completed']['architecture_horizon_v2']={'tuning_runs':27,'formal_runs':45,'dependent_budget_points':130,'all_caps_pass':True,'all_frozen_sources_match':True,'nonfinite_runs':0,'observed_competing_load_runs':0};w['next_required']=['Complete submission-focused evidence and related-work review','Consider matched-learning-rate correction/coarse check before mechanism claims','Verify author declarations and public reproducibility archive before submission'];w['active_study']['status']='complete';(P/'WORK_STATE.json').write_text(json.dumps(w,indent=2,ensure_ascii=False))
f=P/'README.md';s=f.read_text().replace('## 正在运行：第二架构与长预算复核','## 已完成：第二架构与长预算复核').replace('`architecture_horizon_v2/` 已启动：','`architecture_horizon_v2/` 已完成：').replace('完整统计在全部运行结束后生成。此前结果保留。','完整统计与审核解释已生成，见该目录`results_report.md`与`interpretation.md`。此前结果保留。');f.write_text(s)
print('Manuscript and work state updated without changing frozen experimental sources.')
