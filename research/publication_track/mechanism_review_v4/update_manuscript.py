from pathlib import Path
import json,csv,shutil
P=Path(__file__).resolve().parents[1];R=P/'mechanism_review_v4';paper=P/'manuscript/paper'
notes='''# 文献定位核实与贡献对照（2026-10-27，Asia/Seoul）

本轮核查针对正文主张，不是穷尽性系统综述。已读证据与未能取得的证据分开；不将未知的预算包含项判为缺失。下表不宣称本文首次提出成本分析或同时计入获取与训练。

|工作与可追溯来源|本轮核实位置和成本口径|本文可区分之处|不能声称什么|
|---|---|---|---|
|[de Hoop et al., 2022](https://arxiv.org/html/2203.13181v3)，[出版信息](https://www.global-sci.com/jml/article/view/13181)|§4.3、§4.3.1以训练数据量作为训练成本代理；§4.3.6讨论优化与泛化。出版商确认JML 1(3),299–341，DOI 10.4208/jml.220509。|本文记录单次受限构建过程的标签、训练及选择墙钟开销。|不能声称首次研究成本—精度或首次发现优化影响。|
|[Setinek et al., 2025](https://arxiv.org/html/2511.01830v1)|§1将数据轴拆成生成预算（core hours）与保真度组成；明确假设模型大小和训练计算非限制因素。|本文让数值获取和网络训练竞争同一个构建时限。|不拟合缩放律，不宣称首次预算约束下多保真组合。|
|[Rowbottom et al., v2](https://arxiv.org/html/2505.12940v2)，[作者代码](https://github.com/JRowbottomGit/mlmc_optim)|§2.3为配对梯度的望远镜估计；§3.1图6横轴为累计训练时间，其他图报告epoch时间；三种子，A100。|本文的数值目标校正不是MLMC梯度校正；保留独立的获取时间账本和HF复用对照。|本次没有复现其全流程；不能说本文超过该算法。|
|[Huang & Stuart, 2026](https://arxiv.org/html/2609.24021v1)|摘要、§1及§2.3明确为训练后多查询极限，固定离线成本完全摊销。|本文是有限构建阶段的CPU比较，公共参考集另列。|不能把构建结果解释为生命周期或推理成本优势。|
|[Xu et al., 2026：作者机构摘要](https://scholars.duke.edu/publication/1697477)，[出版商预览](https://www.sciencedirect.com/science/article/pii/S0045782525009478)|可核实物理信息评分、贝叶斯自适应采样、多PDE/FNO/DeepONet及等计算预算的声明；完整计时公式、调参口径和可验证仓库地址尚未取得。|目前仅能明确：本文不提出或复现该OED算法，而研究所测数值目标与复用策略的有限时限表现。|不得声称该文只计标签数量、排除训练成本，或本文首次联合计时。|
|[Faza et al., v1 HTML](https://arxiv.org/html/2608.04708v1)|摘要承认多保真方案并非都优于HF；§3表1/3/6给出200 epochs。HTML题名与摘要页题名不同，保留版本标识。|本文增加所测数值目标下的墙钟账本和匹配初始学习率比较。|负结果与保真度失配本身不是新发现；不把POD玩具失配描述成真实不同方程求解。|

## 可保留的贡献

1. 一组可审计、分开报告的Darcy构建预算比较：强HF求解器、HF标签反复训练、粗/校正目标及验证选择成本均可查。
2. 对简单优越性主张的边界检验：强HF复用常优于在线获取；校正收益随所测预算/架构/初始学习率条件变化，全部不利结果保留。
3. 本轮新增的事后诊断：把累计支出、完成更新与选中模型的更新数区分，并说明墙钟衰减与获取速度的耦合；不声称已识别因果效应。

## 未解决的文献问题

已尝试精确题名、作者+arXiv、作者/题名+GitHub、PDF搜索，核对作者机构页面及出版商入口。Xu全文抓取返回403；搜索中的其他论文代码链接不能当作其代码。未发送作者邮件、未绕过付费访问。其完整方法/附录/代码与预算公式仍是投稿前定位核查项。这个缺口需要文献访问，不需要新增训练。当前正文撤去任何依赖该缺口的排他性创新声明。

## 检索与版本记录

以上链接均在2026-10-27核查。定位以原文小节/图表为准，不用网页抓取行号作为论文页码。Rowbottom审阅对象为arXiv v2，期刊版沿用既有书目信息，不假定两版逐字相同。既有文献清单保留历史状态，由本文件覆盖本轮已核实项。
'''
(P/'literature/contribution_review_20261027.md').write_text(notes)
tex=r'''\subsection{Relation to existing cost studies}
\begin{table}[htbp]\centering\small
\caption{Verified comparison axes in related work. These are scope distinctions, not a claim that an unverified accounting item is absent. The Xu row remains an explicit access limitation.}
\begin{tabular}{p{.19\linewidth}p{.36\linewidth}p{.36\linewidth}}\toprule
Work & Verified cost or experimental axis & Relation to this study\\\midrule
De Hoop et al.\ \cite{dehoop2022} & Data volume as training-cost proxy; optimization also discussed (Secs.\ 4.3.1, 4.3.6). & We measure construction wall time; cost--accuracy analysis is not new.\\
Setinek et al.\ \cite{setinek2025} & Data-generation core hours and fidelity mix; training compute assumed non-limiting (Sec.\ 1). & Label generation and optimization share our finite deadline.\\
Rowbottom et al.\ \cite{rowbottom2026} & MLMC gradient corrections; cumulative training and epoch timings (preprint Sec.\ 3.1). & Numerical target correction is distinct from gradient telescoping; no head-to-head superiority claim.\\
Huang and Stuart \cite{huang2026} & Per-query costs after amortization of fixed offline expenses (Sec.\ 2.3). & Our construction-stage comparison is not a lifecycle or inference-speed comparison.\\
Xu et al.\ \cite{xu2026} & Abstract reports OED sampling under equivalent compute; detailed timing inclusions remain unverified. & No claim of first joint accounting or superiority over OED.\\
Faza et al.\ \cite{faza2026} & Fidelity mismatch and 200-epoch training protocols (v1 Tables 1, 3, 6). & Adverse multifidelity outcomes alone are not novel; we add measured deadline comparisons.\\\bottomrule
\end{tabular}\end{table}
Our contribution is the documented set of conditional comparisons and their accounting diagnostics. It is not a new numerical correction, a new optimization algorithm, or an exhaustive benchmark of acquisition methods. Full timing and tuning definitions for the closest OED study remain a pre-submission literature-verification item; missing access cannot establish novelty.
'''
(paper/'contribution_comparison.tex').write_text(tex)
mech=r'''\section{Post-hoc cost and optimization-opportunity audit}
This analysis reads the existing records of 100 formal trajectories and 280 dependent budget endpoints across the three prospective cohorts. It neither trains new models nor recomputes test predictions. Cohorts remain separate. Replaying validation-selection bookkeeping agrees with all archived selected checkpoints; the 305 inspected records, protocols and computation sources are unchanged.

\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{images/cost_opportunity_error.pdf}
\caption{Cost, available optimization and retained-model error in the matched-rate cohort at the 600-second cap. (a) Mean exclusive cost categories; black dots mark each run's actual decision-ready time and the dotted line is the cap. Labels include the common 32 HF purchases; other time includes input construction, validation, checkpointing and residual overhead. (b) Each connected pair shows completed updates (filled circles) and updates of the validation-selected checkpoint (open squares). (c,d) ID errors of those selected checkpoints, plotted against their own update counts; gray segments join the same training seed across target policies, not successive training states. All five seeds at both rates are shown. Update count is an outcome of each policy, not a randomized causal exposure.}\label{fig:cost-opportunity}
\end{figure}

\begin{table}[htbp]\centering\small
\caption{Matched-rate cohort at 600 seconds; means over five training units. Purchased targets are distinct cached labels, not exact sample exposures. Completed and selected updates have different meanings.}
\begin{tabular}{llrrrr}\toprule
Target & Initial rate & Label time (s) & Purchased & Completed & Selected\\\midrule
Coarse & 0.002 & 36.4 & 6164.0 & 1422.8 & 1112.2\\
Corrected & 0.002 & 271.0 & 4428.6 & 795.0 & 688.4\\
Coarse & 0.008 & 36.2 & 6180.8 & 1430.8 & 1372.4\\
Corrected & 0.008 & 270.8 & 4411.0 & 790.4 & 695.6\\\bottomrule
\end{tabular}\end{table}

\paragraph{Spent computation is not retained-model age.}
The existing error tables evaluate the best validation-selected checkpoint available by each cap, not necessarily the last optimizer iterate. At 600 seconds, 15 of the 20 matched-rate runs select an earlier iterate. Coarse targets at rate 0.002 spend 1422.8 updates on average but retain checkpoints averaging 1112.2 updates; the corresponding corrected counts are 795.0 and 688.4. All completed work remains charged. It would be incorrect to attribute the selected model's error to all updates spent, or to infer that the later updates worsened test error: unsaved later states do not have archived test predictions.

\paragraph{Label purchases are not sample exposures.}
Mean distinct label purchases are approximately 6164--6181 for coarse and 4411--4429 for corrected targets, from an 8192-input pool, including the initial 32 HF labels. These caches are not exhausted at the endpoint. Cache counts can include a boundary acquisition without an ensuing optimizer step; batch draws can repeat across updates and are deduplicated within a batch. The logs therefore support purchased-label coverage, not exact cumulative or unique training exposure. Reconstructing an exact exposure sequence from update counts alone is unwarranted.

\paragraph{A shared wall-time schedule is not a shared update-index schedule.}
The frozen engine sets $\eta(t)=\eta_0/(1+t/60)$ immediately before an optimizer update, after obtaining targets. Thus numerical work consumes time on the learning-rate clock as well as reducing the number of updates. Equal initial rates match this wall-time policy, not $\eta$ as a function of update index. Figure~\ref{fig:schedule-audit} evaluates the known schedule at recorded assessment-completion times; these values are diagnostic proxies, not logged per-update learning rates. Exact optimizer-step timestamps and learning-rate integrals are unavailable. This coupling is an implementation fact, while its contribution to the error difference remains unidentified.

\paragraph{The wider high-rate gap does not demonstrate improvement of corrected models.}
In a post-hoc within-policy contrast at 600 seconds, increasing the initial rate from 0.002 to 0.008 changes mean ID error by $+1.731$ percentage points for coarse targets and $+0.019$ for corrected targets. Approximate paired 95\% intervals are $[-0.601,4.064]$ and $[-2.375,2.413]$, respectively. Both include zero and are not multiplicity-adjusted. The corresponding OOD changes are $+1.562$ and $-0.191$ points, also with intervals containing zero. Descriptively, the larger corrected-versus-coarse mean gap at 0.008 accompanies a higher coarse mean rather than a lower corrected mean. This neither proves rate robustness nor licenses selecting a rate using test performance. The predeclared interaction comparison remains imprecise.

\paragraph{What the records cannot identify.}
The audit supports a measured policy trade-off, not a decomposition of error into numerical bias, sample coverage, update opportunity and scheduler effects. A causal claim about the source of the correction advantage would require new controlled training. A separate fixed-update diagnostic would need common input sequences, validation opportunities and an update-index schedule, with target policy varied and actual runtime still reported. A claim specifically about wall-time decay would additionally require a prespecified scheduler intervention. Such diagnostics would answer different questions from the original fixed-wall-time comparisons and cannot replace them. They were not run here. The present bounded empirical claims do not require a favorable new result; cross-PDE guidance, an optimal allocation rule or a causal mechanism would require additional evidence.
'''
(paper/'mechanism_review.tex').write_text(mech)
schedule=r'''\section{Wall-time learning-rate diagnostic}
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{images/schedule_diagnostic.pdf}
\caption{The frozen decay function evaluated at recorded validation-assessment completion times, divided by the initial rate and plotted against completed updates. Each curve is one of five matched-rate runs; solid/dashed curves denote coarse/corrected targets. These are sparse assessment-time proxies, not exact optimizer learning-rate trajectories, and lines interpolate only for display. No learning-rate exposure integral or causal effect is estimated.}\label{fig:schedule-audit}
\end{figure}
'''
(paper/'schedule_diagnostic.tex').write_text(schedule)
for name in ['cost_opportunity_error.pdf','schedule_diagnostic.pdf']:shutil.copy2(R/name,paper/'images'/name)
f=paper/'main.tex';s=f.read_text()
s=s.replace('Their precise accounting and implementation require comparison before any novelty claim about budget-aware acquisition.','Their precise accounting and implementation require comparison before any novelty claim about budget-aware acquisition.')
if '\\input{contribution_comparison.tex}' not in s:s=s.replace('\\section{Problem and accounting}','\\input{contribution_comparison.tex}\n\n\\section{Problem and accounting}')
if '\\input{mechanism_review.tex}' not in s:s=s.replace('\\section{Discussion and limitations}','\\input{mechanism_review.tex}\n\n\\section{Discussion and limitations}')
if '\\input{schedule_diagnostic.tex}' not in s:s=s.replace('\\input{legacy_mechanism_appendix.tex}','\\input{legacy_mechanism_appendix.tex}\n\\input{schedule_diagnostic.tex}')
s=s.replace('Fixed-HF reuse remains an important strong comparator in the separate baseline studies.','A post-hoc ledger audit distinguishes spent updates from the earlier validation-selected models and identifies coupling between label-generation time and wall-time learning-rate decay; its causal effect remains unresolved. Fixed-HF reuse remains an important strong comparator in the separate baseline studies.')
s=s.replace('Matching the learning rate removes one configuration difference, not the full cost--bias mechanism.','Matching the initial learning rate and wall-time decay policy removes one configuration difference, not the full cost--bias mechanism.')
s=s.replace('These are measured policy costs, including the common initial HF purchases; other overhead remains in wall time.','These are completed, charged updates rather than necessarily the age of the validation-selected model (Figure~\\ref{fig:cost-opportunity}). The measured policy costs include the common initial HF purchases; other overhead remains in wall time.')
s=s.replace('Earlier unfavorable correction, HF32 and gradient-proxy results remain part of the evidence.','The post-hoc accounting audit also shows why cumulative work cannot be equated with the selected model\'s update count, and why a common wall-time schedule does not equalize optimization exposure. These diagnostics do not causally explain the error gaps. Earlier unfavorable correction, HF32 and gradient-proxy results remain part of the evidence.')
s=s.replace('Journal of Machine Learning 1 (2022), 299--341. arXiv:2203.13181.','Journal of Machine Learning 1(3) (2022), 299--341. \\url{https://doi.org/10.4208/jml.220509}. arXiv:2203.13181.')
f.write_text(s)
report='''# 文献与机制复核结论

本轮没有训练、没有重新评估测试集、没有按测试表现选学习率。读取三批100条正式轨迹、280个相关预算点，逐项核查成本账本与验证选择；305个输入文件哈希保持不变。各批数据没有合并作独立重复。

## 最重要的新增发现

- 600秒匹配学习率实验20个模型中，15个保留了早于最后更新的检查点。粗标签0.002完成1422.8次更新但选中模型平均1112.2次；校正为795.0与688.4次。预算支出必须全部计入，误差只能对应选中模型。
- 校正目标约花271秒生成标签，粗标签约36秒；训练约292秒与523–524秒。缓存的独立标签约4411–4429与6164–6181，均未填满8192输入池。缓存数不等于样本接触量。
- 学习率在获取目标后按墙钟时间设置，因此初始学习率相同不等于按更新次数有相同衰减。日志只有稀疏验证完成时间，没有每步时间与学习率积分；辅助图明确标成评估时刻的衰减函数值。
- 0.008相对0.002的ID均值变化：粗标签+1.731个百分点，校正+0.019。两者配对区间均含零。大均值差主要表现为粗标签在高学习率下均值更高，不能描述为高学习率让校正方法变好，也不能据此证明鲁棒性。

## 文献定位

见[贡献对照表](../literature/contribution_review_20261027.md)。本文定位为可审计的有限构建预算实证比较，不声称新算法、首次成本分析、首次联合计时或超过OED/MLMC方法。Xu等全文/代码的详细预算口径仍未取得，保留投稿前文献核查项；这不是补实验可以解决的缺口。

## 是否必须补实验

**对当前收窄后的描述性论文：没有发现必须通过追加训练才能修复的结果错误，不建议为了得到有利结果开新一轮实验。** 这不等于已经达到目标期刊的录用要求。贡献的新颖程度、最接近工作的完整核对和公开复现交付仍需完成。

**若要主张因果机制：必须补受控实验。** 现有记录不能分离目标偏差、训练机会、样本覆盖、检查点选择和时间衰减。最小诊断先固定输入序列/更新预算/验证次数/按更新次数的学习率，仅改变目标政策，同时报告真实运行时间。若要单独量化时间衰减影响，还需预先指定调度方式的交叉对照。两种诊断均不能混称等墙钟预算主实验，不能用测试集重新挑学习率。

**若要提出跨PDE建议或最优分配法则：必须有新的外部验证或相应方法证据。** 当前不提出这些主张，因此不为凑广度启动第二PDE。

停止规则：先完成受限贡献定位与可复现稿件。只有出现明确、无法通过解释或既有记录解决且正文必须依赖的问题，才提出固定问题、固定比较、固定预算的一次补充设计；无论胜负均报告。不要加种子直到显著，不选择性延长有利配置。

## 交付

论文新增贡献对照表、成本—训练机会—误差主图、检查点/样本覆盖/时间衰减解释，以及不需要或需要新实验的条件。主图只用最新匹配学习率批次，所有种子和两档学习率保留；其他批次逐行账本保存在endpoint_ledger.csv。保存了更新前main.tex快照。
'''
(R/'review_report.md').write_text(report)
f=P/'literature/positioning.md';s=f.read_text();prefix='本轮更新（2026-10-27）：最新核实状态和贡献边界见 [贡献对照表](contribution_review_20261027.md)。以下保留此前检索记录；其中旧的完成状态由新表和 WORK_STATE 覆盖。\n\n';f.write_text(prefix+s if not s.startswith(prefix) else s)
f=P/'results_report.md';s=f.read_text();prefix='# 文献定位与机制复核已完成（2026-10-27）\n\n无新增训练。100条正式轨迹、280个相关预算点的账本与验证选择复核通过；20个匹配学习率终点中15个选中早期模型，完成更新数不能直接当作选中模型的年龄。时间衰减与标签获取速度耦合，但因果效应未知。完整判断见 [复核报告](mechanism_review_v4/review_report.md)，来源见 [贡献对照表](literature/contribution_review_20261027.md)。Xu全文预算口径仍是文献访问缺口；不据此主张首次联合计时。\n\n---\n\n';f.write_text(prefix+s if not s.startswith(prefix) else s)
f=P/'README.md';s=f.read_text();s='## 最新复核：文献定位与成本机制\n\n[复核报告](mechanism_review_v4/review_report.md) · [贡献对照表](literature/contribution_review_20261027.md)。本轮无新增训练；新增主图与论文解释，保留全部不利结果。当前描述性结论不要求为了显著性补训练；因果机制和跨PDE主张仍需额外证据。\n\n'+s;f.write_text(s)
f=P/'WORK_STATE.json';s=json.loads(f.read_text());s['status']='mechanism_review_v4_updated_pending_compile';s['completed']['mechanism_review_v4']=json.loads((R/'audit.json').read_text());s['next_required']=['Obtain full Xu et al. timing/tuning definitions before exclusionary novelty claims; no such claims currently made','Complete submission positioning and public reproducibility package plus author declarations','No new training planned; controlled experiments only if causal or cross-PDE claims are required'];f.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
print('Manuscript, source comparison, review report and root records updated.')
