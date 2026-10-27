"""Finalize evidence-backed decision after all frozen experiments finish."""
from pathlib import Path
import json,statistics,subprocess,sys
r=Path(__file__).resolve().parent
assert json.loads((r/'status.json').read_text())['stage']=='complete'
subprocess.run([sys.executable,str(r/'decision_analysis.py')],check=True,stdout=subprocess.DEVNULL)
d=json.loads((r/'decision_analysis.json').read_text());g=json.loads((r/'results_summary.json').read_text())['groups']
post=json.loads((r/'post_validation.json').read_text())
trials=[json.loads(p.read_text()) for p in (r/'tuning').glob('*/result.json') if not p.parent.name.startswith('._')]
assert len(trials)==48 and all(not t['nonfinite_loss'] for t in trials)
labels={'gradient':'固定率梯度评分','uniform':'固定率均匀购买','field':'固定率场方差评分','data':'原数据自适应版','phys_only':'纯物理校正','hf':'全高精度监督','pretrain':'粗标签预训练后微调','author_mlmc':'作者MLMC实现适配'}
lines=['# Darcy 机制实验最终决策：停止作为论文主线','',
'2026-10-26。48次等额度调参、30条正式20分钟轨迹全部完成，每条保留5分钟预算点，共60个配对预算点；每组3种子。结论是当前梯度评分没有显示可据以继续立项的独立增量。停止为这条路线扩充任务或包装创新。该结论针对当前实现、Darcy任务与实测成本，不是所有梯度采样方法无效的定理。','',
'## 直接隔离评分作用的对照','',
'相同FNO、学习率0.008、平均购买概率0.25、教师模型、损失和时间预算；只改变样本间评分分配。所有评分成本都计入。这里的梯度评分是最后一层梯度误差代理，不是逐样本全网络精确梯度范数。','',
'|方法|5分钟ID误差%|20分钟ID误差%|20分钟OOD误差%|','|---|---:|---:|---:|']
for m in ['gradient','uniform','field']:
 a=g[f'matched_300_{m}'];b=g[f'matched_1200_{m}']
 lines.append(f'|{labels[m]}|{100*a["test_mean"]:.4f}|{100*b["test_mean"]:.4f}|{100*b["ood_mean"]:.4f}|')
lines+=['','下表使用各方法相同调参额度选出的配置，完整种子标准差和所有预算见 results_report.md。','',
'|方法|学习率|5分钟ID误差%|20分钟ID误差%|20分钟OOD误差%|','|---|---:|---:|---:|---:|']
for m in labels:
 a=g[f'selected_300_{m}'];b=g[f'selected_1200_{m}']
 lines.append(f'|{labels[m]}|{b["lr"]}|{100*a["test_mean"]:.4f}|{100*b["test_mean"]:.4f}|{100*b["ood_mean"]:.4f}|')
lines+=['','## 逐种子证据','','正值表示左侧误差更低。两个预算点来自同一轨迹，不能当成独立重复。三种子支持投入决策，不宣称已证明任意小差异具有统计显著性。','','|比较|预算|指标|平均配对相对改善|胜出种子|逐种子改善|','|---|---:|---|---:|---:|---|']
for c in d['comparisons']:
 lines.append(f'|{c["left"]} 对 {c["right"]}|{c["budget"]//60}分钟|{c["metric"]}|{c["paired_relative_improvement"]:.2%}|{c["wins"]}/3|'+', '.join(f'{s}: {v:.2%}' for s,v in c['per_seed'].items())+'|')
lines+=['','## 成本与一致性','','|方法|20分钟高精度标签均数|训练步数|评分秒|高精度求解秒|训练秒|','|---|---:|---:|---:|---:|---:|']
for m,c in d['costs'].items():
 t=c['timings'];lines.append(f'|{labels[m]}|{c["fine_mean"]:.1f}|{c["steps_mean"]:.1f}|{t["scoring"]:.2f}|{t["fine"]:.2f}|{t["training"]:.2f}|')
lines+=['',
'评分开销已经很小，因此继续优化评分实现不足以构成有根据的救援方案。数据版在20分钟中用满2048个高精度标签，固定率各组也已接近用满；纯物理组只使用共同初始化的32个高精度标签。标签生成在当前任务中不是主要时间瓶颈。这是适用范围的解释，不将其当成未经实验验证的因果分解。',
'',
'初始固定状态的旧新评分、采样概率和梯度等价检查在前轮通过；本轮还在训练后模型及独立新输入上检查场方差压缩公式、教师均值、概率均值和精确枚举购买门的全网络梯度期望。三种策略检查均通过。MSE梯度对目标线性是此检验的依据；不据此宣称任意非线性损失的随机目标都无偏。',
'',
f'训练后枚举最大相对梯度误差：{max(x["exact_gate_enumeration_gradient_relative_error"] for x in post["cases"]):.3g}。源文件哈希全部一致，48次调参和30条正式训练无非有限损失，所有预算点检查通过。所有计时训练串行、单CPU线程。作者MLMC采用真实作者代码并适配共同Darcy/FNO协议，不冒充原论文完整复现。',
'',
'## 投入安排','',
'保留求解器、FNO基线、计时与一致性检查作为可复用设施，停止梯度评分主线。第一轮数据版20分钟的优势没有在更强机制对照中变成稳定独立收益，不再以第一轮最好数字作为立项依据。',
'',
'后续优先调查制造缺陷下多稳态结构的分支概率梯度。具体方法、最近全文边界与完整实验比较见 ../../post_mechanism_search/research_direction.md 和 method_specification.md。新方案尚未取得实验优势；通用BVP、条件采样、单分支伴随、SSM降阶和DynIn均是已有方法，只有超过这些强基线的独立能力才进入论文主张。']
(r/'interpretation.md').write_text('\n'.join(lines)+'\n')
s=json.loads((r/'WORK_STATE.json').read_text());s.update(status='complete',completed_tuning=48,completed_formal=30,decision='Stop Darcy gradient-scoring mainline: independent gain not established',results='interpretation.md',checks='audit.json; post_validation.json; decision_analysis.json');(r/'WORK_STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
for fn in ['proposal.md','experiment_protocol.md']:
 p=r.parent/fn;old=p.read_text();mark='<!-- darcy-mechanism-final-20261026 -->'
 if mark not in old:
  block=f'''{mark}
> **2026-10-26 最终更新：停止 Darcy 梯度评分论文主线。** 48次调参、30条正式训练及训练后一致性检查已完成；同学习率、同平均购买概率对照未支持梯度评分的独立增量，原数据版也未超过纯物理校正。以[最终机制结论](darcy_mechanism/interpretation.md)为准。以下为历史方案，不再代表现行立项推荐。后续研究对象见[新方向及强基线要求](../post_mechanism_search/research_direction.md)。

'''
  p.write_text(block+old)
print('Final decision report and project status updated.')
