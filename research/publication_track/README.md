# Finite construction budgets for multifidelity operator learning

更新：2026-10-27（Asia/Seoul）。用户策略：2025中科院大类先投二区、被拒后考虑三区。

当前稿件：[main.tex](manuscript/paper/main.tex)。投稿准备：[submission_v5](submission_v5/README.md)。本轮未新增训练或测试评估，全部不利结果保留。

## 已完成证据

|批次|正式轨迹|相关预算点|主要用途|
|---|---:|---:|---|
|formal_mg_v1|35|70|强HF求解器、等预算调参、固定HF复用与粗/校正目标|
|architecture_horizon_v2|45|130|第二架构和FNO长预算；独立新数据|
|matched_rate_v3|20|80|两档固定初始学习率×两种目标×五种子，无新增调参|
|原机制研究（单独附录）|30|60|梯度代理及训练主导情形；共享数据，不能与新批次合并|

现有数据机制复核覆盖前三批100条轨迹/280点。完成更新与选中模型的更新数分开；墙钟衰减和标签时间耦合，因果作用未识别。完整解释见[机制复核报告](mechanism_review_v4/review_report.md)。

## 文献与限制

[贡献对照表](literature/contribution_review_20261027.md)及[Xu专项核查](submission_v5/xu_verification.md)。Xu全文、附录和对应代码的详细计时规则尚未取得；不据此声称首次联合计时。本文限单PDE家族、已测FNO/DeepONet设置、CPU和有限时限，不宣称跨PDE、收敛、因果机制或普遍优越。

## 复现与保存

优先使用投稿包中的`tools/reproduce.py`核验保存预测和统计，不运行训练。完整清单、依赖、公共与离线成本、复现层次和模型资产均在submission_v5中说明。原计算源码保持冻结；旧清单保留为各历史版本快照。旧派生报表模块冲突事件见formal_mg_v1/analysis_import_incident.json，不能笼统声称整个旧目录从未变化。
