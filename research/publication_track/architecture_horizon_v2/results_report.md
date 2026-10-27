# 第二架构与长预算复核

27次主调参、45条正式轨迹、每方法5个独立训练池/优化种子；共130个相关预算点。DeepONet五种方法180秒；FNO四种方法600秒，记录120/180/360/600秒。新参考集与新训练池，不能与v1混为重复。

|架构|方法|预算秒|ID误差%（均值±SD）|OOD误差%|HF标签|更新次数|
|---|---|---:|---:|---:|---:|---:|
|deeponet|Coarse 32|180|31.268 ± 2.809|35.915|32.0|14352.8|
|deeponet|Corrected 32|180|40.470 ± 1.039|47.153|32.0|384.0|
|deeponet|HF online|180|49.921 ± 0.519|56.504|1064.8|138.2|
|deeponet|HF reuse 32|180|37.757 ± 1.795|42.850|32.0|22850.4|
|deeponet|HF reuse 128|180|31.105 ± 0.861|37.130|128.0|20176.0|
|fno|Coarse 32|180|18.224 ± 2.157|20.153|32.0|379.8|
|fno|Corrected 32|180|19.195 ± 1.562|21.374|32.0|191.2|
|fno|HF online|180|21.700 ± 2.432|24.802|805.2|101.6|
|fno|HF reuse 128|180|15.741 ± 0.884|18.052|128.0|401.0|
|fno|Coarse 32|600|15.805 ± 0.579|16.159|32.0|1422.0|
|fno|Corrected 32|600|14.222 ± 1.779|15.990|32.0|791.2|
|fno|HF online|600|15.884 ± 1.166|17.918|2699.6|403.6|
|fno|HF reuse 128|600|13.453 ± 0.647|15.618|128.0|1532.0|

## 审计与限制
{
  "formal_runs": 45,
  "tuning_runs": 27,
  "budget_points": 130,
  "all_sources_match": true,
  "all_caps_pass": true,
  "nonfinite_formal": 0,
  "nonfinite_tuning": 0,
  "all_initial_complete": true,
  "max_hf_residual": 1.000030221470189e-08,
  "tuning_allocated_per_arm": {
    "deeponet_coarse32": 180,
    "deeponet_phys32": 180,
    "deeponet_hf_online": 180,
    "deeponet_hf32": 180,
    "deeponet_hf128": 180,
    "fno_coarse32": 180,
    "fno_phys32": 180,
    "fno_hf_online": 180,
    "fno_hf128": 180
  }
}

长预算检查点不增加独立重复数；新旧评估集不同，不能直接归因跨轮绝对误差差值。两架构容量不同；学习率搜索区间不同但分配时间相同；仅一个调参训练池。所有不利种子和非有限损失标记均保留。
