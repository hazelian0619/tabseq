# 0131 进度匹配汇报（California Housing）

## 0) 范围与口径
- 数据集：California Housing
- 训练/评估输出：`outputs/california_housing/run_20260202_112549`
- 评估口径：
  - greedy：用于分桶命中率（bin_acc）与诊断
  - prefix_beam（K=16/K=32）：主线区间评估
  - prefix_full：上限参考
- 置信度：0.9，温度：1.0

---

## 1) 对应 0131 第一部分：分桶回归基线（greedy）
**目标**：给出固定桶宽下的命中率基准。

**结果（greedy）**
- MAE: 0.4814
- RMSE: 0.7301
- PICP: 0.3098
- MPIW: 0.2968
- bin_acc_0.2: 0.1865
- bin_acc_0.4: 0.3316

**width_stratified_PICP（greedy）**
- Width [0.0, 0.4): 0.3128 (n=4054)
- Width [0.4, 0.8): 0.1486 (n=74)

**解读**：
- greedy 区间极窄（多数在 0.4 内），覆盖率明显偏低，符合 0131 对“过窄区间→PICP低”的描述。

---

## 2) 对应 0131 第二部分：CDF 基线（prefix_beam / full）
**目标**：用完整分布评估区间质量（PICP/MPIW）。

### 2.1 beam K=16（主线）
- MAE: 0.4506
- RMSE: 0.6670
- PICP: 0.7972
- MPIW: 1.2196

width_stratified_PICP：
- Width [0.4, 0.8): 0.8667 (n=30)
- Width [0.8, 1.2): 0.8221 (n=2743)
- Width [1.2, 1.6): 0.6656 (n=963)
- Width [1.6, 2.0): 0.9359 (n=78)
- Width [2.0, 100.0): 0.9427 (n=314)

### 2.2 beam K=32（对照）
- MAE: 0.4440
- RMSE: 0.6502
- PICP: 0.8864
- MPIW: 1.5682

width_stratified_PICP：
- Width [0.4, 0.8): 0.8750 (n=8)
- Width [0.8, 1.2): 0.9455 (n=825)
- Width [1.2, 1.6): 0.8869 (n=1760)
- Width [1.6, 2.0): 0.8473 (n=858)
- Width [2.0, 100.0): 0.8626 (n=677)

### 2.3 prefix_full（上限参考）
- MAE: 0.4428
- RMSE: 0.6465
- PICP: 0.8970
- MPIW: 1.6407

width_stratified_PICP：
- Width [0.4, 0.8): 0.8333 (n=6)
- Width [0.8, 1.2): 0.9469 (n=810)
- Width [1.2, 1.6): 0.9003 (n=1665)
- Width [1.6, 2.0): 0.8515 (n=633)
- Width [2.0, 100.0): 0.8807 (n=1014)

**解读**：
- K 从 16 → 32 → full，PICP 明显上升，MPIW 也随之增加。
- 目前 full 的 PICP ≈ 0.897，接近目标 0.9，但仍有宽度增长的代价。

---

## 3) 对应 0131 温度参数：现有扫参结果与状态
**现状**：已有温度 sweep 的图表与汇总（主要是 greedy/evalmask 口径）。
- 汇总文件：`outputs/sweeps/temperature/temperature_Tstar_summary_evalmask_20260126.csv`
- 图表：`outputs/sweeps/temperature/*_metrics_20260126.png`

**解读**：
- greedy/evalmask 的 PICP 普遍偏低，说明该口径不适合作为区间主线。
- 主线（beam）尚未做温度扫参，当前仍固定 T=1。

---

## 4) 对应 0131 波峰合并法：状态与下一步
**状态**：
- 设计已完成（单峰合并 + 不跨谷规则），但主线评估尚未接入 peak-merge。
- 已有可用输入：
  - `leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_20260203_123038.npz`
  - `leaf_probs_val_prefix_beam_K32_T1_C0p9_Nall_20260203_123051.npz`
  - `leaf_probs_val_prefix_full_T1_C0p9_Nall_20260203_123110.npz`

**下一步（与 0131 对齐）**：
1) 把 peak-merge 接入主线 eval，输出 PICP/MPIW/width_stratified。
2) 固定 T=1，扫 alpha（0.5/0.33/0.25/0.2）。
3) 以 PICP 达标为硬约束，最小 MPIW 为软优化。
4) 在多峰样本上启用“回退 CDF/不跨谷”规则验证稳定性。

---

## 5) 结论（匹配 0131 目标）
- greedy 仅能作为分桶基线与诊断口径，不适合作区间评估主线。
- beam/full 的 CDF 结果符合“覆盖率↑ ↔ 区间变宽”的规律，具备作为主线基准的意义。
- peak-merge 需要尽快接入主线评估，否则无法验证 0131 的核心假设。
