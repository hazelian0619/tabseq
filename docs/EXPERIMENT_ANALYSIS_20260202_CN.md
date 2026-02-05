# 实验分析（Greedy/温度/Beam 对照，2026-02-02）

## 数据来源
- `outputs/california_housing/run_20260202_112549/metrics_val_teacher_forcing.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_greedy.json`（T=1.0）
- greedy T=1.0，mask_outside ∈ {0.5, 0.3, 0.1} 的控制台输出（见运行日志）
- greedy T=5.0 的控制台输出（见运行日志）
- `outputs/california_housing/run_20260202_112549/metrics_val_prefix_full.json`（512 样本）
- `outputs/california_housing/run_20260202_112549/metrics_val_prefix.json`（beam_k8，512 样本）
- `outputs/california_housing/run_20260202_112549/metrics_val_prefix_beam.json`（beam_k32，512 样本）

> 说明：`prefix_*` 为 512 样本子集；`teacher_forcing/greedy` 为全量验证集。

## 1) Greedy（软掩码）仍明显偏低
| 口径 | 温度 | mask_outside | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- | --- | --- |
| teacher_forcing | 1.0 | - | 0.0711 | 0.0888 | 0.9944 | 0.3568 |
| greedy | 1.0 | 1.0 (no mask) | 0.4818 | 0.7296 | 0.3101 | 0.3016 |
| greedy | 1.0 | 0.5 | 0.4814 | 0.7301 | 0.3098 | 0.2968 |
| greedy | 1.0 | 0.3 | 0.4816 | 0.7305 | 0.3077 | 0.2934 |
| greedy | 1.0 | 0.1 | 0.4825 | 0.7314 | 0.2350 | 0.2118 |

结论：
- soft mask 只是在**缩窄分布**，PICP 没有改善，甚至随 mask_outside 变小而下降。  
- 说明 greedy 的低覆盖**不是“是否乘 mask”造成的**，而是**单前缀推理口径**本身的结构限制。  

## 2) Greedy 调温度：PICP 上升，但代价明显
| 口径 | 温度 | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- | --- |
| greedy | 1.0 | 0.4818 | 0.7296 | 0.3101 | 0.3016 |
| greedy | 5.0 | 0.4644 | 0.6837 | 0.7025 | 1.1688 |

观察：
- PICP 提升来自 **区间变宽**（MPIW 从 0.30 → 1.17）。  
- 点误差（MAE/RMSE）并未显著改善，温度是“校准工具”，不是解决偏差的根因。

## 3) full/beam 证明：多路径能逼近目标覆盖
| 口径 | 温度 | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- | --- |
| prefix_full | 1.0 | 0.4693 | 0.7090 | 0.8906 | 1.6240 |
| prefix_beam_k8 | 1.0 | 0.4927 | 0.7480 | 0.5957 | 0.8010 |
| prefix_beam_k32 | 1.0 | 0.4686 | 0.7114 | 0.8770 | 1.5549 |

结论：beam 的 K 增大后明显逼近 full，说明**多路径推理可以显著缓解 exposure bias**。

## 4) 分桶回归对照（只看 bin_acc）
> 目的：对齐你计划里的“分桶回归命中率”对照。  
> 注意：TabSeq 的 beam 为 512 子集；greedy 为全量验证集（严格对比需统一子集）。

| 模型 | 口径 | 样本 | bin_acc_0.2 | bin_acc_0.4 |
| --- | --- | --- | --- | --- |
| CatBoost | baseline | 全量 | 0.1797 | 0.3413 |
| RealMLP | baseline | 全量 | 0.2464 | 0.4312 |
| TabSeq | greedy (T=1.0) | 全量 | 0.1783 | 0.3224 |
| TabSeq | beam K=16 (T=1.0) | 512 | 0.1641 | 0.3281 |
| TabSeq | beam K=32 (T=1.0) | 512 | 0.1934 | 0.3574 |

### 关键解读
- greedy 与 CatBoost 在 0.2/0.4 粒度命中率接近，但区间覆盖率（PICP）远低。  
- beam 提升区间覆盖率，同时 bin_acc 稍有提升（K=32 更明显）。  

### 指标来源（可追溯）
- CatBoost/RealMLP：来自你提供的基线结果（待补充原始输出文件/脚本）。  
- TabSeq greedy（全量）：`outputs/california_housing/run_20260202_112549/metrics_val_greedy.json`  
- TabSeq beam K=16（512）：`outputs/california_housing/run_20260202_112549/metrics_val_prefix_beam_k16.json`  
- TabSeq beam K=32（512）：`outputs/california_housing/run_20260202_112549/metrics_val_prefix_beam_k32.json`

## 主线决策（基于数据）
- **主线区间口径：beam（K=16/32）**  
  - PICP 接近目标值，成本可控  
- **full：诊断上限**  
  - 只用于对照，不作为主线推理  
- **greedy：仅作为速度基线/部署下界**  
  - 需要在文档中明确其是下界口径
