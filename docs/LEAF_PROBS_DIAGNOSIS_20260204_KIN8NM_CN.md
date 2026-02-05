# kin8nm 波峰/波谷分布对照实验（与 California 对齐）

## 0. 目的与 0131 主线对齐
我们要回答两件事（对应 0131 主线）：
1) **桶回归（bin classification）**：看 `bin_acc_0.2 / bin_acc_0.4` 的表格对比。  
2) **区间预测（interval）**：看 PICP/MPIW + 图形化分布形态（波峰波谷 + n 聚集）。

核心判断：
- greedy 在区间覆盖上结构性偏低；soft mask 只会缩窄分布，不会修复覆盖。  
- beam/full 的多路径推理能显著提升覆盖，逼近 full 的上界。

---

## 1. 实验设置（固定条件）
- 数据集：**kin8nm**（OpenML 44980）验证集
- checkpoint：`outputs/kin8nm/run_20260204_111705`
- 温度：`T=1.0`
- 置信度：`confidence=0.9`
- greedy 软掩码：`mask_outside ∈ {1.0, 0.5, 0.3, 0.1}`
- 分布导出样本：`N=200`（仅用于画图，指标仍在全量验证集上计算）
- 评估命令：`scripts/run_eval_suite.sh`（与 California 配置一致）

**对齐说明**  
本次评估参数与 California 的 `run_20260202_112549` 完全对齐（T=1.0 / confidence=0.9 / seed=0 / mask 扫描 / beam K=16/32 / full）。

---

## 2. 数据来源（可追溯）
**指标文件**
- `outputs/kin8nm/run_20260204_111705/metrics_val_greedy_T1_C0p9_mask1_20260204_121703.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_greedy_T1_C0p9_mask0p5_20260204_121705.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_greedy_T1_C0p9_mask0p3_20260204_121706.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_greedy_T1_C0p9_mask0p1_20260204_121708.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_prefix_beam_K16_T1_C0p9_Nall_20260204_121710.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_prefix_beam_K32_T1_C0p9_Nall_20260204_121716.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_prefix_full_T1_C0p9_Nall_20260204_121725.json`
- `outputs/kin8nm/run_20260204_111705/metrics_val_teacher_forcing_T1_C0p9_20260204_121709.json`

**分布文件（用于画图）**
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_greedy_T1_C0p9_mask1_N200_20260204_121703.npz`
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_greedy_T1_C0p9_mask0p5_N200_20260204_121705.npz`
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_greedy_T1_C0p9_mask0p3_N200_20260204_121706.npz`
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_greedy_T1_C0p9_mask0p1_N200_20260204_121708.npz`
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_20260204_121710_E200.npz`
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_prefix_beam_K32_T1_C0p9_Nall_20260204_121716_E200.npz`
- `outputs/kin8nm/run_20260204_111705/leaf_probs_val_prefix_full_T1_C0p9_Nall_20260204_121725_E200.npz`

**width_stratified_PICP 汇总**
- `outputs/analysis/kin8nm/width_stratified_picp.md`

---

## 3. 桶回归（bin accuracy）结果
**3.1 Greedy + soft mask**
| 口径 | mask_outside | bin_acc_0.2 | bin_acc_0.4 |
| --- | --- | --- | --- |
| greedy | 1.0 | 0.4167 | 0.6266 |
| greedy | 0.5 | 0.4167 | 0.6266 |
| greedy | 0.3 | 0.4167 | 0.6266 |
| greedy | 0.1 | 0.4167 | 0.6266 |

**3.2 Beam / Full**
| 口径 | bin_acc_0.2 | bin_acc_0.4 |
| --- | --- | --- |
| prefix_beam_k16 | 0.4491 | 0.6870 |
| prefix_beam_k32 | 0.4393 | 0.6925 |
| prefix_full | 0.4393 | 0.6894 |

**结论（桶回归）**  
- bin_acc 对推理口径不敏感（greedy/beam/full 差距不大），与 California 结论一致。  
- 桶回归可继续作为“主线表格指标”进行稳定对比。

---

## 4. 区间预测（PICP/MPIW）结果
**4.1 Greedy + soft mask**
| 口径 | mask_outside | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- | --- |
| greedy | 1.0 | 0.1368 | 0.1771 | 0.3862 | 0.1641 |
| greedy | 0.5 | 0.1405 | 0.1808 | 0.2971 | 0.1279 |
| greedy | 0.3 | 0.1428 | 0.1832 | 0.2196 | 0.0970 |
| greedy | 0.1 | 0.1453 | 0.1858 | 0.1489 | 0.0658 |

**结论**  
- `mask_outside` 越小 → 分布更窄（MPIW 下降），PICP 继续下降。  
- soft mask 只会“缩窄分布”，不能修复 greedy 的低覆盖问题。

**4.2 Beam / Full**
| 口径 | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- |
| prefix_beam_k16 | 0.1330 | 0.1700 | 0.7828 | 0.4120 |
| prefix_beam_k32 | 0.1308 | 0.1670 | 0.8755 | 0.5440 |
| prefix_full | 0.1310 | 0.1670 | 0.8816 | 0.5518 |

**结论**  
- K 增大后 PICP 明显逼近 full，上界在 `~0.88` 左右。  
- 多路径推理能有效缓解 exposure bias；主线区间推理推荐 **beam（K=16/32）**。

**4.3 teacher_forcing（上限口径）**
- MAE 0.0380 / RMSE 0.0509 / PICP 0.9549 / MPIW 0.1654  
- 说明模型“理论上能覆盖”，问题主要出在 greedy 推理口径。

补充说明：`width_stratified_PICP` 记录在 `outputs/analysis/kin8nm/width_stratified_picp.md`。  
注意：该诊断项使用固定桶宽（0.4/0.8…），在 kin8nm 小尺度数据上会显得偏粗，仅作趋势参考。

---

## 4.3 区间宽度分桶命中率（width_stratified_PICP，含 n）
**口径: greedy_mask1**
- Width [0.0, 0.4): 0.3862 (n=1639)

**口径: greedy_mask0p5**
- Width [0.0, 0.4): 0.2971 (n=1639)

**口径: greedy_mask0p3**
- Width [0.0, 0.4): 0.2196 (n=1639)

**口径: greedy_mask0p1**
- Width [0.0, 0.4): 0.1489 (n=1639)

**口径: beam16**
- Width [0.0, 0.4): 0.7232 (n=925)
- Width [0.4, 0.8): 0.8599 (n=714)

**口径: beam32**
- Width [0.0, 0.4): 0.8611 (n=108)
- Width [0.4, 0.8): 0.8766 (n=1531)

**口径: full**
- Width [0.0, 0.4): 0.8829 (n=111)
- Width [0.4, 0.8): 0.8815 (n=1528)

---

## 5. 与 California 对标（同配置）
> 注意：MAE/RMSE/MPIW 是目标值尺度相关，不同数据集**不可直接比较数值大小**；
> 这里只对比“趋势形态”（greedy 低覆盖，beam/full 逼近上限）。

| 口径 | California (MAE/RMSE/PICP/MPIW) | kin8nm (MAE/RMSE/PICP/MPIW) |
| --- | --- | --- |
| greedy_mask1 | 0.4818/0.7296/0.3101/0.3016 | 0.1368/0.1771/0.3862/0.1641 |
| beam_k16 | 0.4506/0.6670/0.7972/1.2196 | 0.1330/0.1700/0.7828/0.4120 |
| beam_k32 | 0.4440/0.6502/0.8864/1.5682 | 0.1308/0.1670/0.8755/0.5440 |
| full | 0.4428/0.6465/0.8970/1.6407 | 0.1310/0.1670/0.8816/0.5518 |
| teacher_forcing | 0.0711/0.0888/0.9944/0.3568 | 0.0380/0.0509/0.9549/0.1654 |

**对标结论**  
- **规律一致**：greedy 低覆盖，beam/full 明显改善并接近上限。  
- **kin8nm 更“容易”覆盖**：greedy 的 PICP 高于 California，但仍明显低于 0.90。  
- **beam_k32 已接近 full**：与 California 的结论一致（推荐 K=16/32 作为主线口径）。

---

## 6. 可视化产物（图表清单）
**单样本曲线（形状对照）**
- `outputs/analysis/kin8nm/leaf_probs_compare_sample0.png`
- `outputs/analysis/kin8nm/leaf_probs_compare_sample1.png`
- `outputs/analysis/kin8nm/leaf_probs_compare_sample2.png`
- `outputs/analysis/kin8nm/leaf_probs_compare_sample3.png`
- `outputs/analysis/kin8nm/leaf_probs_compare_sample4.png`
说明：每一行对应一个推理口径（greedy_mask1 / beam16 / full），红线为 `y_true`。

**聚合版（波峰波谷 + n）**
- `outputs/analysis/kin8nm/leaf_probs_summary_greedy_vs_beam16.png`
- `outputs/analysis/kin8nm/leaf_probs_summary_all.png`

**热力图（基础版 / 诊断用）**
- `outputs/analysis/kin8nm/leaf_probs_overview_beam16.png`
- `outputs/analysis/kin8nm/leaf_probs_overview_greedy_vs_beam16_base.png`

**区间带图（CDF / peak‑merge）**
- `outputs/analysis/kin8nm/interval_bands_cdf.png`
- `outputs/analysis/kin8nm/interval_bands_peakmerge_a0p33.png`

---

## 7. 关键结论（面向后续决策）
- kin8nm 上 greedy 仍显著低覆盖（PICP≈0.39），soft mask 只会进一步降低覆盖。  
- beam K=16/32 能把 PICP 拉到 0.78/0.88，full 上限约 0.88，与 California 的趋势一致。  
- 说明“推理口径”仍是主要瓶颈；如果要对标主线，优先固定 beam 作为区间评估口径。
