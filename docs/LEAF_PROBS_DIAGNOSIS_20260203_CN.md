# 波峰/波谷分布对照实验（Greedy Mask vs Beam/Full）

## 0. 目的与 0131 主线对齐
我们要同时回答两件事（对应 0131 的主线）：
1) **桶回归（bin classification）**：看分桶命中率 `bin_acc_0.2 / bin_acc_0.4` 的表格对比。  
2) **区间预测（interval）**：看 PICP/MPIW + 图形化分布形态（波峰波谷 + n 聚集）。

核心判断：
- greedy 在区间覆盖上结构性偏低；soft mask 只会缩窄分布，不会修复覆盖。  
- beam/full 的多路径推理能显著提升覆盖，逼近 full 的上界。

---

## 1. 实验设置（固定条件）
- 数据集：California Housing 验证集  
- checkpoint：`outputs/california_housing/run_20260202_112549`  
- 温度：`T=1.0`  
- 置信度：`confidence=0.9`  
- greedy 软掩码：`mask_outside ∈ {1.0, 0.5, 0.3, 0.1}`  
- 分布导出样本：`N=200`（用于画图，取同一随机子集）  

**为什么这样设计？**  
1) 固定温度，排除“调温度改变区间宽度”的干扰。  
2) 同一 ckpt + 同一随机子集，保证可视化对比公平。  
3) greedy/beam/full 只改变“推理口径”，更容易归因。  

---

## 2. 数据来源（可追溯）
**指标文件**
- `outputs/california_housing/run_20260202_112549/metrics_val_greedy_T1_C0p9_mask1_20260203_123015.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_greedy_T1_C0p9_mask0p5_20260203_123017.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_greedy_T1_C0p9_mask0p3_20260203_123020.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_greedy_T1_C0p9_mask0p1_20260203_123023.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_prefix_beam_K16_T1_C0p9_Nall_20260203_123038.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_prefix_beam_K32_T1_C0p9_Nall_20260203_123051.json`
- `outputs/california_housing/run_20260202_112549/metrics_val_prefix_full_T1_C0p9_Nall_20260203_123110.json`

**分布文件（用于画图）**
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask1_N200_20260203_123015.npz`
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p5_N200_20260203_123017.npz`
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p3_N200_20260203_123020.npz`
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p1_N200_20260203_123023.npz`
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_20260203_123038_E200.npz`
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K32_T1_C0p9_Nall_20260203_123051_E200.npz`
- `outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_full_T1_C0p9_Nall_20260203_123110_E200.npz`

---

## 3. 桶回归（bin accuracy）结果
**3.1 Greedy + soft mask**
| 口径 | mask_outside | bin_acc_0.2 | bin_acc_0.4 |
| --- | --- | --- | --- |
| greedy | 1.0 (no mask) | 0.1783 | 0.3224 |
| greedy | 0.5 | 0.1865 | 0.3316 |
| greedy | 0.3 | 0.1887 | 0.3406 |
| greedy | 0.1 | 0.1863 | 0.3387 |

**3.2 Beam / Full**
| 口径 | bin_acc_0.2 | bin_acc_0.4 |
| --- | --- | --- |
| prefix_beam_k16 | 0.1814 | 0.3406 |
| prefix_beam_k32 | 0.1829 | 0.3411 |
| prefix_full | 0.1819 | 0.3421 |

**结论（桶回归）**  
- bin_acc 在不同推理口径之间变化不大，说明“单点命中”对推理口径不敏感。  
- 桶回归主线可以继续用表格对比（符合 0131 要求）。

---

## 4. 区间预测（PICP/MPIW）结果
**4.1 Greedy + soft mask**
| 口径 | mask_outside | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- | --- |
| greedy | 1.0 (no mask) | 0.4818 | 0.7296 | 0.3101 | 0.3016 |
| greedy | 0.5 | 0.4814 | 0.7301 | 0.3098 | 0.2968 |
| greedy | 0.3 | 0.4816 | 0.7305 | 0.3077 | 0.2934 |
| greedy | 0.1 | 0.4825 | 0.7314 | 0.2350 | 0.2118 |

**结论**  
- `mask_outside` 越小 → 分布更窄（MPIW 下降），PICP 进一步下降。  
- soft mask 只是在缩窄分布，不能修复 greedy 的结构性低覆盖问题。  

**4.2 Beam / Full**
| 口径 | MAE | RMSE | PICP | MPIW |
| --- | --- | --- | --- | --- |
| prefix_beam_k16 | 0.4506 | 0.6670 | 0.7972 | 1.2196 |
| prefix_beam_k32 | 0.4440 | 0.6502 | 0.8864 | 1.5682 |
| prefix_full | 0.4428 | 0.6465 | 0.8970 | 1.6407 |

**结论**  
- K 增大后 PICP 明显逼近 full，上界在 `~0.90` 左右。  
- 多路径推理能有效缓解 exposure bias。  
- 主线区间推理推荐 **beam（K=16/32）**，full 只作诊断上限。  

补充说明：`width_stratified_PICP`（含 n）记录在上面的 metrics 文件中，可用于按区间宽度进一步分析。  

---

## 4.3 区间宽度分桶命中率（width_stratified_PICP，含 n）
这一部分**对应学长给的格式**：`Width [a, b): value (n=...)`。  
它衡量的是“预测区间宽度”下的覆盖率分层情况，是**区间预测质量**的直接指标。  

数据来源：`outputs/analysis/width_stratified_picp.md`（由 `scripts/summarize_width_stratified_picp.py` 生成）  
注意：该表来自**完整验证集**的 metrics（非 N=200 可视化子集）。  

**口径: greedy_mask1**  
- Width [0.0, 0.4): 0.3130 (n=4054)  
- Width [0.4, 0.8): 0.1486 (n=74)  

**口径: greedy_mask0.5**  
- Width [0.0, 0.4): 0.3128 (n=4054)  
- Width [0.4, 0.8): 0.1486 (n=74)  

**口径: greedy_mask0.3**  
- Width [0.0, 0.4): 0.3092 (n=4088)  
- Width [0.4, 0.8): 0.1500 (n=40)  

**口径: greedy_mask0.1**  
- Width [0.0, 0.4): 0.2350 (n=4128)  

**口径: beam16**  
- Width [0.4, 0.8): 0.8667 (n=30)  
- Width [0.8, 1.2): 0.8221 (n=2743)  
- Width [1.2, 1.6): 0.6656 (n=963)  
- Width [1.6, 2.0): 0.9359 (n=78)  
- Width [2.0, 100.0): 0.9427 (n=314)  

**口径: beam32**  
- Width [0.4, 0.8): 0.8750 (n=8)  
- Width [0.8, 1.2): 0.9455 (n=825)  
- Width [1.2, 1.6): 0.8869 (n=1760)  
- Width [1.6, 2.0): 0.8473 (n=858)  
- Width [2.0, 100.0): 0.8626 (n=677)  

**口径: full**  
- Width [0.4, 0.8): 0.8333 (n=6)  
- Width [0.8, 1.2): 0.9469 (n=810)  
- Width [1.2, 1.6): 0.9003 (n=1665)  
- Width [1.6, 2.0): 0.8515 (n=633)  
- Width [2.0, 100.0): 0.8807 (n=1014)  

**如何理解（与 6.1 的 y_range 表格区别）**  
- 4.3 是**区间预测的质量分层**（按“区间宽度”分桶）。  
- 6.1 是**分布形态与 n 聚集**（按“y 值范围”分桶），用于解释波峰波谷图。  

---

## 5. 外部基线（CatBoost / RealMLP，对齐口径复跑）
> 说明：已按 TabSeq 口径对齐（`v_min/v_max` + `clip_range`），可以直接对比。  
> 数据来源：  
> - CatBoost: `outputs/baselines/baseline_catboost_20260204_110630/metrics_val.json`  
> - RealMLP: `outputs/baselines/baseline_realmlp_20260204_110705/metrics_val.json`  

**模型: CatBoost**  
- 分桶回归命中率 (0.2): 0.1899  
- 分桶回归命中率 (0.4): 0.3530  
- 预测区间命中率: 0.8404  
- 不同宽度区间命中率:  
  - Width [0.0, 0.4): 0.6923 (n=273)  
  - Width [0.4, 0.8): 0.8214 (n=812)  
  - Width [0.8, 1.2): 0.8491 (n=974)  
  - Width [1.2, 1.6): 0.8750 (n=840)  
  - Width [1.6, 2.0): 0.8741 (n=683)  
  - Width [2.0, 100.0): 0.8315 (n=546)  

**模型: RealMLP**  
- 分桶回归命中率 (0.2): 0.2163  
- 分桶回归命中率 (0.4): 0.3917  
- 预测区间命中率: 0.8811  
- 不同宽度区间命中率:  
  - Width [0.0, 0.4): 0.9226 (n=168)  
  - Width [0.4, 0.8): 0.8941 (n=1171)  
  - Width [0.8, 1.2): 0.8904 (n=985)  
  - Width [1.2, 1.6): 0.8555 (n=706)  
  - Width [1.6, 2.0): 0.8762 (n=501)  
  - Width [2.0, 100.0): 0.8626 (n=597)  

**对比说明（与学长旧结果的差异）**  
- 本节结果是**按 TabSeq 口径对齐复跑**得到的（同 v_min/v_max + clip_range）。  
- 如果学长旧结果来自 notebook 或不同口径（范围/切分/clip），数值会出现偏差，不能直接对比。  

**差异的深层原因（为什么“旧 notebook”数值会变）**  
- **数据切分 seed 不同**：旧 notebook 用 `random_state=42`，当前脚本默认 `seed=0`，训练/验证拆分不同会直接影响指标。  
- **bin 范围不同**：旧 notebook 采用固定 `0~5.21`，当前脚本按 TabSeq `v_min/v_max` 构造 bin；范围不同会改变 bin_acc/width_stratified_PICP。  
- **是否 clip 不同**：当前脚本可选 `--clip-range`，旧 notebook 没有 clip；clip 会影响区间宽度与 PICP。  
- **RealMLP 训练超参不同**：旧 notebook 未显式指定 epochs/batch_size（默认 `max_epochs=256`），当前脚本默认 `--epochs 50 --batch-size 256`；训练长度不同会改变性能。  
- **与 TabSeq 训练超参无关**：TabSeq 的 `epoch=2/batch≈512` 是**另一套模型**，不应直接套用到 RealMLP 基线。  

**若要与旧 notebook 更接近的复跑方式**  
- `--seed 42`（对齐切分）  
- `--epochs 256`（对齐 RealMLP 默认训练长度）  
- `--tabseq-config <TabSeq config.json> --clip-range`（对齐 bin 口径）  

---

## 6. 可视化产物（图表清单）
**单样本曲线（看形状）**
- `outputs/analysis/leaf_probs_compare_sample0.png`
- `outputs/analysis/leaf_probs_compare_sample1.png`
- `outputs/analysis/leaf_probs_compare_sample2.png`
- `outputs/analysis/leaf_probs_compare_sample3.png`
- `outputs/analysis/leaf_probs_compare_sample4.png`
说明：每一行对应一个推理口径（greedy + mask、beam、full），红线为 `y_true`。

**聚合版（波峰波谷 + n）**
- `outputs/analysis/leaf_probs_summary_greedy_vs_beam16.png`
- `outputs/analysis/leaf_probs_summary_all.png`（greedy + softmask + beam/full 全对照）

**热力图（基础版 / 诊断用）**
- `outputs/analysis/leaf_probs_overview_beam16.png`
- `outputs/analysis/leaf_probs_overview_greedy_vs_beam16_base.png`

**区间带图（CDF / peak‑merge）**
- `outputs/analysis/interval_bands_cdf.png`
- `outputs/analysis/interval_bands_peakmerge_a0p33.png`

---

## 6.1 聚合图的“n 分布”表格（对应 leaf_probs_summary_all.png）
这些表格与聚合图下半部分一致，**用 y 范围统计 n**，格式参考学长的 width‑stratified 写法。  
说明：  
- `y_true_n` = 真实样本落在该 y 范围的数量（n）。  
- `soft_mass` = 该范围内累计概率质量（期望样本数）。  
- `peak_n` = 峰值 bin 落在该范围的样本数（argmax 统计）。  

数据来源：`outputs/analysis/leaf_probs_summary_ranges.md`（由 `scripts/summarize_leaf_probs_ranges.py` 生成；基于 N=200 可视化子集）  

**口径: greedy_mask1**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.0 | 0 |
| [0.4, 0.8) | 16 | 17.3 | 8 |
| [0.8, 1.2) | 34 | 36.4 | 48 |
| [1.2, 1.6) | 40 | 52.7 | 61 |
| [1.6, 2.0) | 28 | 39.5 | 29 |
| [2.0, 100.0) | 82 | 54.0 | 54 |

**口径: greedy_mask0.5**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.0 | 0 |
| [0.4, 0.8) | 16 | 15.6 | 8 |
| [0.8, 1.2) | 34 | 39.1 | 48 |
| [1.2, 1.6) | 40 | 56.2 | 61 |
| [1.6, 2.0) | 28 | 35.1 | 29 |
| [2.0, 100.0) | 82 | 54.0 | 54 |

**口径: greedy_mask0.3**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.0 | 0 |
| [0.4, 0.8) | 16 | 14.6 | 8 |
| [0.8, 1.2) | 34 | 40.6 | 48 |
| [1.2, 1.6) | 40 | 58.3 | 61 |
| [1.6, 2.0) | 28 | 32.5 | 29 |
| [2.0, 100.0) | 82 | 54.0 | 54 |

**口径: greedy_mask0.1**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.0 | 0 |
| [0.4, 0.8) | 16 | 13.2 | 8 |
| [0.8, 1.2) | 34 | 42.5 | 48 |
| [1.2, 1.6) | 40 | 61.1 | 61 |
| [1.6, 2.0) | 28 | 29.2 | 29 |
| [2.0, 100.0) | 82 | 54.0 | 54 |

**口径: beam16**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.0 | 0 |
| [0.4, 0.8) | 16 | 14.2 | 7 |
| [0.8, 1.2) | 34 | 35.3 | 54 |
| [1.2, 1.6) | 40 | 42.4 | 26 |
| [1.6, 2.0) | 28 | 42.6 | 62 |
| [2.0, 100.0) | 82 | 65.6 | 51 |

**口径: beam32**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.1 | 0 |
| [0.4, 0.8) | 16 | 15.0 | 7 |
| [0.8, 1.2) | 34 | 34.0 | 54 |
| [1.2, 1.6) | 40 | 39.4 | 26 |
| [1.6, 2.0) | 28 | 38.5 | 62 |
| [2.0, 100.0) | 82 | 73.1 | 51 |

**口径: full**  
| y_range | y_true_n | soft_mass (expected) | peak_n |
| --- | --- | --- | --- |
| [0.0, 0.4) | 0 | 0.1 | 0 |
| [0.4, 0.8) | 16 | 15.3 | 9 |
| [0.8, 1.2) | 34 | 34.1 | 48 |
| [1.2, 1.6) | 40 | 39.7 | 54 |
| [1.6, 2.0) | 28 | 37.6 | 36 |
| [2.0, 100.0) | 82 | 73.2 | 53 |

## 7. 图的计算口径（保证可解释）
**7.1 单样本曲线**
- 数据源：`leaf_probs_val_*.npz`（greedy：`scripts/eval.py`，beam/full：`scripts/eval_prefix_search.py`）  
- 作图：`scripts/plot_leaf_probs.py`  
- 含义：某个样本的 `leaf_probs` 形状；红线为 `y_true`。

**7.2 聚合版图（波峰波谷 + n）**
- `mean_probs[j] = mean_i leaf_probs[i, j]`  
- `band[j] = percentile(leaf_probs[:, j], 10~90)`  
- `soft_mass[j] = sum_i leaf_probs[i, j]`（期望样本数）  
- `y_true_hist[j] = count_i(y_true_i 落在 bin j)`

### 7.2.1 聚合版图怎么读（以 `leaf_probs_summary_all.png` 为例）
每个“口径”占两行：**上半行是形状**，**下半行是数量**。  

**上半行（mean shape + band）**  
- 蓝线 = 平均分布形状（整体“波峰波谷”）。  
- 浅蓝带 = 10%~90% 分位，表示样本间波动范围。  
- 关注点：峰的位置、峰的数量、以及右侧尾巴是否被覆盖。  

**下半行（soft_mass + y_true_hist）**  
- 橙线 `soft_mass`：把每个样本在该 bin 的概率累加，得到“期望样本数”。  
- 灰条 `y_true_hist`：真实样本在该 bin 的数量。  
- 关注点：  
  1) 橙线是否覆盖真实分布（灰条），这代表模型是否把概率质量放在真实范围。  
  2) greedy + mask 是否把质量挤在左侧，导致尾部覆盖不足。  

**重要说明（避免误读）**  
- 每一行的 y 轴是独立缩放的，所以不要拿“高度绝对值”横向比较；请重点看**形状与覆盖范围**。  

**7.3 区间带图（仅用于对照）**
- CDF：根据累计概率取 0.05/0.95 分位。  
- peak‑merge：以峰值的 `alpha` 倍阈值合并连续区间。  
- 注意：它们是后处理区间，会把多峰/断层压缩为单一区间，**仅作对照**。  
- 若目标是“看清真实分布 + 统计高概率区域的样本数量”，应直接使用 raw `leaf_probs`。  

---

## 8. 指标解释（口径说明）
- **MAE**：平均绝对误差，越小越好。  
- **RMSE**：均方根误差，越小越好，对大误差更敏感。  
- **PICP**：区间覆盖率，真实值落在预测区间内的比例。  
- **MPIW**：平均区间宽度，越小越好但需结合 PICP 看。  
- **bin_acc_0.2 / bin_acc_0.4**：以桶宽 0.2 / 0.4 计算的分桶命中率（单点命中）。  
- **width_stratified_PICP**：按区间宽度分桶统计的 PICP（含 n）。  
- **oob_rate**：真实值超出预测范围的比例（越小越好）。  

---

## 9. 复现命令（原始执行口径）
**Greedy + mask（导出分布）**
```bash
PYTHONPATH=src python3 scripts/eval.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode greedy --temperature 1.0 --mask-outside 1.0 \
  --export-leaf-probs --export-samples 200

PYTHONPATH=src python3 scripts/eval.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode greedy --temperature 1.0 --mask-outside 0.5 \
  --export-leaf-probs --export-samples 200

PYTHONPATH=src python3 scripts/eval.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode greedy --temperature 1.0 --mask-outside 0.3 \
  --export-leaf-probs --export-samples 200

PYTHONPATH=src python3 scripts/eval.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode greedy --temperature 1.0 --mask-outside 0.1 \
  --export-leaf-probs --export-samples 200
```

**Beam / Full（导出分布）**
```bash
PYTHONPATH=src python3 scripts/eval_prefix_search.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode beam --beam-size 16 --temperature 1.0 \
  --export-leaf-probs --export-samples 200

PYTHONPATH=src python3 scripts/eval_prefix_search.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode beam --beam-size 32 --temperature 1.0 \
  --export-leaf-probs --export-samples 200

PYTHONPATH=src python3 scripts/eval_prefix_search.py \
  --ckpt outputs/california_housing/run_20260202_112549 \
  --mode full --temperature 1.0 \
  --export-leaf-probs --export-samples 200
```

**单样本对比图（同一样本 idx=0）**
```bash
PYTHONPATH=src python3 scripts/plot_leaf_probs.py \
  --inputs \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask1_N200_*.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p5_N200_*.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p3_N200_*.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p1_N200_*.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_*_E200.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K32_T1_C0p9_Nall_*_E200.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_full_T1_C0p9_Nall_*_E200.npz \
  --labels greedy mask0.5 mask0.3 mask0.1 beam16 beam32 full \
  --sample-idx 0 \
  --out outputs/analysis/leaf_probs_compare_sample0.png
```

**聚合版（greedy vs beam16）**
```bash
PYTHONPATH=src python3 scripts/plot_leaf_probs_summary.py \
  --inputs \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask1_N200_*.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_*_E200.npz \
  --labels greedy beam16 \
  --show-y-hist \
  --out outputs/analysis/leaf_probs_summary_greedy_vs_beam16.png
```

**聚合版（全对照：greedy + softmask + beam/full）**
```bash
PYTHONPATH=src python3 scripts/plot_leaf_probs_summary.py \
  --inputs \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask1_N200_20260203_123015.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p5_N200_20260203_123017.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p3_N200_20260203_123020.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_T1_C0p9_mask0p1_N200_20260203_123023.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_20260203_123038_E200.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K32_T1_C0p9_Nall_20260203_123051_E200.npz \
    outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_full_T1_C0p9_Nall_20260203_123110_E200.npz \
  --labels greedy_mask1 greedy_mask0.5 greedy_mask0.3 greedy_mask0.1 beam16 beam32 full \
  --show-y-hist \
  --out outputs/analysis/leaf_probs_summary_all.png
```

> 说明：导出子集使用固定随机种子，因此各文件 sample_idx 对齐；若你改动 seed 或 export_samples，需要重新同步导出。

---

## 10. 扩展（保留但不用于主线决策）
以下内容保留作后续扩展，但不影响当前“桶回归 + 区间预测”主线结论。

**10.1 heatmap 方案（诊断用）**
- `scripts/plot_leaf_probs_overview.py` 可输出热力图 + n 曲线。  
- 适合检查断层/多峰，但可读性差，不作为主线图。  
示例命令：  
```bash
PYTHONPATH=src python3 scripts/plot_leaf_probs_overview.py \
  --inputs outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_*_E200.npz \
  --labels beam16 \
  --sort-by y_true \
  --top-k 3 5 \
  --top-p 0.8 0.9 \
  --peak-merge-alpha 0.33 \
  --show-y-hist \
  --out outputs/analysis/leaf_probs_overview_beam16.png
```

**10.2 top‑k / top‑p / peak‑merge 定义**
- **top‑k**：每个样本取“概率最高的 k 个 bin”，统计这些 bin 的样本数。  
- **top‑p**：每个样本取“累计概率 ≥ p 的最小 bin 集合”，统计这些 bin 的样本数。  
- **peak‑merge**：以峰值的 `alpha` 倍阈值合并连续区间。  

**10.3 额外统计建议（可选）**
- `p_true` 直方图（真值 bin 概率）  
- true_bin 排名分布（rank）  
- top‑k / top‑p 覆盖率统计  

这些统计可用于回答“真值是否落在高概率区域”的更细粒度问题，但目前不是主线。  

**10.4 如果“高概率区定义还没定”**  
- 先画 `soft_mass[j] = sum_i leaf_probs[i, j]` 的曲线（无阈值）。  
- 再用 top‑k / top‑p / peak‑merge 作为候选口径对照。  
