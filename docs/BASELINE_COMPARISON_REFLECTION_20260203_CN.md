# Baseline 对照实验反思（CatBoost / RealMLP vs TabSeq）

## 1. 目标与主线对齐
本反思聚焦一个问题：**基线结果是否可与 TabSeq 直接对比**。  
主线仍是两件事：  
1) 桶回归（bin_acc_0.2 / 0.4）  
2) 区间预测（PICP/MPIW + width_stratified_PICP）

## 2. 我们遇到的口径风险
### 2.1 y 范围与分桶边界不一致
- TabSeq 的 bin_acc 与 width_stratified_PICP 依赖 **v_min/v_max**。  
- 基线若使用默认范围（0~5.2），会与 TabSeq 的实际范围不一致。  
影响：bin_acc_0.2/0.4 与 width_stratified_PICP **不可直接对比**。

### 2.2 点预测口径不一致
- 基线脚本用 **区间中点** 当点预测（y_pred = (L+U)/2）。  
- TabSeq 的点预测来自 **leaf_probs 期望**。  
影响：bin_acc 只能“近似对比”，不是严格同一口径。

**为什么会这样（深层逻辑）**  
- 这是**输出类型决定的口径差异**，不是“模型优劣”。  
- CatBoost/RealMLP 在基线里只输出两个分位点 `[L, U]`，没有完整分布，因此**唯一自然的点预测**是取中点。  
- TabSeq 输出的是完整离散分布 `leaf_probs`，因此**点预测自然用期望值**（概率加权的均值）。  

**脚本点位（可追溯）**  
- Baseline 的中点逻辑在 `archive/src/tabseq/metrics/regression.py:31`：未提供 `y_pred` 时自动使用 `(y_lower + y_upper) / 2`。  
- Baseline 调用该逻辑的位置：  
  - `archive/scripts/run_baseline_catboost.py:88`  
  - `archive/scripts/run_baseline_realmlp.py:91`  
- TabSeq 的期望点预测在 `src/tabseq/metrics/holographic.py:140`：`y_pred_point = sum(leaf_probs * bin_values)`。

### 2.3 数据切分依赖 seed
- baseline 的 split 依赖 `--seed`，TabSeq 默认 seed=0。  
影响：seed 不一致会导致训练/验证分布不同。

## 3. 我们做了哪些对齐修复
- 给 baseline 脚本增加 `--tabseq-config`，读取 TabSeq 的 `v_min/v_max`。  
- 增加 `--clip-range`，把 y_true/y_lower/y_upper 裁剪到 `[v_min, v_max]`。  
- 用同样的 `bin_edges_02/04` 计算 bin_acc 与 width_stratified_PICP。  
- 记录 `oob_rate`，方便判断范围偏差对覆盖率的影响。  

修改位置：  
- `archive/scripts/run_baseline_catboost.py`  
- `archive/scripts/run_baseline_realmlp.py`

## 4. 当前对比结论（如何使用这些结果）
### 4.1 区间预测指标（主对比）
- **PICP/MPIW/width_stratified_PICP** 可视为严格对比指标。  
前提：必须带 `--tabseq-config` 与 `--clip-range`，并保持 seed 一致。

### 4.2 桶回归指标（近似对比）
- bin_acc 的点预测口径不同，因此只能用于“趋势对比”。  
如需严格对比，需为 baseline 增加更接近 TabSeq 的点预测定义。

## 5. 标准跑法（保证口径一致）
```bash
PYTHONPATH=src python3 archive/scripts/run_baseline_catboost.py \
  --seed 0 --confidence 0.9 \
  --tabseq-config outputs/california_housing/run_20260202_112549/config.json \
  --clip-range

PYTHONPATH=src python3 archive/scripts/run_baseline_realmlp.py \
  --seed 0 --confidence 0.9 \
  --tabseq-config outputs/california_housing/run_20260202_112549/config.json \
  --clip-range
```

已复跑输出（口径对齐）：  
- CatBoost: `outputs/baselines/baseline_catboost_20260204_110630/metrics_val.json`  
- RealMLP: `outputs/baselines/baseline_realmlp_20260204_110705/metrics_val.json`  

## 6. 仍需明确的假设
- baseline 与 TabSeq 使用的训练/验证划分一致（seed 一致）。  
- 基线模型输出的区间与 TabSeq 置信度一致（confidence=0.9）。  
- 若上述任一不一致，应在报告中注明“仅作参考，不可直接对比”。  
