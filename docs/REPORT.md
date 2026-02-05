# 汇报：TabSeq‑Trace 回归工程化进展

## 0. 结论先行

我们已经把 PDF 的主线方法完整工程化：y 被编码成序列，模型按步输出概率，连乘得到叶子分布，再计算 MAE/RMSE + PICP/MPIW。  
基线（CatBoost/RealMLP/MLP/Quantile/FT‑Transformer）已补齐并用统一口径出对比表（`outputs/benchmark.csv`）。  
当前 TabSeq‑DACA+ACM 在 greedy 口径下通过温度校准把 PICP 拉到≈0.90，但区间更宽（MPIW 变大），体现可靠性 vs 区间宽度的权衡。

## 1. 我们到底在做什么

目标：对连续回归 y，给出“点预测 + 区间预测”，并让区间覆盖率接近目标（如 90%）。  
方法：把 y 离散成固定 bins，模型不是直接回归数值，而是预测“序列化的选择过程”，最后还原成分布与区间。

## 1.0 这是什么项目

- 项目名：TabSeq‑Trace 回归原型（把回归问题转成“序列化的二分决策”）。
- 目标交付：可训练、可评估、可复现、可对比的完整流水线（M0→M3）。
- 主模型：TabSeq（表格编码器 + 序列解码器），对齐 `pre/tabseq.pdf` 的 DACA + ACM。
- 对比基线：CatBoost / RealMLP / MLP / Quantile / FT‑Transformer（同口径输出）。

## 1.1 用的数据集与设置
- 数据集：`California Housing`（sklearn 内置），仅数值特征（8 列），暂无类别特征。
- 切分方式：80/20 训练/验证，`random_state=seed`。
- 数值标准化：只用训练集统计做 `StandardScaler`，验证集复用同一 scaler。
- TabSeq 的默认设置：`depth=6`，`n_bins=64`，`confidence=0.90`，`x_cat` 为空。
- 基线额外设置：从训练集中再切出一部分做“校准集”（默认 20%），用于区间校准（残差分位数）。

## 1.2 每一步如何落地

1) 标签编码（把 y 变成序列）  
   - 实现：`TraceLabelEncoder`  
   - 位置：`src/tabseq/labels/trace_encoder.py`

2) 数据集封装（构造 dec_input / y_mht）  
   - 实现：`TabSeqDataset`  
   - 位置：`src/tabseq/data/tabseq_dataset.py`

3) 模型结构（主模型 TabSeq）  
   - 表格编码器：`FeatureTokenizer + TabularEncoder`  
     - 位置：`src/tabseq/models/feature_tokenizer.py`、`src/tabseq/models/tabular_encoder.py`
   - 序列解码器：Transformer Decoder + cross‑attention + causal mask  
     - 位置：`src/tabseq/models/transformer_model.py`
   - DACA gate：对每步 Key 做门控  
     - 位置：`src/tabseq/models/transformer_model.py`

4) 训练（ACM）  
   - ACM 权重：`alpha(x,t)=alpha_depth*alpha_instance`  
   - 位置：`scripts/train.py`

5) 评估（指标统一）  
   - TabSeq 指标：`ExtendedHolographicMetric`  
   - 位置：`src/tabseq/metrics/holographic.py`  
   - 基线指标：`compute_point_interval_metrics`  
   - 位置：`src/tabseq/metrics/regression.py`

## 2. 主线链路

1) 标签编码  
   - 输入：连续 y  
   - 输出：`y_seq`（bit 序列）、`y_mht`（multi‑hot）、`y_raw`

2) 数据集封装  
   - 生成 `dec_input = [SOS] + y_seq[:-1]`（teacher forcing）

3) 模型预测  
   - 输入：`x_num/x_cat` + `dec_input`  
   - 输出：每步 logits，形状 `(B, depth, n_bins)`

4) 分布还原  
   - logits → 概率 → 连乘得到叶子分布

5) 指标计算  
   - 点预测：分布期望  
   - 区间：CDF 分位点 → [L,U]  
   - 指标：MAE / RMSE / PICP / MPIW

## 2.1 指标是怎么计算的

1) 叶子分布  
   - 每一步概率相乘，再归一化：  
     `leaf_probs = prod_t(step_probs[t])`  

2) 点预测  
   - 用桶中心点的期望：  
     `y_hat = sum(leaf_probs * bin_center)`  

3) 区间  
   - 用 CDF 取分位点：  
     `L = q_{alpha/2}`, `U = q_{1-alpha/2}`

4) 指标  
   - `MAE = mean(|y_hat - y|)`  
   - `RMSE = sqrt(mean((y_hat - y)^2))`  
   - `PICP = mean( L <= y <= U )`  
   - `MPIW = mean( U - L )`

含义：  
PICP 是“覆盖率”，目标≈0.90；MPIW 是“区间宽度”，在 PICP 达标前提下越小越好。

## 3. PDF 对齐点

1) DACA（PDF 2.2.2）  
   - 公式：`G_t = sigmoid(MLP(E_pos(t)))`，`K_t = X_ctx ⊙ G_t`  
   - 解释：每一步都有自己的“关注重点”，只强调某些上下文维度（gate 只作用在 Key）。

2) ACM（PDF 2.3.2）  
   - 公式：`alpha(x,t)=alpha_depth(t)*alpha_instance(x)`  
   - 解释：对“非目标分支”的损失减权，让模型不要过度自信，从而保留不确定性。

## 4. 当前工程实现（做了什么、在哪里）

1) 标签/数据  
   - `TraceLabelEncoder` + `TabSeqDataset`

2) 主模型（TabSeq）  
   - 表格编码：`FeatureTokenizer + TabularEncoder`  
   - 解码器：Transformer Decoder（自回归 + cross‑attention + causal mask）  
   - DACA：按步 gate，仅作用在 Key

3) 训练（ACM）  
   - `BCEWithLogitsLoss(reduction="none")`  
   - 负分支乘 `(1 - alpha)`

4) 评估  
   - TabSeq：`ExtendedHolographicMetric`  
   - 基线：`compute_point_interval_metrics`  
   - 主口径：greedy（真实推理）  
   - 温度校准：`sigmoid(logits / T)` 用于把 PICP 拉回目标

## 5. 指标解释（为什么必须这 4 个）

- MAE / RMSE：点预测误差  
- PICP：预测区间覆盖率（目标≈0.90）  
- MPIW：预测区间宽度（在 PICP 达标前提下越小越好）

## 6. M3 基线与对比表

基线覆盖：  
CatBoost / RealMLP / MLP / Quantile / FT‑Transformer  

所有模型输出 `metrics_val.json`，统一汇总到 `outputs/benchmark.csv`。

## 6.1 基线是怎么实现的

- CatBoost：分位数回归，直接输出上下分位点（区间）。  
  - 脚本：`archive/scripts/run_baseline_catboost.py`
- RealMLP：pytabkit 的分位数回归版本。  
  - 脚本：`archive/scripts/run_baseline_realmlp.py`
- MLP：点预测 + 校准集残差分位数构区间。  
  - 脚本：`archive/scripts/train_baseline.py`（`--model mlp`）
- Quantile：pinball loss 训练两个分位点。  
  - 脚本：`archive/scripts/train_baseline.py`（`--model quantile`）
- FT‑Transformer：表格 Transformer 基线 + 校准残差。  
  - 脚本：`archive/scripts/train_baseline.py`（`--model ft_transformer`）

## 7. 当前结果（seed=42，统一口径）

TabSeq‑DACA+ACM（greedy, T=6.5）：  
- MAE 0.4603 / RMSE 0.6339 / PICP 0.8985 / MPIW 2.0418

基线（同口径）：  
- CatBoost：0.3777 / 0.5257 / 0.8503 / 1.2831  
- RealMLP：0.3724 / 0.5254 / 0.8735 / 1.3058  
- MLP：0.4114 / 0.5908 / 0.9060 / 1.8577  
- Quantile：0.4848 / 0.6388 / 0.9038 / 1.8549  
- FT‑Transformer：0.3789 / 0.5515 / 0.8929 / 1.6297

含义：TabSeq 的 PICP 接近 0.90（可靠性达标），但区间更宽（MPIW 较大）。

## 7.1 结果到底说明了什么（更清楚的解释）

- TabSeq 当前“可靠性达标”，但“区间偏宽”；这是符合 PICP/MPIW 权衡的正常现象。  
- 有些基线 MAE/RMSE 更低，但 PICP 更低（区间不够保守）；这说明“点误差小 ≠ 区间可靠”。  
- 因为真实部署只能用 greedy，所以我们以 greedy 作为主口径；teacher forcing 仅用于上限参考。

## 8. Tradeoff

1) 覆盖率 vs 区间宽度  
PICP ↑ 往往伴随 MPIW ↑，更可靠就更宽。  

2) 训练口径 vs 推理口径  
teacher forcing 更乐观；真实部署用 greedy，误差会累积（暴露偏差）。  

3) 校准 vs 结构  
温度校准能快速拉高 PICP，但区间变宽；结构改进更根本但要做消融验证。

## 9. 下一步建议（可选讨论）

1) 减少暴露偏差：scheduled sampling / prefix dropout  
2) DACA 内存优化：gate 融入 key 投影，降低 `(B,T,S,D)` 复制成本  
3) 多数据集复现：验证稳健性  
4) 校准集/共形预测：提升 PICP 的统计保证

---

附：关键产物位置  
- 汇总表：`outputs/benchmark.csv`  
- TabSeq 模型：`src/tabseq/models/transformer_model.py`  
- ACM 实现：`scripts/train.py`  
- 统一指标：`src/tabseq/metrics/holographic.py`  
- 基线入口：`archive/scripts/train_baseline.py` / `archive/scripts/eval_baseline.py`
