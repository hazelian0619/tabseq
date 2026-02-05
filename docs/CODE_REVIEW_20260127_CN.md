# Code Review（2026-02-02）

## 范围
- 代码范围：`src/tabseq/` 与 `scripts/` 主线实现（训练、评估、指标、扫参）
- 目标：找出可能影响结论可信度的逻辑/评估问题

## 发现（按严重度排序）

### 高
1) 区间上下界用“桶中心值”而非边界，导致区间系统性偏窄  
   - 位置：`src/tabseq/metrics/holographic.py:83`  
   - 说明：区间由 CDF 分位点得到桶索引，但最终用 `decode_bin_index`（桶中心）作为 L/U。  
   - 影响：PICP、MPIW 会系统性偏小，和常见“区间边界”定义不一致。

2) 评估时用单一前缀生成的 step_probs 构造全叶子分布，导致非该前缀路径条件概率错配  
   - 位置：`scripts/eval.py:101` + `src/tabseq/metrics/holographic.py:71`  
   - 说明：`step_probs` 是基于单一路径前缀得到的，但 `leaf_probs` 用全叶子连乘。  
   - 影响：非贪心路径的叶子概率不正确，分布过尖/过自信，PICP/MPIW 可能严重偏差。

### 中
3) `width_scan.py` 使用 softmax+argmax 解码，语义与主线 sigmoid 多热监督不一致  
   - 位置：`scripts/width_scan.py:32`  
   - 影响：宽度扫描结果与主线评估不可比，容易得到误导性结论。

4) `inference.py` 的 greedy 评估未对区间应用 mask，评估口径与 `eval.py` 不一致  
   - 位置：`scripts/inference.py:74`  
   - 影响：固定区间命中率与主评估指标无法直接对齐。

5) `inference.py` 的 greedy 使用“单前缀 step_probs 构造全叶分布”，与 `eval.py` 口径不一致  
   - 位置：`scripts/inference.py:54` + `scripts/inference.py:174`  
   - 说明：推理时用单一路径前缀产生 step_probs，但后续对全部叶子做连乘得到 leaf_probs。  
   - 影响：leaf_probs 语义与 `eval.py` 的 greedy（带 mask）不一致，固定区间命中率不可与主指标对齐。

### 低
6) `bin_acc_0.2/0.4` 的分桶边界硬编码为 California Housing 范围  
   - 位置：`src/tabseq/metrics/holographic.py:67`  
   - 影响：迁移到其他数据集时，粗粒度命中率失真。

7) 缺少最小单测  
   - 位置：`tests/` 为空  
   - 影响：评估逻辑变更容易引入回归，难以及时发现。

8) `eval.py` 默认 `random_state=0`，未读取 checkpoint 里的 `seed`  
   - 位置：`scripts/eval.py:137`  
   - 影响：训练/评估划分可能不一致，指标对比有漂移风险。

## 补充更新（2026-02-02 迭代）

### 已修复/对齐
- `eval.py` 读取 checkpoint 的 `seed` 作为默认划分种子（除非显式指定），原条目 8 已对齐。  
- `width_scan.py` 当前已使用 `sigmoid`（非 softmax/argmax），与主线多热监督一致，原条目 3 为过期描述。  
- `inference.py` greedy 已按当前区间应用 mask，口径与 `eval.py` 对齐，原条目 4 已修复。  
- `bin_acc_0.2/0.4` 的分桶边界基于 `encoder.v_min/v_max` 生成，原条目 6 已不适用。  

### 体系性改进（本次新增）
- 评估用 `y_clipped` 与编码范围对齐，并记录 `oob_rate`（超出编码范围比例），避免“裁剪标签但用原值评估”的口径漂移。  
  - 位置：`src/tabseq/data/tabseq_dataset.py:8`、`scripts/eval.py:210`  
- `v_min == v_max` 的退化范围保护，防止除零和 NaN。  
  - 位置：`src/tabseq/labels/trace_encoder.py:5`  
- `TabSeqDataset` 对 `y_mht` 采用“按规模预计算/惰性生成”策略，避免 N×depth×n_bins 内存爆炸。  
  - 位置：`src/tabseq/data/tabseq_dataset.py:16`  

### 仍未解决
- 条目 1（区间边界 vs 桶中心）仍为口径偏差的主要来源。  
- 条目 2（单前缀 step_probs 拼全叶子分布）仍是核心评估偏差，需要 prefix_full/beam 口径对照或改造。  
- 最小单测仍缺失（条目 7）。  

## 待确认点
- 区间用桶中心值是否为“对齐 notebook”而刻意为之？若是，应在文档中明确口径。  
- 单前缀 leaf_probs 是否被视为“可接受近似”？如果是，需要在报告里说明偏差方向。

## 建议（不改主线的最低成本修复顺序）
1) 先把区间上下界从“桶中心”改为“桶边界”，并记录口径变化。  
2) 新增“前缀展开/beam”评估入口作为对照（无需影响训练）。  
3) 统一 `width_scan.py` 与 `eval.py` 的概率语义。  
4) 对齐 `inference.py` 的 greedy 掩码逻辑，避免与 `eval.py` 口径漂移。  
5) `bin_acc_0.2/0.4` 的边界建议按编码器范围生成，避免迁移到新数据集失真。  
6) `eval.py` 默认使用 checkpoint 的 `seed` 作为数据划分种子，除非显式指定。  
7) 增加最小单测（TraceLabelEncoder round-trip / 指标形状检查）。

## 修复记录（2026-02-XX）
- `width_scan.py` 已重写为“按 checkpoint 扫宽度”的入口，强制使用 ckpt 的 depth/n_bins 并走 `ExtendedHolographicMetric` 口径，消除 softmax/argmax 与缺失 loader 的问题。  
  - 位置：`scripts/width_scan.py:1`
- `inference.py` 的 greedy 已补齐子树 mask，并默认使用 checkpoint 的 seed，口径对齐 `eval.py`。  
  - 位置：`scripts/inference.py:74`、`scripts/inference.py:105`
- `eval_prefix_search.py` 默认跟随 checkpoint 的 seed，避免评估划分漂移。  
  - 位置：`scripts/eval_prefix_search.py:187`
- `bin_acc_0.2/0.4` 分桶边界改为基于编码器 `v_min/v_max` 生成，支持跨数据集复用。  
  - 位置：`src/tabseq/metrics/holographic.py:52`

## 测试状态
- 当前未运行任何测试（项目无测试用例）。

## 讨论补充：Exposure Bias 的取舍与主线口径

### 现象与本质
- 训练使用 teacher forcing（真实前缀），推理使用 greedy（模型自生成前缀）。
- 当前缀走错，后续所有概率都建立在“错误条件”上，分布会系统性偏窄。
- 这不是代码 bug，而是“自回归 + 贪心推理”的结构性偏差（exposure bias）。

### 取舍（tradeoff）
- greedy：速度快，路径单一，偏差大，区间往往过窄，PICP 低。
- beam：保留多条前缀，偏差显著降低，但推理成本上升。
- full：前缀枚举，口径最一致，但计算成本最高，仅适合诊断/上限对照。

### 实验证据（2026-02-02）
- teacher_forcing（T=1）：PICP ≈ 0.994（上限口径）
- greedy（T=1）：PICP ≈ 0.310（单前缀偏差明显）
- prefix_full（T=1, 512样本）：PICP ≈ 0.891（接近目标覆盖）
- prefix_beam_k32（T=1, 512样本）：PICP ≈ 0.877（逼近 full，但仍略低）

### 主线建议
- 区间/不确定性任务的主线口径应优先使用 beam（K=16/32），full 作为诊断基准。
- greedy 可作为速度基线或部署口径，但需明确它是“下界口径”。
