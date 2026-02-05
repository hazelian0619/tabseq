# 项目架构（数据 → 编码 → 模型 → 评估）

目标：把连续回归 y 转成“二叉决策序列”，主线模型输出每一步对叶子桶的概率，
再合成为 leaf 分布，用 CDF 分位点得到区间并计算 MAE/RMSE/PICP/MPIW。

下面按执行顺序给出总览，便于从目录直接定位关键文件。

## 目录定位（现实可用版）

| 逻辑阶段 | 文件 | 核心职责 | 主要输入 | 主要输出 | 被谁调用 |
| --- | --- | --- | --- | --- | --- |
| 数据 | `src/tabseq/data/datasets.py` | 数据集加载与切分 | seed | train/val split | `scripts/train.py` / `scripts/eval.py` |
| 数据 | `src/tabseq/data/california_housing.py` | California Housing loader | 配置参数 | DataLoader / encoder | `scripts/width_scan.py` |
| 数据 | `src/tabseq/data/tabseq_dataset.py` | 把样本封装成训练 batch | X_num/X_cat/y | `x_num/x_cat/dec_input/y_seq/y_mht/y_raw` | `scripts/train.py` / `scripts/eval.py` |
| 编码 | `src/tabseq/labels/trace_encoder.py` | y→序列/多热标签 | 连续 y | `y_seq` + `y_mht` + 叶子桶索引 | Dataset、指标 |
| 模型 | `src/tabseq/models/feature_tokenizer.py` | 特征 token 化 | `x_num/x_cat` | token embeddings | `ft_encoder.py` |
| 模型 | `src/tabseq/models/ft_encoder.py` | 表格编码器（无 CLS） | token embeddings | ctx tokens | `transformer_model.py` |
| 模型 | `src/tabseq/models/transformer_model.py` | 主模型（DACA + ACM） | `x_num/x_cat/dec_input` | logits `(B, depth, n_bins)` | `scripts/train.py` / `scripts/eval.py` |
| 训练 | `scripts/train.py` | 主线训练入口 | 数据集 + 超参 | checkpoint/config | 手动运行 |
| 评估 | `src/tabseq/metrics/holographic.py` | 主线指标（点/区间） | step_probs + y_true | MAE/RMSE/PICP/MPIW | `scripts/eval.py` |
| 评估 | `scripts/eval.py` | teacher_forcing/greedy 评估 | checkpoint + val | metrics_val*.json | 手动运行 |
| 推理 | `scripts/inference.py` | 推理/生成 | checkpoint + 输入 | 预测结果 | 手动运行 |
| 汇总 | `scripts/benchmark.py` | 汇总各模型指标 | outputs/**/metrics | `benchmark.csv` | 手动运行 |
| 分析 | `scripts/width_scan.py` | 区间宽度扫描 | n_bins 列表 | CSV/图 | 手动运行 |
| 支撑 | `src/tabseq/utils/seed.py` | 固定随机种子 | seed | - | train/eval |
| 记录 | `docs/EXECUTION.md` | 里程碑路线与验收 | - | - | 规划参考 |
| 记录 | `docs/DEVLOG.md` | 过程记录与复盘 | - | - | 汇报审计 |
| 概览 | `README.md` | 快速入口 | - | - | 新同学入口 |

## 核心流程（从 y 到区间）

1) **连续 y → 二叉序列/多热监督**
   - `TraceLabelEncoder` 用训练集 `v_min/v_max` 做等宽分桶，`n_bins = 2^depth`。
   - `encode()` 得到二进制序列 `y_seq`，`encode_multi_hot()` 生成每步的半区间多热标签 `y_mht`。
   - 位置：`src/tabseq/labels/trace_encoder.py`。

2) **数据封装**
   - `TabSeqDataset` 把 `X_num/X_cat/y` 打包成训练 batch：`x_num/x_cat/dec_input/y_seq/y_mht/y_raw`。
   - 位置：`src/tabseq/data/tabseq_dataset.py`。

3) **模型前向**
   - `FeatureTokenizer` + `FTEncoder` 生成 ctx tokens（不含 CLS）。
   - `TransformerTabSeqModel` 用 DACA 解码器输出每步 logits `(B, depth, n_bins)`。
   - 位置：`src/tabseq/models/feature_tokenizer.py`、`src/tabseq/models/ft_encoder.py`、`src/tabseq/models/transformer_model.py`。

4) **step_probs → leaf_probs**
   - 评估时对 logits 做 `sigmoid(logits / T)` 得到每步概率。
   - `leaf_probs = prod(step_probs, dim=1)` 再归一化，得到叶子分布。
   - 位置：`scripts/eval.py`、`src/tabseq/metrics/holographic.py`。

5) **CDF 选区间 + 点预测**
   - 计算 `CDF = cumsum(leaf_probs)`，取等尾分位点得到 `[L, U]`。
   - 当前实现把桶索引映射为**桶中心**，而不是边界。
   - 点预测为 `E[y] = sum(leaf_probs * bin_center)`。
   - 位置：`src/tabseq/metrics/holographic.py`。

6) **指标输出**
   - MAE/RMSE：点预测误差。
   - PICP/MPIW：区间覆盖率与平均宽度。
   - 位置：`src/tabseq/metrics/holographic.py`、`scripts/eval.py`。

## 目录容易让人迷路的原因（简述）

- `docs/ARCHITECTURE.md` 曾引用过已移除或归档的文件，导致“按文档找不到”。
- `archive/` 和 `outputs/` 不是主流程代码，但体积大、文件多，容易干扰定位。
- `scripts/` 下既有训练/评估入口，也有临时分析脚本，需要一个“入口地图”来区分。
