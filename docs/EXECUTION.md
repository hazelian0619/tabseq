# 工业级执行文档（从 0 到可复现对比实验）

本文件回答三个问题：
1) 这个项目最终交付什么（对导师/提案的“工程化标准”）
2) 我们把 notebook 里的方法拆成哪些模块（逻辑架构）
3) 每一步怎么做（按里程碑推进、每步有可验收产物）

> 约定：本文用“桶/bin/叶子桶”表示对连续 y 的均匀离散化；用“序列/bit 序列”表示叶子桶编号的二进制表示。

---

## 0. 最终交付标准（Definition of Done）

做到下面这些，基本就满足“工业化、可复现、可对比”的提案标准：

- 能用单条命令训练：输入数据集 + 配置，输出 checkpoint、训练日志、评测指标 JSON/CSV
- 能用单条命令评测：给 checkpoint + 数据集，输出固定口径指标（MAE/RMSE、PICP、MPIW、分桶命中率等）
- 能跑基线并对齐同一口径：至少 2~3 个 baseline（如分位数回归/区间估计基线、普通回归 MLP、一个 tabular transformer）
- 可复现：固定随机种子、记录环境/依赖、记录配置与数据切分
- 可追踪：每次实验都有唯一 ID，能从日志定位“哪一次跑出来的结果”
- 可协作：代码结构清晰（模块化、可测试）、有 README、能走 PR/review（即便你是一个人也按这个习惯做）

---

## 1. 逻辑架构（把 notebook 拆成 6 个模块）

从输入到输出是一条流水线：

1) 数据与预处理（X）
   - 输入：原始表格（数值列 + 类别列）
   - 输出：`x_num`（float tensor）、`x_cat`（long tensor）

2) 标签编码（y → 序列/桶）
   - 位置：训练前（把回归标签变成“可监督的序列任务”）
   - 组件：`TraceLabelEncoder`
   - 输出：
     - `y_seq`：长度 `depth` 的 0/1 序列（叶子桶编号的二进制）
     - `y_mht`：形状 `[depth, n_bins]` 的 multi-hot（每一层二分后“仍可能的叶子集合”）
     - `y_raw`：原始连续 y（评估仍要用）

3) 数据集封装（Dataset / DataLoader）
   - 组件：`TabSeqDataset`
   - 输出（每个样本）：
     - `dec_input = [SOS] + y_seq[:-1]`（teacher forcing：把真实前缀喂给模型）
     - `y_mht`（监督目标）
     - `x_num/x_cat`（条件信息）

4) 模型（你们当前缺口）
   - 输入：`x_num/x_cat` + `dec_input`
   - 输出：`model_logits` 或 `model_probs`，形状建议为 `[batch, depth, n_bins]`
   - 模型由两部分组成：
     - Tabular Encoder：把表格 X 变成上下文向量（embedding）
     - Sequence Decoder：按步预测（或一次性输出）每一步对叶子桶的“软约束”

5) 还原与评估（模型输出 → 分布 → 点/区间 → 指标）
   - 位置：训练中/训练后（统一口径出结果）
   - 组件：`ExtendedHolographicMetric` + `decode_*`
   - 输出：MAE、RMSE、PICP（区间覆盖率）、MPIW（区间宽度）、分桶命中率等

6) 基线与对比（Benchmark）
   - 位置：最终论文/提案要的对比表
   - 要求：同数据切分、同评估口径、同日志记录方式

---

## 2. 工程化目录结构（推荐）

把 notebook 里的“概念验证”收敛成脚本后，建议落成如下结构：

```
.
├── README.md
├── docs/
│   └── EXECUTION.md
├── src/
│   ├── data/
│   │   ├── datasets.py          # 数据集加载/切分/预处理
│   │   └── tabseq_dataset.py    # TabSeqDataset（含dec_input/y_mht）
│   ├── labels/
│   │   └── trace_encoder.py     # TraceLabelEncoder（encode / multi-hot / decode）
│   ├── models/
│   │   ├── tabular_encoder.py   # 数值/类别embedding与融合
│   │   └── seq_decoder.py       # 序列预测头（最简版到Transformer版）
│   ├── metrics/
│   │   └── holographic.py       # ExtendedHolographicMetric
│   └── utils/
│       ├── seed.py              # 固定随机种子
│       └── logging.py           # 统一日志/保存config
├── scripts/
│   ├── train.py                 # 训练入口：配置 -> 训练 -> 保存
│   └── eval.py                  # 评测入口：checkpoint -> 指标
├── configs/
│   ├── default.yaml             # 默认超参（depth、lr、batch等）
│   └── datasets/
│       └── california_housing.yaml
├── outputs/                     # 每次实验自动创建子目录（不进git）
└── tests/
    ├── test_trace_encoder.py    # 编码/解码一致性
    └── test_smoke_train.py      # 小数据跑1-2步不报错
```

> 目前仓库还没有 `src/` 等目录；先按里程碑逐步补齐即可。

---

## 3. 里程碑计划（每一步都能验收）

### M0：理解与验收（不改模型）
目标：把“标签编码/评估口径”完全理解且可复现。

步骤：
- 跑通 `tabseq_trace_design.ipynb` 的验证单元（能输出指标）
- 解释清楚三件事（写到 README 或汇报里）：
  - y 如何变成 `leaf_idx`、如何变成 `y_seq`
  - multi-hot 每一行代表什么
  - 如何从叶子分布得到点预测与区间

验收产物：
- 一份简短说明（可直接发导师/学长）
- notebook 输出的指标截图或保存的结果表

### M1：最小闭环（端到端可训练）
目标：不追最强效果，先把流水线 2→3→4→5 串起来。

步骤：
1) 把 `TraceLabelEncoder` 和 `TabSeqDataset` 从 notebook 抽到 `src/`
2) 实现一个“最简模型”：
   - Tabular Encoder：`x_num/x_cat` → 单个向量（MLP + embedding）
   - Sequence Head：输出 `[depth, n_bins]` 的 logits（先不做复杂Transformer也可以）
3) 写训练脚本 `scripts/train.py`：
   - 固定 seed、记录 config
   - 保存 checkpoint（最优验证集 MAE 或 RMSE）
4) 写评测脚本 `scripts/eval.py`：
   - 载入 checkpoint
   - 用 `ExtendedHolographicMetric` 输出指标 JSON/CSV

验收产物：
- `python scripts/train.py ...` 一条命令能跑完并生成 `outputs/exp_xxx/`
- `python scripts/eval.py ...` 能复现同一口径指标

### M2：对齐论文方法（Transformer + 条件生成）
目标：实现“表格条件 + 序列预测”的更标准版本，对齐 Ord2Seq 思路。

步骤：
- 把 Sequence Head 升级为 decoder（可先用 PyTorch TransformerDecoder）
- 让 decoder 以 `dec_input` 自回归，条件信息来自 tabular embedding（cross-attention 或 prefix）
- 训练策略：teacher forcing；推理阶段可 greedy/beam（先 greedy）

验收产物：
- 训练曲线稳定
- 推理可输出点预测 + 区间（评估脚本不改即可）

### M3：基线、表格、实验管理
目标：出对比表，达到“提案/论文可呈现”的工程状态。

步骤：
- 实现/接入 baseline：
  - 简单回归（MLP）
  - 分位数回归（pinball loss / quantile regression）
  - 一个 tabular transformer（FT-Transformer/TabTransformer/SAINT 任选一个先做）
- 统一评估口径（同一份 metric 代码）
- 记录每个实验：
  - 数据集名、depth、seed、模型名、参数量、训练时长
  - 指标表（CSV）+ 关键图（可选）

验收产物：
- `results/benchmark.csv`（或 `outputs/**/metrics.json` 汇总）
- README 写明如何复现表格

---

## 4. 关键实现细则（避免走弯路）

### 4.1 标签范围 v_min/v_max 怎么设
- 训练/评估必须一致：同一数据集同一套 `v_min/v_max/depth`
- 推荐：在训练集统计后固定（例如分位数裁剪），并把它写进 config/日志

### 4.2 depth 怎么选
- depth 越大：桶越细，点预测上限更好，但输出维度 `n_bins=2^depth` 会指数爆炸
- 建议起步：`depth=6`（64桶）或 `depth=8`（256桶），先跑通再调

### 4.3 multi-hot 监督的理解（训练时）
- 第 t 行不是“唯一正确 leaf”，而是“包含 leaf 的那半棵子树的所有叶子”
- 训练目标是学会逐步收缩候选集合（前粗后细）

### 4.4 记录与可复现（强制要求）
每次训练输出目录里至少要有：
- `config.yaml`（完整配置）
- `metrics_val.json`（验证集指标）
- `checkpoint.pt`
- `git.txt`（commit hash）

---

## 5. 推荐工作方式（给新手的最小操作手册）

你每天只做三件事：
1) 选一个里程碑任务（例如：抽出 TraceLabelEncoder 到 src）
2) 做完就能跑一个“最小验证”（例如：单元测试或一次 train/eval）
3) 提交 git：让每一步都可回滚、可解释

提交信息建议：
- `feat(labels): extract TraceLabelEncoder`
- `feat(train): add minimal training loop`
- `fix(metrics): align PICP computation with benchmark`

