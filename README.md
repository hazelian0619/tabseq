# TabSeq

TabSeq 是一个把表格回归改写成“二叉决策轨迹预测”的实验仓库。核心思路是把连续目标 `y` 编码成一条长度为 `depth` 的 0/1 trace，模型逐步预测每一层向左还是向右，最后恢复出叶子分布、点预测和预测区间。

当前代码已经具备可复现的训练、测试、批量实验和基线对比能力，适合直接跑实验，也适合继续扩展到新数据集。

## 方法概览

当前主方法的流程是：

1. 把连续目标 `y` 按固定深度 `depth` 划分成 `2^depth` 个 bin。
2. 用 `TraceLabelEncoder` 把每个样本的目标值编码成一条二叉路径。
3. 用表格编码器读取数值特征和类别特征，逐步预测每一层 bit。
4. 推理时使用 beam search 保留 top-k 前缀路径，默认 `beam_size=8`。
5. 将 beam 中完整路径对应的叶子概率合并为 leaf distribution。
6. 基于 leaf distribution 输出：
   - 点预测
   - 区间预测
   - 四个核心指标：`bin_acc`、`tol_bin_acc@1`、`avg_coverage`、`avg_length`

当前默认主配置见 [configs/default.yaml](/mnt/lustre/liuzhiwei/cpx/project/tabseq/configs/default.yaml)：

- `binning_strategy: uniform`
- `encoder_type: ft_transformer`
- `depth: 6`
- `interval_method: relaxed_mass`
- `mode: beam`
- `beam_size: 8`
- `leaf_prior_weight: 0.0`
- `confidence_grid: [0.86, ..., 0.98]`

## 仓库结构

- [src/tabseq](/mnt/lustre/liuzhiwei/cpx/project/tabseq/src/tabseq): 核心代码
- [scripts/train.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/scripts/train.py): 训练入口
- [scripts/test.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/scripts/test.py): 测试/推理入口
- [scripts/run_train_test_all_datasets.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/scripts/run_train_test_all_datasets.py): 批量跑主方法
- [scripts/download_openml_regression_datasets.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/scripts/download_openml_regression_datasets.py): 下载并保存标准回归数据集
- [scripts/collect_all_methods_summary.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/scripts/collect_all_methods_summary.py): 汇总 TabSeq 和各基线结果
- [docs/EXECUTION.md](/mnt/lustre/liuzhiwei/cpx/project/tabseq/docs/EXECUTION.md): 执行计划与阶段目标

## 安装

推荐使用 Python 3.10。

```bash
conda create -n tabseq python=3.10 -y
conda activate tabseq

python -m pip install -U pip
python -m pip install -e .
```

如果你还要跑树模型基线，再安装可选依赖：

```bash
python -m pip install catboost lightgbm xgboost pytest
```

`pyproject.toml` 中的最小依赖只有：

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `PyYAML`

因此基线相关包需要额外安装。

## 快速开始

### 1. 下载一个已有数据集

例如下载仓库内置的一组回归数据集：

```bash
python scripts/download_openml_regression_datasets.py --preset core
```

下载后数据会放到：

- `data/openml_regression/<dataset_name>/table.csv.gz`
- `data/openml_regression/<dataset_name>/metadata.json`

### 2. 训练单个数据集

以 `diamonds` 为例：

```bash
python scripts/train.py --config configs/default.yaml --dataset diamonds
```

输出目录默认在：

```text
outputs/diamonds/run_<timestamp>/
```

其中通常会包含：

- `checkpoint.pt`
- 训练阶段保存的配置和指标

### 3. 测试一个训练好的 checkpoint

```bash
python scripts/test.py --ckpt outputs/diamonds/run_<timestamp>/ --dataset diamonds
```

测试结果会写回对应的 run 目录，例如：

- `metrics_test_beam_*.json`

当前 `test.py` 会按 checkpoint 中保存的 `seed` 和 `val_size` 重新构造 held-out split 并评估；文件名叫 `metrics_test_*.json`，但工程内部很多地方仍把这个 split 记作 `val`，这是当前实现习惯。

### 4. 批量跑主方法

如果你要对一个或多个数据集做 sweep：

```bash
python scripts/run_train_test_all_datasets.py --datasets diamonds
```

它会自动：

1. 选择 depth 网格
2. 依次训练不同 temperature
3. 调用 `scripts/test.py` 跑推理
4. 在 `outputs/batch_runs/<timestamp>/` 下保存汇总结果

最终会得到：

- `summary.csv`
- `summary.json`

## 常用命令

### 训练

```bash
python scripts/train.py --config configs/default.yaml --dataset diamonds
python scripts/train.py --config configs/default.yaml --dataset diamonds --depth 6 --epochs 60 --temperature 1.2
```

### 测试

```bash
python scripts/test.py --ckpt outputs/diamonds/run_<timestamp>/ --dataset diamonds
python scripts/test.py --ckpt outputs/diamonds/run_<timestamp>/ --dataset diamonds --confidence 0.95 --beam-size 8
```

### 批量实验

```bash
python scripts/run_train_test_all_datasets.py --datasets diamonds
python scripts/run_train_test_all_datasets.py --datasets diamonds --temperature-grid 1.2
python scripts/run_train_test_all_datasets.py --datasets diamonds --epochs-grid 60 --beam-size 8
```

## 结果指标

当前主要关注四个指标：

- `bin_acc`: 预测区间对应 bin 是否命中真值 bin
- `tol_bin_acc@1`: 允许相邻 bin 误差的命中率
- `avg_coverage`: 平均覆盖率
- `avg_length`: 平均区间宽度

论文里如果只看区间质量，通常最重要的是：

- 在 `avg_coverage` 接近目标置信度时，尽量让 `avg_length` 更短

## 如何添加一个新的数据集

推荐方式是把新数据集整理成仓库支持的本地格式。数据加载逻辑在 [src/tabseq/data/datasets.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/src/tabseq/data/datasets.py)。

### 目录格式

新数据集推荐放在以下任一位置：

- `data/<dataset_name>/`
- `data/openml_regression/<dataset_name>/`

该目录下至少需要两个文件：

```text
data/<dataset_name>/
├── table.csv.gz
└── metadata.json
```

### `table.csv.gz` 格式

- 必须是一张完整表
- 特征列和目标列都放在同一个 CSV 里
- 目标列名稍后在 `metadata.json` 里声明

例如：

```text
feature_1,feature_2,feature_3,target
0.1,A,12.3,105.6
0.4,B,10.1,97.2
...
```

### `metadata.json` 格式

最少只需要一个字段：

```json
{
  "target_name": "target"
}
```

也就是说，只要告诉加载器哪一列是回归目标即可。

### 加载器会自动做什么

当前加载器会自动完成这些步骤：

- 数值列和类别列自动分开
- 数值缺失值用训练集列中位数填充
- 数值特征用训练集统计量做标准化
- 类别特征用 `OrdinalEncoder` 编码
- 未见类别和缺失类别都会安全处理
- 按 `val_size` 随机划分 train/val split

因此，新数据集不需要你手动写额外预处理代码，只要目录格式正确即可。

## 如何训练和测试一个新的数据集

假设你新增的数据集名叫 `my_regression`，目录如下：

```text
data/my_regression/
├── table.csv.gz
└── metadata.json
```

那么训练命令就是：

```bash
python scripts/train.py --config configs/default.yaml --dataset my_regression
```

测试命令就是：

```bash
python scripts/test.py --ckpt outputs/my_regression/run_<timestamp>/ --dataset my_regression
```

如果要直接批量跑：

```bash
python scripts/run_train_test_all_datasets.py --datasets my_regression
```

如果你想同时跑多个数据集：

```bash
python scripts/run_train_test_all_datasets.py --datasets diamonds,my_regression
```

## 另一种接入方式：直接使用 OpenML 名称

如果数据集本身在 OpenML 上，且名称可被 `fetch_openml()` 正确识别，也可以直接写：

```bash
python scripts/train.py --config configs/default.yaml --dataset abalone
```

不过从可复现性和论文实验管理角度，更推荐先把数据固化为本地格式。

如果你希望这个新数据集也出现在预设下载脚本里，还可以修改 [scripts/download_openml_regression_datasets.py](/mnt/lustre/liuzhiwei/cpx/project/tabseq/scripts/download_openml_regression_datasets.py) 中的 `CURATED_REGRESSION_DATASETS`，把它加入 `core` 或 `extended` 预设。不过这一步不是训练所必需的，只是为了统一数据管理。

## 基线方法

仓库里已经包含多种基线脚本。

### 残差 conformal 基线

```bash
bash scripts/run_lightgbm_4metrics_all.sh --datasets diamonds
bash scripts/run_xgboost_4metrics_all.sh --datasets diamonds
bash scripts/run_catboost_4metrics_all.sh --datasets diamonds
```

### 分位数/CQR 基线

```bash
bash scripts/run_lightgbm_quantile_4metrics_all.sh --datasets diamonds
bash scripts/run_xgboost_quantile_4metrics_all.sh --datasets diamonds
bash scripts/run_catboost_quantile_4metrics_all.sh --datasets diamonds
```

这些基线结果通常保存在：

```text
outputs/baselines_four_metrics/<dataset>/<method>/run_<timestamp>/
```

## 汇总所有方法结果

如果你已经跑完 TabSeq 和各基线，可以用：

```bash
python scripts/collect_all_methods_summary.py --datasets diamonds
```

它会自动读取：

- 最新的 TabSeq `outputs/batch_runs/*/summary.csv`
- 各基线最新的 `metrics_val.json`

并输出统一汇总表。

## 常见注意事项

### 1. 新数据集必须包含目标列

`table.csv.gz` 里必须包含 `metadata.json` 指定的目标列，否则会报错。

### 2. 目标值必须是数值

加载器会把目标强制转成 `float32`。如果目标列里有非数值或 NaN，会直接报错。

### 3. 本地数据集会优先于 OpenML

同名数据集如果同时存在于：

- `data/openml_regression/<name>`
- `data/<name>`

那么加载器会优先使用本地目录，而不是在线下载。

### 4. 默认设备是 `cuda:0`

如果没有 GPU，训练和测试时可以显式指定：

```bash
python scripts/train.py --config configs/default.yaml --dataset diamonds --device cpu
python scripts/test.py --ckpt outputs/diamonds/run_<timestamp>/ --dataset diamonds --device cpu
```
