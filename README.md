# TabSeq

TabSeq 是一个把表格回归改写成“二叉决策轨迹预测”的实验仓库。核心思路是把连续目标 `y` 编码成一条长度为 `depth` 的 0/1 trace，模型逐步预测每一层向左还是向右，最后恢复出叶子分布、点预测和预测区间。

## 方法概览

当前主方法的流程是：

1. 把连续目标 `y` 按固定深度 `depth` 划分成 `2^depth` 个 bin。
2. 用 `TraceLabelEncoder` 把每个样本的目标值编码成一条二叉路径。
3. 模型主干采用 FT-Transformer 风格表格编码器加 Transformer decoder。数值特征被编码成逐列 token，类别特征被编码成 embedding token，再和 `CLS` token 一起送入 FT-Transformer blocks（multi-head self-attention + ReGLU FFN + LayerNorm）；decoder 读取 `[SOS] + prefix bits`，并与表格编码结果做交互，逐步预测整条 trace。
4. 训练时同时优化三个输出头：`bit_head` 预测每层左右分支，`mht_head` 预测每层对所有叶子的 multi-hot 监督，`leaf_head` 直接预测最终叶子分布。总损失是加权和：`L = 1.0 * L_bit + 0.5 * L_mht + 0.25 * L_leaf + 0.0 * L_consistency`，其中 `L_bit` 和 `L_mht` 是 `BCEWithLogits`，`L_leaf` 是 `CrossEntropy`，`L_consistency` 是 bit 概率与 leaf 分布诱导分支质量之间的 `MSE`，当前权重为 0。
5. 推理时使用 beam search 保留 top-k 前缀路径，默认 `beam_size=8`。
6. 将 beam 中完整路径对应的叶子概率合并为 leaf distribution。
7. 基于 leaf distribution 输出：
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

## 安装

推荐使用 Python 3.10。

```bash
conda create -n tabseq python=3.10 -y
conda activate tabseq

python -m pip install -U pip
python -m pip install -e .
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

### 1. 按 OpenML data id 下载 `diamonds`

当前项目统一以 OpenML `42225` 对应的 `diamonds` 作为示例：

```bash
python scripts/download_openml_regression_datasets.py --openml-ids 42225
```

脚本会把数据保存到本地，并使用 OpenML 的真实数据集名字作为目录名：

```text
data/openml_regression/diamonds/
```

下载后数据会放到：

- `data/openml_regression/<dataset_name>/table.csv.gz`
- `data/openml_regression/<dataset_name>/metadata.json`

### 2. 运行 TabSeq 主方法


```bash
python scripts/run_train_test_all_datasets.py --datasets diamonds --temperature-grid 1.2
```

这个脚本会自动：

1. 选择 depth 网格
2. 依次训练不同 temperature
3. 调用 `scripts/test.py` 跑推理
4. 在 `outputs/batch_runs/<timestamp>/` 下保存汇总结果

主结果保存在：

- `summary.csv`
- `summary.json`

## 结果指标

当前主要关注四个指标：

- `bin_acc`: 预测区间对应 bin 是否命中真值 bin
- `tol_bin_acc@1`: 允许相邻 bin 误差的命中率
- `avg_coverage`: 平均覆盖率
- `avg_length`: 平均区间宽度

最重要的是：

- 在 `avg_coverage` 接近目标置信度时，尽量让 `avg_length` 更短

## 基线方法


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

### 默认设备是 `cuda:0`

如果没有 GPU，可以在最终脚本里显式指定，例如：

```bash
python scripts/run_train_test_all_datasets.py --datasets diamonds --temperature-grid 1.2 --device cpu
```
