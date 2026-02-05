# 换数据集思路（OpenML CTR23）

## 1) 目的（先说清楚）
我们要把“只用 California Housing”变成“可以换成其他回归数据集”，并且每次评估都有一致的输出文件，方便对比。

## 2) 选数据集的标准（工程上）
- 先选**数值特征为主**的数据集：现有主线最稳。
- 要有**规模梯度**：小/中/大，便于发现性能和训练稳定性问题。
- 要有**噪声差异**：让区间指标能体现“可靠/保守”的变化。

## 3) 首批推荐的 3 个（够用、易跑）
- **kin8nm (OpenML 44980)**：小规模、低维，跑得快，适合排错。
- **naval_propulsion (OpenML 44969)**：中等规模、噪声随工况变化，适合验证区间可靠性。
- **superconductivity (OpenML 44964)**：高维、噪声复杂，检验模型容量。

## 4) 数据落地位置（固定目录）
下载后文件在：
- `data/openml/<dataset>/X_num.npy`
- `data/openml/<dataset>/X_cat.npy`
- `data/openml/<dataset>/y.npy`
- `data/openml/<dataset>/meta.json`

示例：
- `data/openml/kin8nm/`

## 5) 工业化切换流程（统一入口）
目标：所有脚本只需要一个参数 `--dataset`。

**步骤 A：下载**
- 输入：数据集名字（如 `kin8nm`）
- 输出：上面的 `X_num/X_cat/y/meta`
- 例子：`python3 scripts/download_openml.py --datasets kin8nm`

**步骤 B：加载与切分**
- 输入：`X_num/X_cat/y` + `random_state`
- 输出：`X_train/X_val/X_test` 和同样的 `y_*`
- 例子：固定 80/20 切分，保证可复现

**步骤 C：训练**
- 输入：训练集 + 编码器参数（`v_min/v_max/depth`）
- 输出：`outputs/<dataset>/run_*/checkpoint.pt` 与 `outputs/<dataset>/run_*/config.json`

**步骤 D：评估**
- 输入：checkpoint + 验证集
- 输出：`outputs/<dataset>/run_*/metrics_val_*.json`

## 6) “smoke 训练”是啥
这是“最小能跑通”的训练，用来确认管线没断。
- 输入：小批量数据 + 很少 epoch
- 输出：能保存 checkpoint，评估能跑完
- 例子：用 kin8nm，batch=32，epochs=1

## 7) 指标解释（简单版）
每个指标都用“输入/输出/例子”说明：

- **MAE**
  - 输入：预测值 `y_hat` 和真实值 `y_true`
  - 输出：平均绝对误差（越小越好）
  - 例子：真实 10，预测 12，误差是 2

- **RMSE**
  - 输入：`y_hat` 和 `y_true`
  - 输出：均方根误差（更重视大误差）
  - 例子：一个样本误差很大时，RMSE 会更大

- **PICP**
  - 输入：预测区间 `[L, U]` 和 `y_true`
  - 输出：真实值落在区间内的比例
  - 例子：10 个样本里有 9 个被区间覆盖，PICP=0.9

- **MPIW**
  - 输入：预测区间 `[L, U]`
  - 输出：区间平均宽度
  - 例子：区间越宽，MPIW 越大

常见经验：
- 工业界更关注“稳定 + 可复现 + 误差不爆炸”
- 学术界会同时看“覆盖率(PICP) + 宽度(MPIW)”

## 8) 我们要核对的结果（文件名与位置）
训练产物（每次训练都应该有）：
- `outputs/<dataset>/run_*/checkpoint.pt`
- `outputs/<dataset>/run_*/config.json`
- `outputs/<dataset>/run_*/metrics_train.json`

评估产物（每次评估都应该有）：
- `outputs/<dataset>/run_*/metrics_val_teacher_forcing.json`
- `outputs/<dataset>/run_*/metrics_val_greedy.json`

宽度扫描（如果跑）：
- `width_scan_results_*.csv`
- `width_performance_curve_*.png`

## 9) 类别特征接入与硬校验（新增）
为了以后能顺利切换到“带类别特征”的数据集，必须同时满足 2 个条件：

1) 数据切分阶段要提供：
   - `split.X_cat_train` / `split.X_cat_val`（形状: [N, C]）
   - `split.cat_cardinalities`（长度 = C，每列的取值基数）

2) 训练与评估必须一致：
   - 训练时: `X_cat` 列数 = `len(cat_cardinalities)`
   - 评估/推理时: 数据的 `X_cat` 列数必须与 checkpoint 里的 `cat_cardinalities` 一致

硬校验策略（防止“悄悄丢特征”）：
- 有 `X_cat` 但没 `cat_cardinalities` -> 直接报错
- 有 `cat_cardinalities` 但 `X_cat` 为 0 列 -> 直接报错
- 列数不匹配 -> 直接报错

这样做的目的只有一个：保证类别特征不被静默忽略，避免实验结果不可解释。
