# 开发记录（DEVLOG）

目的：用“人能读懂的一句话”记录每次从 notebook 原型走向工程实现时的关键改动，方便向导师/学长汇报；更细的改动细节以 Git commit 为准。

## 记录规则（建议）

- 每完成一个小步：先能跑通最小验证 → 再 `git commit` → 再在这里补 1~3 行记录。
- 记录四要素：做了什么 / 为什么 / 影响是什么 / 对应 commit（可选）。

## 2026-01-20

- 抽取模块：将 `TraceLabelEncoder` 从 `pre/tabseq_trace_design.ipynb` 抽到 `src/tabseq/labels/trace_encoder.py`，用于工程化复用与脚本化训练（notebook 未修改）。
- 抽取模块：将 `TabSeqDataset` 抽到 `src/tabseq/data/tabseq_dataset.py`，定义训练样本字段：`x_num/x_cat/dec_input/y_seq/y_mht/y_raw`（notebook 未修改）。
- 工程化：新增 `pyproject.toml`，支持 `pip install -e .`，使 `import tabseq...` 在任意工作目录可用。
- 优化（语义不变）：将 `src/tabseq/data/tabseq_dataset.py` 中 list→tensor 的构造改为 `np.asarray(...)` + `torch.from_numpy(...)`，消除 “Creating a tensor from a list of numpy.ndarrays is extremely slow” 警告并加速数据准备（notebook 未修改；用 `python scripts/smoke_model.py` 复核通过）。
- M1 闭环：新增最简模型 `src/tabseq/models/minimal_model.py`，并实现 `scripts/train.py` / `scripts/eval.py`，在 California Housing 上完成 “train -> 保存 checkpoint -> eval 输出 MAE/RMSE 并写入 metrics_val.json” 的最小可复现流程。
- 评估口径扩展：`scripts/eval.py` 增加区间指标输出 `PICP/MPIW`（基于叶子分布 CDF 提取 `[L,U]` 再统计覆盖率与宽度），为后续模型优化提供明确目标。
- M2 准备：新增 `src/tabseq/models/transformer_model.py`（`TransformerTabSeqModel`），接口与 minimal 保持一致（输入 batch，输出 `(B, depth, n_bins)` logits），便于直接替换进入训练/评估。
- M2 对齐推理口径：`TransformerTabSeqModel` 加 causal mask，`scripts/eval.py` 增加 `--mode teacher_forcing|greedy|both` 并分别落盘；在 `outputs/california_housing/run_20260120_213920/` 上评估结果为：teacher_forcing (MAE 0.3776 / RMSE 0.5285 / PICP 0.5824 / MPIW 0.7686)，greedy (MAE 0.9457 / RMSE 1.3950 / PICP 0.3704 / MPIW 0.4694)。
- M2 结构升级：新增 `FeatureTokenizer`/`TabularEncoder`，`TransformerTabSeqModel` 改为 Transformer Decoder + cross-attention，并加入 `dec_pos_emb` 与因果 `tgt_mask`，输出形状仍是 `(B, depth, n_bins)`；评估脚本兼容新旧模型结构。
- 细化与兼容：`FeatureTokenizer` 增加输入形状校验与 dtype 对齐，类型标注保持 Python 3.9 兼容；`scripts/eval.py` 可读取 `cat_cardinalities`（如配置里提供）。
- 说明：本次是“评估口径对齐”（防止 teacher_forcing 偷看未来、让 greedy 成为真实推理口径），不算模型结构升级；`docs/EXECUTION.md` 已要求推理阶段用 greedy，但 notebook 里尚未明确写出这一点。
- M2 结构升级（表格编码器 + 解码器）：新增 `FeatureTokenizer`（数值列 token = x_i*w_i+b_i；可选类别 embedding）与 `TabularEncoder`（对特征 token 做 Transformer 编码），并将 `TransformerTabSeqModel` 改为“解码器自回归 + cross-attention 读上下文 + 位置 embedding + 因果掩码”；旧 checkpoint 需重新训练。
- M2 复验（新结构闭环）：用 `PYTHONPATH=src python3 scripts/train.py` 重新训练并以 `PYTHONPATH=src python3 scripts/eval.py --mode both` 评估；在 `outputs/california_housing/run_20260120_234258/` 上得到 teacher_forcing (MAE 0.0635 / RMSE 0.0873 / PICP 0.7408 / MPIW 0.2534) 与 greedy (MAE 0.4192 / RMSE 0.6200 / PICP 0.2129 / MPIW 0.2244)。
- 代码审阅修正：`FeatureTokenizer` 增加输入列数校验与 dtype 对齐，类型标注保持 Python 3.9 兼容；`scripts/eval.py` 读取 `cat_cardinalities`（如 config 提供），避免新旧结构误配。

## 2026-01-21

- DACA 讨论（设计思路）：DACA = “按解码步数调控上下文信息流”的门控机制；PDF 写法为 `G_t = sigmoid(MLP(E_pos(t)))` 与 `K_t = X_ctx ⊙ G_t`，notebook 未覆盖这一段。
- DACA 实现结构（保持接口不变）：`FeatureTokenizer + TabularEncoder` 产出 `X_ctx`；基于位置向量生成 `G_t`；每个步 t 用 `X_ctx ⊙ G_t` 作为 cross-attention 的 memory；自回归 self-attention + cross-attention + FFN，输出仍为 `(B, depth, n_bins)`。
- Review 结论（对齐）：当前实现已是“只 gate Key、不 gate Value”，与 PDF 的 `K_t = X_ctx ⊙ G_t` 一致。
- Review 结论（效率）：当前 memory 扩展为 `(B, T, S, D)`，深度/特征数大时会有显存与速度开销；后续可把 gate 融入 key 投影避免显式复制。
- Review 结论（一致性）：`G_t` 复用 `dec_pos_emb` 作为 `E_pos(t)`，与 PDF 的位置编码描述一致。
- Review 结论（阈值语义）：PDF 中 “t<=3 / t>=8” 属于机制效果的描述性直觉，公式未给出硬阈值或静态/动态特征分组；当前决策是按公式做软门控，不硬编码阈值（若要硬约束需先定义特征分组与阈值规则）。
- 快速校验：随机输入前向形状正确（`torch.Size([2, 5, 32])`）。
- M2 复验（DACA 版本闭环）：用 `PYTHONPATH=src python3 scripts/train.py` 重新训练并以 `PYTHONPATH=src python3 scripts/eval.py --mode both` 评估；在 `outputs/california_housing/run_20260121_100226/` 上得到 teacher_forcing (MAE 0.0599 / RMSE 0.0786 / PICP 0.7418 / MPIW 0.2513) 与 greedy (MAE 0.4321 / RMSE 0.6331 / PICP 0.2185 / MPIW 0.2306)。
- 温度校准（greedy 粗扫）：在同一 ckpt 上扫 `T=0.5~10`，PICP 随 T 上升；取 `T=6.5` 时 PICP 0.9067 最接近 0.90，但 MPIW 变宽到 2.1508（可靠性达标、区间变宽的权衡）。
- 反思（主线优先级）：此前误把 `docs/EXECUTION.md` 当作最高优先级主线；正确做法是以 PDF 为最高对齐标准。
- 讨论（缺口定位）：PDF 2.3.2 Adaptive Confidence Masking 曾是 M2 方法缺口，已在本次补齐；评估侧校准不能替代 ACM。
- 讨论（影响判断）：greedy 低 PICP 与“自回归误差累积”仍显著相关；ACM 更偏训练口径的可靠性修正，需在其落地后再比较温度校准收益。
- 讨论（执行逻辑）：坚持“PDF-first”顺序——先补齐 ACM（训练损失权重），再做温度校准与 A/B 对比，避免先校准后改损失造成结果不可比。
- ACM 标准化落地：实现 `α(x,t)=α_depth(t)*α_instance(x)`，其中 `alpha_depth` 线性递增固定调度、`alpha_instance` 由 `X_ctx` 均值池化+MLP+sigmoid 得到；损失仅对负样本分支用 `1-α` 权重；在 `outputs/california_housing/run_20260121_110922/` 上复验，teacher_forcing (MAE 0.0701 / RMSE 0.0883 / PICP 0.7626 / MPIW 0.2765)，greedy (MAE 0.4427 / RMSE 0.6555 / PICP 0.2190 / MPIW 0.2447)，说明 ACM 提升训练口径覆盖率但未显著改善 greedy 覆盖率。
- ACM 版本温度校准（greedy 扫描）：在 `outputs/california_housing/run_20260121_110922/` 上扫 T∈{1,2,4,6,6.5,7,8,10}，PICP 随 T 单调上升；T=6.5 时 PICP≈0.8985 最接近 0.90，MPIW≈2.0418；T=7 时 PICP≈0.9174、MPIW≈2.3521。结论：校准能把可靠性拉回目标，但区间明显变宽。
- M2 收尾（校准版口径）：确认 DACA+ACM 已对齐 PDF，greedy 经 T=6.5 校准后 PICP≈0.90 达标；以该温度作为本轮评估口径基准，进入 M3 同时明确 MPIW 变宽的 tradeoff。
- ACM 稳定性补丁：训练端对 `alpha` 做 `clamp(0,1)` 避免负样本权重反向；`alpha_depth` 仅在初始化时搬到 device 并记录完整向量到配置，减少重复拷贝并提升复现性。
- ACM 细化：训练时缓存 `alpha_depth` 的广播视图，去掉每个 batch 的重复 `.to(device)` 判断，降低微小开销并保持行为一致。
- 工程判断与策略（对齐 PDF + 落地改进）：DACA 与 ACM 已对齐 PDF 主体，但 greedy 口径仍低，核心问题是“训练用真实前缀 vs 推理自回归”的暴露偏差。后续优先以 greedy 为主口径做 A/B：1) 先做温度校准（最小改动，提升可靠性但区间变宽）；2) 再引入训练策略（如 prefix dropout / scheduled sampling）减小口径差距；3) 固定 split/seed/confidence，保留 ACM 开关做消融，避免改动叠加导致不可比。
- 简化项清单（仍可提升）：ACM 的 `alpha_instance` 当前为样本标量（可扩展到按深度 `(B, depth, 1)`）；`alpha_depth` 为固定线性调度（可改为可学习参数并记录向量）；基线区间校准目前用训练残差分位数（更严格可改为独立校准集）。
- Tradeoff 补充（术语+直觉）：PICP=覆盖率（可靠性），MPIW=区间宽度（信息量）；“校准/温度缩放”通常让 PICP 上升但 MPIW 变宽；“训练口径 vs 推理口径”差异属于 exposure bias（暴露偏差），会导致 greedy 指标显著下降；ACM 通过降低非目标分支惩罚来保留不确定性，可能抬高 PICP 但牺牲点预测精度与 MPIW。工程上必须固定数据切分/seed/confidence，并用消融对比区分“结构收益”和“口径收益”。
- Tradeoff 展开（通俗+专业）：可以把区间预测看成“范围报数”——范围越宽越容易覆盖真实值（PICP↑，可靠性更高），但信息量下降（MPIW↑）；范围越窄越精确（MPIW↓），但更容易漏掉真实值（PICP↓）。温度校准属于“最小代价修正”，几乎一定是 PICP↑、MPIW↑。ACM 属于“训练时保留不确定性”，通常提高覆盖率但可能牺牲点预测（MAE/RMSE）或增大区间宽度。训练口径（teacher forcing）比真实推理口径（greedy）更乐观，差异来源是 exposure bias（前缀依赖导致误差累积）。策略上应先让 greedy 的 PICP 达标（可靠性优先），再在不掉 PICP 的前提下压 MPIW/MAE，并用消融分离“结构改进”和“校准改进”的贡献。
- M3 差距评审（battle 点位）：
  - 训练脚本目前是“单模型 + 单数据集 + 硬编码超参”，缺少可切换的 baseline 入口与可复现实验配置（`scripts/train.py:17`）。
  - 评估脚本绑定 TabSeq 流水线（必须有 TraceLabelEncoder/dec_input），所以无法直接评估 MLP/分位数回归这种只输出点预测/区间的基线（`scripts/eval.py:24`）。
  - 指标实现只接受 step_probs→leaf_probs 的分布输入，因此基线模型需要一个“通用评估入口”（点预测/区间）才能对齐同一口径（`src/tabseq/metrics/holographic.py:15`）。
- M3 标准化策略（工业级执行）：
  - 基线动机：MLP=朴素点预测参照；分位数回归=直接区间参照；与 TabSeq 形成“点/区间/结构”三角对照，证明改进是否真实。
  - 统一协议：固定数据切分/标准化/随机种子/置信度，所有模型复用同一份 split 与 v_min/v_max/ depth 设定，确保“同口径可比”。
  - 双通道评估：TabSeq 走 step_probs→leaf_probs→指标；基线走 y_hat 或 [L,U]→指标，输出同样的 MAE/RMSE/PICP/MPIW 字段。
  - 产物规范：每个实验目录至少有 `config`、`metrics_val.json`、`checkpoint`、`git.txt`；新增汇总脚本生成 `benchmark.csv`。
  - 自查点位：同一数据切分与 scaler；同一 confidence；指标函数一致；运行可复现（固定 seed）；对比表可直接复盘。
  - 范围声明：M3 只负责“基线与汇总对比”，不承接 M2 的 ACM/温度改动；如需纳入对比，统一在汇总表里标注是否校准/是否启用 ACM。
- M3 适配 M2‑ACM：`scripts/benchmark.py` 汇总表新增 `temperature/acm_enabled/alpha_* /seed` 字段，`scripts/eval.py` 在指标里写入 `model`，确保 TabSeq‑ACM 版本与基线行在同一表里可追踪区分。
- M3 基础设施：新增统一数据加载 `src/tabseq/data/datasets.py`、随机种子工具 `src/tabseq/utils/seed.py`、基线模型 `archive/src/tabseq/models/baselines.py`、通用回归指标 `src/tabseq/metrics/regression.py`，并加入 `archive/scripts/train_baseline.py`/`archive/scripts/eval_baseline.py` 与 `scripts/benchmark.py`。
- M3 基线复验：在 `outputs/baselines/baseline_mlp_20260121_104151/` 得到 MLP (MAE 0.3979 / RMSE 0.5722 / PICP 0.8951 / MPIW 1.7256)；在 `outputs/baselines/baseline_quantile_20260121_104212/` 得到 Quantile (MAE 0.4659 / RMSE 0.8688 / PICP 0.9147 / MPIW 1.8460)；并生成 `outputs/benchmark.csv` 汇总表。
- M3 汇总对齐（ACM/温度）：将 `outputs/california_housing/run_20260121_110922/` 以温度 `T=6.5` 的 greedy 指标写入 `metrics_val.json` 并更新 `outputs/benchmark.csv`，表中记录 `temperature=6.5` 与 `acm_enabled=True` 便于对比。
- M3 结构基线补齐：新增 Tabular Transformer 基线并复验，在 `outputs/baselines/baseline_tab_transformer_20260121_120933/` 得到 (MAE 0.3656 / RMSE 0.5255 / PICP 0.8891 / MPIW 1.5425)；`outputs/benchmark.csv` 已更新。
- M3 FT‑Transformer 对齐：按 PDF/背景论文的 “FeatureTokenizer + Transformer + CLS” 结构调整基线，实现 `ft_transformer` 并复验，在 `outputs/baselines/baseline_ft_transformer_20260121_140236/` 得到 (MAE 0.3590 / RMSE 0.5302 / PICP 0.8961 / MPIW 1.5557)；`outputs/benchmark.csv` 已更新。
- M3 边界说明：FT‑Transformer 属于“基线对比”（M3），不是 TabSeq 主模型（M2）；M2 的 DACA 直接使用 token 级上下文，因此不需要 CLS 聚合，与 PDF 的异构编码描述不冲突。
- M3 基线校准更严：基线区间宽度从“训练残差分位数”改为“校准集残差分位数”，并新增 `calibration_fraction` 记录；`tab_transformer` CLI 保留但仅作为 `ft_transformer` 兼容别名并打印警告。
- M3 FT‑Transformer 校准复验：基于校准集残差，`outputs/baselines/baseline_ft_transformer_20260121_151533/` 得到 (MAE 0.3665 / RMSE 0.5369 / PICP 0.9082 / MPIW 1.6829)；`outputs/benchmark.csv` 已更新。
- M3 汇总表更严：`scripts/benchmark.py` 支持 `--seed` 过滤与 `--include-extras` 输出扩展诊断（`width_stratified_PICP`/`bin_acc_0.2/0.4`），用于生成干净交付表。
- TabSeq seed=42 对齐：新增 `TABSEQ_SEED` 环境变量以便复现种子控制；在 `outputs/california_housing/run_20260121_162451/` 上用 `T=6.5` 评估得到 greedy (MAE 0.4222 / RMSE 0.5978 / PICP 0.9261 / MPIW 2.3492)，并更新 `outputs/benchmark_seed42.csv` 对齐基线。
- 讨论（评估口径）：greedy 的 PICP 仅 0.2185，明显低于 0.90 目标；进入“评估侧优化”的信号已出现，优先考虑温度校准或训练时的校准策略。
- 讨论（实现代价）：DACA 里每步 gate 都要作用到上下文 token，最直接的写法会把 memory 扩到 `(B, T, S, D)`；这是“每步都有一份被 gate 的上下文”的原因，后续可把 gate 融入 key 投影避免显式复制。
- 讨论（主线阶段）：结构升级（表格编码器 + 解码器 + DACA）已完成，当前属于“评估侧优化”阶段；后续若做结构改动应以对齐 PDF 与提升 greedy 为前提。
- 讨论（温度校准必要性）：greedy 的 PICP 明显偏低时，温度校准是最小代价的可靠性修正（不改结构，只调 `sigmoid(logits / T)`）。

## 2026-01-24

- 架构文档：新增 `docs/ARCHITECTURE.md`，按“数据→编码→模型→评估→汇总”梳理全链路文件与职责，便于从架构视角定位入口与依赖关系。
- 注释规范：新增 `docs/COMMENTING_GUIDE.md`，统一“用途/设计意义/使用场景/字段含义/指标含义/层间关系”的注释标准，作为后续所有注释的模板。
- 烟测脚本注释增强：`scripts/smoke_dataset.py` 补充设计意义、占位维度理由（3x4/3x2）、示例 y 值的范围约束、`ds[0]` 的取样目的，以及字段/形状/用途解释。
- 数据链路烟测通过：`PYTHONPATH=src python scripts/smoke_dataset.py` 输出包含 `x_num/x_cat/dec_input/y_seq/y_mht/y_raw`，形状为 `x_num=(4,)`、`x_cat=(2,)`、`dec_input=(6,)`、`y_mht=(6,64)`，说明 `depth=6` 且 `n_bins=2^6=64` 的标签编码与 batch 封装正确。
- 主线训练通过：运行 `python scripts/train.py` 生成 `outputs/california_housing/run_20260124_104159/checkpoint.pt`，训练 loss 从 0.57 降到 ~0.09，说明训练闭环可用；swanlab 记录保存在 `swanlog/run-20260124_104202-a1zrkyw6i9wwfo08vmc9h`。

## 2026-01-25

- 指标代码注释补强：`src/tabseq/metrics/holographic.py` 增补“术语速记 + 设计原因 + CDF/width_bins 解释”，方便新人理解“leaf/bin/step_probs/leaf_probs/CDF/PICP/MPIW”等概念与指标的衔接逻辑。
- 讨论澄清：PICP 随温度上升主要来自“分布变平→分位点间距变大→区间变宽”的正常现象，不是 CDF 计算错误或“区间被累加”的 bug。
- 讨论澄清：单个 bin 宽度固定，但 [L, U] 可能跨多个 bin，所以“区间宽度”不是常数；`width_bins` 只是按区间宽度做诊断分组，不参与区间生成。
- 温度诊断报告：新增 `docs/TEMPERATURE_DIAGNOSIS_20260125.md`，记录 logit 尺度、概率饱和率、叶子分布熵与温度扫参关键点，用数据解释“为什么 T=1 不够”。 

## Tradeoff 说明（直白版，便于复盘）

- 天气预报类比：想保证“90% 的时候预报包含真实温度”，就得报一个更宽的区间；这样覆盖率高但不精确。报很窄的区间更精确，但更容易漏掉真实值。
- PICP vs MPIW：PICP 高=更可靠（覆盖率高），但通常需要更宽的区间（MPIW ↑）；MPIW 低=更精确（区间窄），但覆盖率会下降（PICP ↓）。两者不可同时极大化。
- Teacher forcing vs Greedy：训练时用真实前缀更容易收敛，但推理时只能自回归生成，错误会累积（exposure bias）。使用更贴近推理的训练策略会更难训，但可能提升真实推理表现。
- 温度缩放与 ACM：温度缩放是最小代价的可靠性修正，通常让 PICP ↑、MPIW ↑；ACM 通过减小非目标分支惩罚保留不确定性，可能提高覆盖率，但点预测精度与区间宽度可能变差。
- 结构复杂度权衡：DACA/ACM 提升建模能力，但参数与计算量增加，训练不稳定风险也更高；需要消融对比确认收益。
- 示例场景：若预测股票价格区间，选高 PICP 可提高“包含真实值”的可靠性，但区间会变大；追求更小 MPIW 则更精确但漏报风险更高。
- 当前结果示例：teacher forcing PICP 约 0.76，greedy 仅约 0.22；温度缩放到 T≈6.5 可把 PICP 拉到 0.90，但 MPIW 变宽到约 2.15，体现覆盖率与区间宽度的权衡。
- 执行建议：先设定目标覆盖率（如 PICP=0.90）；调温度或 ACM 以达到覆盖率；以 greedy 口径为主评估；固定 split/seed/confidence 做 A/B 消融，分离“结构改进”和“校准改进”的贡献。

## Tradeoff 超简版（只记 4 句话）

- 你要“更准”，区间就要变宽；你要“更窄”，就更容易漏掉真值。
- teacher forcing 指标更好，是因为训练时用了真前缀；真实推理没有真前缀，指标会掉。
- 温度调大能提高覆盖率，但区间会变更宽（更保守）。
- 先把 greedy 的覆盖率拉到目标，再谈怎么让区间变窄。

## Tradeoff 逻辑链（详细版，按因果顺序）

- 定义先看清：PICP 是“真实值落在预测区间里的比例”；MPIW 是“预测区间平均宽度”。区间越宽，覆盖率越高；区间越窄，覆盖率越低。
- 为什么必然冲突：如果模型输出分布更“发散”，CDF 上的分位点会更远，区间变宽，PICP 上升；如果分布更“尖锐”，区间变窄，PICP 下降。这是概率分布本身的结构性权衡，不是代码问题。
- teacher forcing vs greedy：训练时给真前缀（teacher forcing）=“喂答案开卷”，推理时不给真前缀（greedy）=“闭卷自写”。闭卷时前几步错了，后续会累积误差，导致 greedy 指标显著下降，这就是 exposure bias。
- 温度缩放 T 的作用：把 `sigmoid(logits / T)` 做“拉平”。T 越大 → 概率更接近 0.5 → 分布更发散 → 区间更宽 → PICP 上升，但 MPIW 也上升。
- ACM 的作用：对 `y_mht=0` 的分支减少惩罚，让模型“不必过分确信排除某些叶子”，结果是分布更发散 → PICP 可能上升，但 MPIW 可能变宽、点预测可能变差。
- 现实例子：当前 greedy PICP ≈0.22 很低，说明区间太窄；当 T 调到 6.5 时 PICP≈0.90，但 MPIW≈2.15 变宽，体现“覆盖率↑↔区间宽度↑”的必然权衡。
- 工程顺序：先把 greedy 的 PICP 拉到目标（可靠性优先），再在不掉 PICP 的前提下压 MPIW/MAE；所有改动都要固定 split/seed/confidence 并做消融，避免“结构改进”和“校准改进”混在一起。

## 术语小抄（给新人/汇报用）

- M1 最小闭环：先不追最好效果，只验证“数据→标签→模型→评估→落盘”全链路能跑通。
- California Housing：一个公开表格回归数据集；输入是 8 个数值特征 X，输出是连续标签 y（房价相关）。
- MLP：多层感知机，本质是“几层全连接(Linear)+非线性(ReLU)”的网络。
  - 输入：一个向量（例：8 个数值特征）
  - 输出：一个向量（例：64 维隐藏表示）
- logits：模型的“原始打分”（还没变成 0~1 概率的数）。
- sigmoid：把 logits 变成 0~1 的概率；例：sigmoid(0)=0.5，sigmoid(大正数)≈1。
- BCEWithLogitsLoss：二分类/多标签常用损失；输入是 logits，目标是 0/1；内部自带 sigmoid（更稳定）。
- MAE：平均绝对误差；例：真实 3.0，预测 2.5，误差是 |2.5-3.0|=0.5。
- MSE：平均平方误差；例：误差 0.5 的平方是 0.25。
- RMSE：均方根误差 = sqrt(MSE)；单位和 y 一样，便于直觉理解（比 MSE 更常汇报）。
- checkpoint：训练过程中保存的模型文件（参数 + 配置）；用于后续 eval、继续训练、复现实验。
- DACA（Depth-Aware Contextual Attention）：按解码步数生成门控 `G_t`，让“第 t 步该看哪些上下文特征”可学习。
- gate（门控）：一组 0~1 权重，用来放大/缩小向量的不同维度。
- ACM（Adaptive Confidence Masking）：训练阶段的损失加权机制，用 `α(x,t)` 降低非目标分支的惩罚，增强不确定性表达。
- alpha_depth：形状 `(depth,)`，随深度变化的权重调度。
- alpha_instance：形状 `(B,1,1)` 或 `(B,depth,1)`，由样本上下文预测的权重。
- FT-Transformer：来自 `pre/Revisiting Deep Learning Models for Tabular Data.pdf` 的通用表格模型（列 token → Transformer Encoder → CLS 汇聚 → 回归）；在本项目中仅作为 M3 对比基线，不是 TabSeq 主模型。
- CatBoost/RealMLP 分位数基线：来自 `pre/quantile_regression_extended_benchmark.ipynb`，直接输出上下分位点 `[L,U]` 用于 PICP/MPIW；已工程化为 `archive/scripts/run_baseline_catboost.py` 与 `archive/scripts/run_baseline_realmlp.py`。

## M2 设计策略（DACA 对齐与优化）

- 设计逻辑：每个解码步 t 都有自己的 gate，代表“这一层该强调哪些上下文维度”（由粗到细）。
- 最小实现：`G_t = sigmoid(MLP(E_pos(t)))`，用 `X_ctx ⊙ G_t` 作为该步 cross-attention 的 memory。
- 严格贴 PDF：只 gate Key、不 gate Value；位置编码可复用 `dec_pos_emb`。
- 工程优化：避免显式 `(B, T, S, D)` 复制，把 gate 融入 key 投影以减小显存开销。
- 评估阶段策略：当 greedy 的 PICP 明显低于目标（如 0.90），优先做温度校准以先把可靠性拉回目标附近，再比较结构改动。
- 主线约束：不改变 “表格编码器 + 解码器 + DACA” 的主线结构，优先在评估侧（温度校准）与实现侧（内存优化）改进。
- PDF-first 执行：先补齐 ACM（训练损失加权），再做温度校准，确保方法对齐与实验可比性。

## M1 闭环到底用了什么（我们当前实现）

- 用的模型：`src/tabseq/models/minimal_model.py`（最简版）
  - 表格部分：只用数值特征 `x_num`，用一个小 MLP 编成上下文向量（没用类别特征 `x_cat`）。
  - 序列部分：把 `dec_input`（长度=depth 的 token：SOS/0/1）做 embedding，然后加上表格上下文，再用线性层输出每一步对所有桶的 logits（形状 `(B, depth, n_bins)`）。
- 训练数据：California Housing 的训练集；标准化后得到 `X_num`；标签 `y` 用 `TraceLabelEncoder` 编码成 `y_mht`（multi-hot 监督目标）。
- 训练优化目标：让模型输出的 logits 经过 sigmoid 后，尽量贴近 `y_mht`（用 `BCEWithLogitsLoss`）。
- 评估（当前只做点预测）：把每一步概率“相乘并归一化”得到每个桶的分布，再用桶中心点做期望得到 `y_hat`，最后算 MAE/RMSE。

## M2 闭环到底用了什么（我们当前实现）

- 用的模型：`src/tabseq/models/transformer_model.py`（DACA + ACM）
  - 表格编码：`FeatureTokenizer` + `TabularEncoder` 产出 `X_ctx`（上下文 tokens）。
  - 序列解码：DACA Decoder（自回归 + cross-attention），带位置编码与因果 mask；gate 按 `G_t = sigmoid(MLP(E_pos(t)))` 生成，仅作用在 Key（对齐 PDF）。
  - 置信权重：ACM 通过 `alpha_depth`（线性调度）× `alpha_instance`（`X_ctx` 均值池化 + MLP + sigmoid）生成样本权重。
- 训练数据：California Housing；`TraceLabelEncoder` 生成 `y_mht` multi-hot 监督。
- 训练目标：`BCEWithLogitsLoss(reduction="none")` 后按权重加权；
  - `y_mht=1` 权重恒为 1；
  - `y_mht=0` 权重为 `1-α`，且 `α` clamp 到 `[0,1]`。
- 评估口径：同时输出 teacher_forcing 与 greedy；greedy 采用温度校准（当前口径基准 T=6.5）。
- 产物示例：`outputs/california_housing/run_20260121_110922/`（checkpoint + metrics）；校准后 greedy PICP≈0.90，但 MPIW 变宽，属于可靠性/信息量 tradeoff。
- 阶段结论：结构已对齐 PDF 2.2/2.3.2，训练闭环完成；进入 M3 以前需把“校准口径”固定到元信息里，保证可复现对比。

## M3 闭环到底用了什么（我们当前实现）

- 目标：把 notebook 里的基线补齐并统一到同一评估口径，生成可汇报的对比表（benchmark）。
- 基线清单：MLP、Quantile MLP、FT‑Transformer（CLS 汇聚）、CatBoost（分位数）、RealMLP（可选依赖）。
- 统一入口：`archive/scripts/train_baseline.py` / `archive/scripts/eval_baseline.py` / `archive/scripts/run_baseline_*`，每个实验目录落 `checkpoint/config/metrics/git.txt`。
- 评估口径：基线走通用指标函数输出 MAE/RMSE/PICP/MPIW，可选输出 `width_stratified_PICP` 与 `bin_acc_0.2/0.4` 作为诊断。
- 校准更严格：基线区间宽度用“校准集残差分位数”（`calibration_fraction`）而非训练残差，避免 PICP 偏乐观。
- 汇总交付：`scripts/benchmark.py` 汇总 `outputs/**/metrics_val.json` 输出 `benchmark.csv`（支持 `--seed/--include-extras/--latest` 生成干净交付表）。
- 阶段结论：M3 完成“基线齐全 + 同口径对比表 + 可复现配置”；与 M2 主线结构不冲突，仅作为对比基线。

## M3 闭环到底用了什么（我们当前实现）

- 目标：补齐基线并统一口径，产出“可对比、可复现”的对照表（MAE/RMSE/PICP/MPIW）。
- 基线覆盖：
  - CatBoost 分位数基线：`archive/scripts/run_baseline_catboost.py`
  - RealMLP 分位数基线（pytabkit）：`archive/scripts/run_baseline_realmlp.py`
  - 简单 MLP：`archive/scripts/train_baseline.py`（mlp）
  - 分位数回归：`archive/scripts/train_baseline.py`（quantile）
  - FT‑Transformer：`archive/scripts/train_baseline.py`（ft_transformer）
- 评估口径统一：基线统一走 `src/tabseq/metrics/regression.py`，TabSeq 走 `src/tabseq/metrics/holographic.py`，都输出 MAE/RMSE/PICP/MPIW；基线额外输出 `width_stratified_PICP` 与 `bin_acc_0.2/0.4`。
- 汇总对比表：`scripts/benchmark.py` 汇总所有 `metrics_val.json`，生成 `outputs/benchmark.csv`（包含 model/mode/temperature/acm/seed 等元信息）。
- 复现实验口径：统一 seed=42（基线已重跑），TabSeq‑DACA+ACM 采用 greedy 且温度校准 T=6.5（PICP≈0.90），作为当前“对比口径”行写入 `outputs/benchmark.csv`。



## 区间指标（PICP/MPIW）与优化思路（待实现/用于后续模型优化）

- PICP（Prediction Interval Coverage Probability，预测区间覆盖率）
  - 输入：每个样本的预测区间 `[L_i, U_i]` + 真实值 `y_i`
  - 定义：`PICP = mean( 1{ L_i <= y_i <= U_i } )`
  - 直觉：你说“这是 90% 区间”，那在很多样本上，真实 y 应该大约有 90% 落进去。
  - 目标：PICP ≈ 你设定的置信度（例如 0.90）。
    - PICP < 0.90：区间偏窄/偏乐观（覆盖不够，可靠性差）
    - PICP > 0.90：区间偏宽/偏保守（覆盖够但信息量低）

- MPIW（Mean Prediction Interval Width，预测区间平均宽度）
  - 输入：每个样本的预测区间 `[L_i, U_i]`
  - 定义：`MPIW = mean( U_i - L_i )`
  - 直觉：区间越宽，越“保守”；区间越窄，越“尖锐/有信息量”。
  - 目标：在 PICP 达标（≈0.90）前提下，MPIW 越小越好。

在本项目里的“可操作优化路径”（先定尺子再改模型）：
- Step 0（先立尺子/统一口径）：先把指标做出来（否则无法谈“优化”）
  - 代码动作：从 `pre/tabseq_trace_design.ipynb` 抽 `ExtendedHolographicMetric` 到 `src/tabseq/metrics/holographic.py`
  - 接入动作：在 `scripts/eval.py` 里，基于模型输出的分布计算 `[L,U]`，输出 `PICP/MPIW` 到 `metrics_val.json`

- Step 1（先把 PICP 拉到目标附近：可靠性优先）
  - 优先用“后处理校准”而不是盲目改模型：
    - Temperature scaling：把 `sigmoid(logits)` 改成 `sigmoid(logits / T)`（T>0）
    - 操作方式：用验证集搜索一个 T（比如 0.5~5.0），让 PICP 最接近 0.90
  - 解释：这一步主要修“概率不准/过自信”的问题，让区间覆盖率先站稳。

- Step 2（在 PICP 不掉的前提下压 MPIW：区间更窄更有信息量）
  - 模型能力：用更强的表格编码器/序列模型（M2：tabular embedding + Transformer 方向）
  - 训练信号：在保持覆盖率的同时让分布更集中（减少“到处都可能”的不确定性）
  - 常见现象：模型更强后，点预测 MAE/RMSE 往往会降，同时 MPIW 也更容易压下来。

- Step 3（如果必须“保证覆盖率”：用共形预测作安全带）
  - 目的：让 PICP 在统计意义上更稳地 ≥ 目标（比如 0.90）
  - 代价：区间通常会变宽（MPIW 上升），但可靠性更强；适合做“最终可交付”的版本。

## 温度缩放（Temperature Scaling）——你刚加的 `sigmoid(logits / T)` 是什么

- 位置：评估阶段（`scripts/eval.py`），不改训练、不改模型结构。
- 做法：把原来的
  - `step_probs = sigmoid(logits)`
  改成
  - `step_probs = sigmoid(logits / T)`，其中 `T > 0`。

怎么理解 T（用直觉记住）：
- `T = 1`：不做任何校准（原样）
- `T > 1`：把 logits “压小” -> sigmoid 输出更靠近 0.5 -> 模型更不自信/更保守
  - 通常效果：区间会变宽（MPIW ↑），覆盖率会上升（PICP ↑）
- `T < 1`：把 logits “放大” -> sigmoid 更接近 0/1 -> 模型更自信
  - 通常效果：区间会变窄（MPIW ↓），覆盖率会下降（PICP ↓）

为什么要试 T=1/2/3 三个值：
- 因为我们当前 PICP 明显低于目标（例如 0.90），说明区间偏窄/模型偏自信。
- 先用 `T=1,2,3` 做一个“粗扫”：看 PICP 能不能明显往目标靠近；如果有效，再在附近细调（例如 1.5、2.5）。

这一步的目标（对齐逻辑）：
- 第一优先：让 PICP 接近设定置信度（例如 0.90，先把“可靠性”站稳）。
- 第二优先：在 PICP 站稳后，再想办法压 MPIW（让区间更窄更有信息量）。

## 当前阶段结论：温度“能改善 PICP”，但还没算最终解决

在当前 eval 口径下（还没做 greedy 推理与 causal mask 对齐），对同一个 Transformer checkpoint 做温度粗扫的现象是：
- T 从 1 -> 2 -> 3：PICP 明显上升（更接近 0.90），MPIW 同时变宽（更保守）；MAE/RMSE 小幅变化。

重要说明（避免汇报跑偏）：
- 温度缩放是“评估/后处理校准”，不是模型结构升级。
- 它只有在“推理阶段对齐”后才算真正闭环：下一步要先实现真实推理（greedy 自回归）+ Transformer causal mask；
  然后再在 greedy 指标上重新选 T，才可以说“温度校准把 PICP 拉到了目标附近”。

## 推理阶段对齐（Teacher Forcing vs Greedy）——为什么要做

你会遇到两个“评估口径”：
- teacher_forcing（训练口径）：eval 时也喂真实前缀 `dec_input = [SOS] + 真实bit[:-1]`，模型更容易做对；它是一个“上限/调试口径”。
- greedy（部署口径/真实推理）：eval 时只有 X，没有 y；只能从 `[SOS]` 起步，让模型自己一步步生成 bit 前缀，再完成整条序列；这是“真正要汇报/对比”的口径。

为什么需要 Transformer 的 causal mask：
- 如果不加 mask，Transformer 在 teacher_forcing 下可能“看到未来 token”（未来真值），指标会偏乐观。
- causal mask 强制第 t 步只能看见历史（包括 ctx 与前缀 token），与自回归推理一致。

工程动作（M2 主线的一步）：
- `src/tabseq/models/transformer_model.py`：加入 causal mask（防止看未来）。
- `scripts/eval.py`：加入 `--mode teacher_forcing|greedy|both`，同一个 ckpt 输出两套指标，直观看到“训练口径 vs 部署口径”的差距。

## 2026-01-21-afternoon

- PDF 核对：`pre/tabseq.pdf` 2.2.2 明确写出 `G_t = σ(MLP(E_pos(t)))` 与 `K_t = X_ctx ⊙ G_t`；并用“t≤3 / t≥8”描述直觉，但未给硬性阈值或显式特征分组。当前实现保持软门控，不硬编码阈值（若要硬约束需先定义静态/动态特征分组）。
- 对齐澄清：M2 的 “异构编码” 使用 `FeatureTokenizer + TabularEncoder`，这是 TabSeq 主模型的一部分；M3 的 `ft_transformer` 是独立基线用于对比，不与 M2 冲突。
- M3 仍可补齐：notebook 的 CatBoost/RealMLP 基线未工程化；当前数据仅数值特征，`x_cat` 实际接入仍为空。
- M3 工程化补齐：新增 CatBoost 分位数基线脚本 `archive/scripts/run_baseline_catboost.py`，与 notebook 对齐为 90% 区间量化；输出 `metrics_val.json` 与 `config.json`，可被 `scripts/benchmark.py` 汇总。
- M3 基线复验：运行 CatBoost 基线在 `outputs/baselines/baseline_catboost_20260121_150825/`，得到 MAE 0.3746 / RMSE 0.5179 / PICP 0.8389 / MPIW 1.2730，并写入 `outputs/benchmark.csv`。
- M3 工程化补齐：新增 RealMLP 分位数基线脚本 `archive/scripts/run_baseline_realmlp.py`（`pytabkit`），默认 90% 区间并可设置 `--epochs/--batch-size`；输出 `metrics_val.json` 与 `config.json`，可被 `scripts/benchmark.py` 汇总。
- M3 基线复验：运行 RealMLP 基线在 `outputs/baselines/baseline_realmlp_20260121_151200/`（epochs=30），得到 MAE 0.3778 / RMSE 0.5328 / PICP 0.8840 / MPIW 1.3208，并写入 `outputs/benchmark.csv`。
- 命名修正：`archive/scripts/train_baseline.py` 里 `ft_transformer` 与 `tab_transformer` 的 run_dir/config 不再混用同一 `model` 名称，避免基线对比表里标签合并。
- M3 指标口径定案：对齐 `pre/tabseq.pdf` 的核心指标仅要求 MAE/RMSE/PICP/MPIW；分桶命中率/分宽度覆盖率作为可选诊断，不强制纳入基线（`scripts/benchmark.py --include-extras` 可按需导出）。
- M3 收口口径：以 seed=0 的 TabSeq‑DACA+ACM（greedy, T=6.5）为主线基准，基线表仅保留同 seed 的核心指标；`scripts/benchmark.py` 增加 `--models` 与 `--latest`，用于生成“每模型最新一次”的干净对比表。

## 2026-01-25

- 失败案例复盘（温度扫 PICP 过高怀疑）：检查 `scripts/eval.py` 的 greedy 计算与 `src/tabseq/metrics/holographic.py` 的区间提取后，未发现“CDF 累加”实现错误；PICP 随温度上升主要来自 `sigmoid(logits/T)` 使分布趋于均匀、MPIW 变宽的预期现象。
- 可能的评估偏差点：greedy 口径只沿单一路径生成 prefix，但仍对所有 bin 做 `step_probs` 连乘（`holographic.py:48`），这与“每个 bin 应有自己的条件前缀”不完全一致，可能放大温度对 PICP 的影响；需考虑改为“按 bin 前缀计算”或“采样多前缀”作对照。

## 2026-01-24

- Phase 1.1（工程化清理策略）：先做“资产分离 + 风险最小化”的整理，不直接删除产物；先盘点哪些是源码、哪些是产物、哪些是临时/缓存，再决定是否归档或清理。
- 工业级术语与策略（给后续执行对齐口径）：  
  - **Artifact segregation（产物隔离）**：训练产物统一放 `outputs/`，不混在源码目录。  
  - **Repo hygiene（仓库卫生）**：清理调试打印、临时文件、重复旧模型文件，避免噪音干扰 review。  
  - **Config externalization（超参外置）**：超参优先写入配置（或环境变量），减少“硬编码”引发的不可复现。  
  - **Deterministic runs（确定性实验）**：固定 seed/切分方式，并记录到 `config.json`。  
  - **Ignore policy（忽略策略）**：`.gitignore` 覆盖 data/outputs/cache 等非源码资产，避免误入版本库。
- FT-Encoder 落地（主线替换）：新增 `src/tabseq/models/ft_encoder.py`（PreNorm+GELU 的 FT 风格编码器）；`TransformerTabSeqModel` 改为使用 `FTEncoder` 作为特征提取器（替换原 `TabularEncoder`），解码器与评估口径保持不变。
- 训练复验（FT-Encoder 版本）：运行 `PYTHONPATH=src python3 scripts/train.py`，产出 `outputs/california_housing/run_20260124_171159/`；训练日志已写入 swanlab（run id `kvp0z70rbapjrv7xhog2k`）。
- 评估复验（greedy, T=1.0）：`outputs/california_housing/run_20260124_171159/metrics_val_greedy.json` 得到 MAE 0.4762 / RMSE 0.6961 / PICP 0.2069 / MPIW 0.2409；`outputs/benchmark.csv` 已更新包含该 run。
- 温度校准说明（评估侧校准，不改训练）：在评估时把 `sigmoid(logits)` 改为 `sigmoid(logits / T)`；T>1 会“拉平”概率，通常 **PICP 上升、MPIW 变宽**；T=1 为未校准基线。
- 指标含义补充（便于汇报）：MAE/RMSE 衡量点预测误差（RMSE 更惩罚大误差）；PICP 是区间覆盖率（目标≈0.90）；MPIW 是区间平均宽度（在 PICP 达标前提下越小越好）。

# 架构澄清：FT‑Feature 与主模型关系（学术术语对齐）

- 结论：主模型 **不是** FT‑Transformer；主模型只**复用 FT 风格的特征 token 化**，然后用 DACA 解码器进行序列预测。
- 术语对齐（学术说法）：  
  - **Feature Tokenization**：`FeatureTokenizer` 把每列特征变成一个 token（列级表征）。  
  - **Context Tokens / Representation**：`TabularEncoder` 把这些 token 编码成上下文表示 `ctx_tokens`。  
  - **Cross‑Attention Decoder**：DACA 解码器读取 `ctx_tokens` 作为 memory（不是 CLS 回归头）。  
  - **CLS Pooling（FT‑Transformer 基线）**：FT‑Transformer 用 `CLS` 汇聚后接回归 head，仅用于基线对比。  
- 对接链路（主模型）：`x_num/x_cat → FeatureTokenizer → TabularEncoder → ctx_tokens → DACA Decoder（cross‑attention） → logits`。  
- 对接链路（FT‑Transformer 基线）：`x_num/x_cat → FeatureTokenizer → TransformerEncoder → CLS pooling → regression head`。

# 变更方案草案：让 FT‑Feature 成为主模型“底座”

- 目标：把 FT‑Transformer 的 **encoder 表征** 作为 TabSeq 的上下文输入（保持解码器不变），形成“FT‑Feature + DACA 解码”的主线版本。
- 逻辑：  
  1) **替换/复用编码器**：保留 `FeatureTokenizer`，将 `TabularEncoder` 替换为 FT‑Transformer 的 encoder（不带回归 head）。  
  2) **上下文形式**：  
     - 方案 A（Token 级）：直接输出所有 tokens 作为 `ctx_tokens`（保持多 token）。  
     - 方案 B（CLS 级）：输出 `CLS` 向量并扩展为 1 个 context token（更轻量但信息更压缩）。  
  3) **保持解码器**：DACA Decoder 的 cross‑attention 接口不变，仍读取 `ctx_tokens`。  
  4) **训练/评估口径不变**：loss、metrics、eval 脚本不改，便于 A/B 对比。

# 细化对比：Token 级 vs CLS 级（概念与权衡）

- **Token 级（token‑level representation）**  
  - 输入/输出：每一列特征 → 对应一个 token；encoder 输出 `(B, S, D)` 的多 token 上下文（`S=特征数`）。  
  - 学术术语：保留 **per‑feature representation**，cross‑attention 能“逐列关注”。  
  - 优点：信息量最大；DACA gate 能对“不同步/不同特征”做细粒度调控。  
  - 代价：cross‑attention 计算量更大（与 `S` 成正比），显存占用更高。

- **CLS 级（CLS pooling / global pooling）**  
  - 输入/输出：把所有 token 汇聚成一个 `CLS` 向量；上下文只有 1 个 token（`S=1`）。  
  - 学术术语：**global representation / information bottleneck**（信息被压缩到一个向量）。  
  - 优点：更轻量，计算和显存开销小。  
  - 代价：细粒度特征信息被压缩；DACA gate 只能作用在“一个整体向量”，可表达性变弱。

- **工程建议**  
  - 如果追求性能与细粒度解释：优先 Token 级。  
  - 如果资源紧张或需要更快的原型：CLS 级更合适。

# FT‑Transformer 标准架构（基线口径）

- 标准结构（学术版）：  
  `FeatureTokenizer → TransformerEncoder → CLS pooling → Regression Head`  
- 关键概念：  
  - **CLS token**：用于“全局汇聚”的可学习向量（global representation）。  
  - **Pooling**：把多个 token 汇成一个向量（信息瓶颈）。  
- 与 TabSeq 主模型的区别：  
  - FT‑Transformer 输出单个回归值；TabSeq 输出序列 logits。  
  - FT‑Transformer 必须有 CLS pooling；TabSeq 主线不需要 CLS。

# 为什么考虑“FT‑Encoder 替换 TabularEncoder”（更详细、可落地的解释）

- 触发来源：新版 `docs/EXECUTION.md` Phase 2 明确“特征提取器替换”，并以 FT‑feature 为基础路径，再接 RealMLP 做替换/对比。  
- 当前主线的“特征提取器”到底是什么（输入/输出/例子）：  
  - **FeatureTokenizer**：  
    - 输入：一行表格特征 `x_num`（例如 8 个数）  
    - 输出：8 个 token（每列 1 个向量，形状 `(B, S, D)`）  
    - 例子：`[8列特征] → [8个token]`  
  - **TabularEncoder**：  
    - 输入：上一步的 token 序列 `(B, S, D)`  
    - 输出：上下文 token `ctx_tokens`（仍是多 token，但已“互相交流”）  
    - 例子：`[8个token] → [8个上下文token]`  
  - **作用位置**：`ctx_tokens` 会被 DACA 解码器用 cross‑attention 读取，作为“条件信息”。  
- 标准 FT‑Transformer 的“特征提取器”是什么：  
  - 结构：`FeatureTokenizer → TransformerEncoder → CLS pooling → Regression Head`  
  - 其中 **CLS pooling** 会把多个 token 压成 1 个向量（全局摘要）。  
  - 所以标准 FT‑Transformer 直接输出一个回归值，而不是序列。
- 为什么要考虑“替换”：  
  - **对齐执行文档**：Phase 2 要用 FT‑feature 作为基础，再替换 RealMLP。  
  - **对齐学术口径**：主线 encoder 与 FT‑Transformer 基线用同一套特征提取逻辑，减少“基线是 FT、主线却是另一套 encoder”的不可解释差异。  
  - **保持主线不变**：只替换“特征提取器”，不动 DACA 解码器、loss、指标口径，风险可控、易做 A/B。

# FT‑Encoder 落地细节：为什么当前实现能跑、旧方案会卡

- 目标约束（固定不改的主线假设）：  
  - 输入侧已经有 `FeatureTokenizer`（我们自己做 tokenization）。  
  - 主线解码器需要 **token 级上下文**（`ctx_tokens` 形状 `(B, S, D)`）。  
  - 不能引入 CLS pooling 或回归头，否则会破坏“序列输出”设计。
- 旧方案易出问题的原因（工程兼容性）：  
  - 依赖外部库的 **API/输入格式不一致** 时，容易出现 import 或形状不匹配。  
  - 一些 FT 实现会在内部**自带 tokenization 或 CLS pooling**，与我们“先 tokenizer，再交给 encoder”的主线流程冲突。  
- 现方案为什么可跑（工程上最稳的三点）：  
  1) **无外部依赖**：直接用 `nn.TransformerEncoder`，避免库版本导致的导入失败。  
  2) **输入输出对齐**：输入是 `FeatureTokenizer` 输出的 tokens，输出仍是 token 级 `ctx_tokens`，DACA 接口无改动。  
  3) **FT 常用配置对齐**：采用 **PreNorm（norm_first=True）+ GELU + FFN=4×d_model** 的常用配置，尽量贴近 FT 论文的训练稳定性经验。
- 学术术语对照（便于汇报）：  
  - **PreNorm**：LayerNorm 放在注意力/前馈之前（训练更稳定）。  
  - **GELU**：更平滑的激活函数（常见于 Transformer/FT）。  
  - **Token‑level context**：保持每列特征的 token 表示（可做细粒度注意力）。
