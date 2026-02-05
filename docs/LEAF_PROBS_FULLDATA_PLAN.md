# LEAF_PROBS 全量替换方案（统一口径）

## 背景
目前 6.1 的 y_range 表来自 N=200 子集，而 4.3 的 width_stratified 来自完整验证集。两者口径不同，容易误解。

## 目标
把“统计类表格和汇总图”统一改成全量验证集，避免口径混用。

## 执行时机（避免反复重跑）
建议等 `alpha` 固定后再做全量替换：
- 现阶段：继续用 N=200 做“形态诊断”（确认波峰合并逻辑是否合理）。
- 决策阶段：确定 `alpha` 后，一次性生成全量表格与图，作为主线结果。

## 术语（输入/输出/例子）
- 全量验证集  
  - 输入：验证集全部样本  
  - 输出：统计表和汇总图  
  - 例子：width_stratified 里 n=4128
- N=200 子集  
  - 输入：固定随机抽样 200 条  
  - 输出：小样本可视化  
  - 例子：y_range 表里 y_true_n 总和为 200

## 替换范围
- 6.1 的 y_range 表格（改用全量）  
- leaf_probs_summary_all.png（改用全量）  
- leaf_probs_summary_ranges.md（改用全量）  

## 不动内容
- 逐样本可视化（如果以后需要，可保留 N=200 作为“示例”）

## 具体步骤（命令）
1) 导出 greedy 全量 leaf_probs（示例：mask_outside=1.0/0.5/0.3/0.1）
```
PYTHONPATH=src python3 scripts/eval.py \
  --ckpt outputs/<run>/checkpoint.pt \
  --mode greedy \
  --mask-outside 1.0 \
  --export-leaf-probs \
  --export-samples 0
```
按需要把 mask_outside 改成 0.5 / 0.3 / 0.1，重复导出。

2) 导出 beam/full 全量 leaf_probs
```
PYTHONPATH=src python3 scripts/eval_prefix_search.py \
  --ckpt outputs/<run>/checkpoint.pt \
  --mode both \
  --beam-size 16 \
  --export-leaf-probs \
  --export-samples 0 \
  --max-samples 0
```
如需 K=32，改 `--beam-size 32` 再跑一次。

3) 生成全量 y_range 表
```
PYTHONPATH=src python3 scripts/summarize_leaf_probs_ranges.py \
  --inputs outputs/<run>/leaf_probs_val_*.npz \
  > outputs/analysis/leaf_probs_summary_ranges_full.md
```

4) 生成全量汇总图
```
PYTHONPATH=src python3 scripts/plot_leaf_probs_summary.py \
  --inputs outputs/<run>/leaf_probs_val_*.npz \
  --labels greedy_mask1 greedy_mask0.5 greedy_mask0.3 greedy_mask0.1 beam16 beam32 full \
  --show-y-hist \
  --out outputs/analysis/leaf_probs_summary_all_full.png
```

5) 更新文档引用
- 把 6.1 的数据源改成 `outputs/analysis/leaf_probs_summary_ranges_full.md`  
- 把图改成 `outputs/analysis/leaf_probs_summary_all_full.png`

## 验收标准
- 6.1 表格的 y_true_n 总和等于验证集样本数  
- 图和表都来自同一批“全量” npz  
- 文档不再出现 N=200 口径
