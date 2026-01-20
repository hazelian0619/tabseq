# TabSeq / Trace Regression (WIP)

这份仓库的目标是把“表格回归(连续 y)”改写成“预测一串二分决策序列(0/1)”，从而同时得到：

- 点预测 `y_hat`（用于 MAE / RMSE）
- 预测区间 `[L, U]`（用于覆盖率/区间宽度等不确定性指标）

当前仓库以两个 notebook 为主：
- `tabseq_trace_design.ipynb`：标签编码/数据组织/评估指标的核心骨架（方法侧）
- `quantile_regression_extended_benchmark.ipynb`：区间评估口径与简化 benchmark（基线侧）

要把它提升到“工业级可复现工程”，请从执行文档开始：
- `docs/EXECUTION.md`

