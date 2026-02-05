# Greedy vs Beam 口径选择（2026-02-02）

## 目的
明确区间评估主线口径：用 beam 替代 greedy，并给出数据依据与决策理由。

## 实验设置
- 模型：`tabseq_daca`
- 运行目录：`outputs/california_housing/run_20260202_112549`
- 温度：T=1.0（除非另标注）
- `prefix_*` 实验使用 `--max-samples 512`（子集验证）

## 真实结果（来自输出文件）

| 口径 | 温度 | PICP | MPIW | MAE | RMSE | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| teacher_forcing | 1.0 | 0.9944 | 0.3568 | 0.0711 | 0.0888 | 全量验证集 |
| greedy | 1.0 | 0.3101 | 0.3016 | 0.4818 | 0.7296 | 全量验证集 |
| prefix_full | 1.0 | 0.8906 | 1.6240 | 0.4693 | 0.7090 | 512 样本 |
| prefix_beam_k8 | 1.0 | 0.5957 | 0.8010 | 0.4927 | 0.7480 | 512 样本 |
| prefix_beam_k32 | 1.0 | 0.8770 | 1.5549 | 0.4686 | 0.7114 | 512 样本 |

> 数据来源：  
> `outputs/california_housing/run_20260202_112549/metrics_val_teacher_forcing.json`  
> `outputs/california_housing/run_20260202_112549/metrics_val_greedy.json`  
> `outputs/california_housing/run_20260202_112549/metrics_val_prefix_full.json`  
> `outputs/california_housing/run_20260202_112549/metrics_val_prefix.json`（beam_k8）  
> `outputs/california_housing/run_20260202_112549/metrics_val_prefix_beam.json`（beam_k32）

## 结论（为何替换 greedy）
1) **greedy 明显偏低**：PICP 只有 0.31，远离目标 0.90。  
2) **prefix_full 接近目标**：PICP≈0.89，说明模型“有能力给出合理区间”，问题主要出在推理口径。  
3) **beam 随 K 增大逼近 full**：k=32 已到 0.877，明显优于 greedy；这说明保留多路径可以显著缓解 exposure bias。  

## 选择策略
- **主线口径：beam（K=16/32）**  
  - 兼顾稳定性与推理成本  
  - 区间覆盖率更接近目标  
- **诊断上限：full**  
  - 仅用于小样本验证“上限”，不作为主线  
- **greedy：仅作为速度基线或部署下界**  
  - 需明确其是下界口径，不能用于区间主结论  

## 备注
- `prefix_*` 结果基于 512 样本子集，若要进入主线报告，建议再跑全量或至少更大子集确认稳定性。
