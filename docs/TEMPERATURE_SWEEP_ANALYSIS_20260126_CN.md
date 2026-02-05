# 温度扫参分析（20260126）

本文整合本次 temperature sweep 的结果，并解释为什么温度 T 上升会让 PICP
上升；同时记录“失败案例”：高温度主要是通过扩大区间来提高覆盖率，而不
是模型更准确。

## 背景：执行阶段（直白版）

- Phase 1：整理代码结构，确保数据管线稳定可复现。
- Phase 2：确认当前特征提取器的主线实现，保证训练/评估流程跑通。
- Phase 3：做区间宽度分析，找“覆盖率-宽度”的性价比点。
- Phase 4：多数据集验证，检查泛化能力。
- Phase 5：补理论（校准/Conformal），提高可信度解释。
- Phase 6：画图与报告，形成可投稿材料。

## 实验设置

- 数据集：California Housing 验证集（random_state=0）
- 模型：TransformerTabSeqModel（FT-feature 风格编码器）
- 解码：greedy（单前缀）
- 温度范围：T=0.5..10.0，步长 0.1
- 结果文件：
  - 汇总表：outputs/sweeps/temperature/temperature_Tstar_summary_20260126.csv
  - 曲线图：outputs/sweeps/temperature/temperature_sweep_d4-8_combined_20260126.png
  - 宽度分桶：outputs/sweeps/temperature/width_stratified_picp_table_20260126.csv

## 核心结果

- T=0.9 时，PICP 只有 0.21-0.29，区间很窄。
- 达到 PICP≈0.9 需要的 T* 随深度变大（约 4.9-8.8）。
- T=10 时，PICP≈0.94-0.95，但 MPIW 变得很大（约 3.3-4.1）。
- 宽度分桶显示：低 T 时大量样本落在窄区间 [0.0, 0.4)，覆盖低；高 T 时
  样本集中到宽区间 [2.0, 100.0)，覆盖高但区间过大。

depth | PICP@T=0.9 | MPIW@T=0.9 |  T*  | PICP@T* | MPIW@T*
----------------------------------------------------------
d4    | 0.2909     | 0.3843     | 4.9  | 0.9002  | 2.2853
d5    | 0.2313     | 0.2872     | 5.4  | 0.8997  | 2.1957
d6    | 0.2066     | 0.2422     | 6.7  | 0.9012  | 2.2351
d7    | 0.2510     | 0.2742     | 7.2  | 0.9021  | 2.0605
d8    | 0.2112     | 0.2555     | 8.8  | 0.8995  | 2.6690

## 为什么 PICP 会随 T 上升（对应代码机制）

1) 温度只是在 logits 上做缩放：
   - step_probs = sigmoid(logits / T)
   - T 越大，概率越靠近 0.5（不再极端自信）。

2) 叶子概率是“逐步概率连乘”：
   - leaf_probs = prod(step_probs)，再归一化。
   - 每一步的“变平”会被连乘放大，整体分布更平。

3) 区间是用 CDF 分位点取出来的：
   - 分布越平，CDF 达到 0.9 的位置越远，所以区间变宽。
   - 区间变宽 → MPIW 上升 → PICP 也上升。

4) greedy 单前缀有额外放大效应：
   - 模型只基于一个前缀输出 logits，但却把这个输出当成所有叶子桶的概率。
   - 这会让分布更尖、更敏感，导致 T 的影响被放大。

## 代码确认：模型测试最终输出与指标计算（按学长需要整理）

下面只保留学长关心的三段关键代码：  
1) 自回归前缀生成（greedy）  
2) 概率合成（leaf_probs）  
3) 区间与 PICP/MPIW 计算  

### A. 自回归前缀生成（greedy）→ 得到 step_probs

```python
def _greedy_step_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    depth: int,
    n_bins: int,
    temperature: float,
    sos_token: int = 2,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    B = x_num.shape[0]
    device = x_num.device
    dec_input = torch.zeros((B, depth), dtype=torch.long, device=device)
    dec_input[:, 0] = sos_token
    step_probs_out = torch.empty((B, depth, n_bins), dtype=torch.float32, device=device)
    start = [0 for _ in range(B)]
    end = [n_bins for _ in range(B)]

    for t in range(depth):
        logits = model({"x_num": x_num, "dec_input": dec_input})
        probs_t = torch.sigmoid(logits[:, t, :] / temperature)
        step_probs_out[:, t, :] = probs_t
        if t < depth - 1:
            bits = torch.empty((B,), dtype=torch.long, device=device)
            for b in range(B):
                s = start[b]
                e = end[b]
                mid = (s + e) // 2
                left = probs_t[b, s:mid].mean().item()
                right = probs_t[b, mid:e].mean().item()
                bit = 1 if right > left else 0
                bits[b] = bit
                if bit == 0:
                    end[b] = mid
                else:
                    start[b] = mid
            dec_input[:, t + 1] = bits

    return step_probs_out
```

### B. 概率合成（step_probs → leaf_probs）

```python
def leaf_probs_from_step_probs(self, step_probs: torch.Tensor) -> torch.Tensor:
    leaf_unnorm_probs = torch.prod(step_probs, dim=1)
    leaf_probs = leaf_unnorm_probs / (torch.sum(leaf_unnorm_probs, dim=1, keepdim=True) + 1e-9)
    return leaf_probs
```

### C. 区间与 PICP/MPIW（leaf_probs → [L,U] → PICP/MPIW）

```python
def interval_from_leaf_probs(self, leaf_probs: torch.Tensor, *, confidence: float) -> IntervalResult:
    cdf = torch.cumsum(leaf_probs, dim=1)
    alpha = 1.0 - float(confidence)
    lower_q = alpha / 2.0
    upper_q = 1.0 - (alpha / 2.0)
    lower_indices = torch.argmax((cdf >= lower_q).int(), dim=1)
    upper_indices = torch.argmax((cdf >= upper_q).int(), dim=1)
    bin_values = torch.tensor(
        [self.encoder.decode_bin_index(i) for i in range(leaf_probs.shape[1])],
        dtype=torch.float32,
        device=leaf_probs.device,
    )
    L = bin_values[lower_indices]
    U = bin_values[upper_indices]
    return IntervalResult(L=L, U=U)

covered = (y_true_np >= L_pred_np) & (y_true_np <= U_pred_np)
picp = float(np.mean(covered))
widths = U_pred_np - L_pred_np
mpiw = float(np.mean(widths))
```

## 失败案例总结（链条）

未校准 logits
  + greedy 单前缀近似
  -> step_probs 过于极端
  -> 叶子分布过尖
  -> T=0.9 时区间过窄，PICP 很低
  -> 只能靠高 T 把分布摊平
  -> PICP 提升主要来自 MPIW 暴涨（过度保守）

结论：高 T 的“提升”不是预测质量更好，而是区间被人为放宽。

## 这不是“区间累加错误”

CDF 分位点取区间是标准做法，并不存在“把区间累加导致命中率虚高”的问题。
当前现象的主要原因是概率分布过尖 + 温度拉平导致区间变宽。

## teacher_forcing 结果补充（单点对照）

说明：teacher_forcing 是“上限口径”，因为用真实前缀，不受自回归前缀误差
影响。这里给出目前 outputs 里的单点结果（非完整 sweep），用于辅助判断。

- 同一 run、同一温度 T=10（depth=6）：
  - teacher_forcing：PICP=0.9537，MPIW=3.7873，MAE=0.3992，RMSE=0.5075
  - greedy：PICP=0.9491，MPIW=3.7513，MAE=0.5482，RMSE=0.6976
  结论：TF 的点预测误差更小，但 PICP/MPIW 接近，说明“温度变大会拉宽区间”
  这个现象在 teacher_forcing 口径下也存在，不是 greedy 才会发生。
  来源：outputs/california_housing/run_d6_20260126_103646/metrics_val_teacher_forcing.json
        outputs/california_housing/run_d6_20260126_103646/metrics_val_greedy.json

- 单点 T=1.0（depth=6）：
  - teacher_forcing：PICP=0.7628，MPIW=0.2767
  结论：即便给了真实前缀，T=1.0 的覆盖率仍明显低于 0.9，说明“logit 过尖”
  是核心问题之一（不是单纯 greedy 误差）。
  来源：outputs/california_housing/run_20260124_171159/metrics_val_teacher_forcing.json

注意：上述是单点结果，若要更严格结论，仍需对 teacher_forcing 做完整 sweep。

## 关于 teacher_forcing 的目的与这次结论

原始目的：用 teacher_forcing 去掉“自回归前缀误差”，判断温度敏感性是否
主要由 greedy 前缀导致。

这次单点结果给出的结论是“部分回答”：

1) T=1.0 的 teacher_forcing 仍明显低覆盖（PICP=0.7628），说明即便没有
   前缀误差，分布依然偏尖。也就是说，温度敏感不是 greedy 单独造成的。
2) 同温度 T=10 下，teacher_forcing 与 greedy 的 PICP/MPIW 很接近，但
   greedy 的 MAE/RMSE 更差，说明 greedy 主要影响点预测误差，而不是温度
   效应本身。

因此：teacher_forcing 结果支持“logit 尺度偏大 + 连乘放大”是核心问题；
greedy 前缀问题会加重误差，但不是温度效应的唯一来源。要彻底验证，还需
做完整的 teacher_forcing 温度 sweep。

## 后续建议

1) 对比 greedy vs teacher_forcing 的温度曲线，分离前缀近似的影响。
2) 在验证集上学习温度（后验校准），避免手工调大 T。
3) 改进 per-leaf 条件概率估计，减少单前缀误差。
