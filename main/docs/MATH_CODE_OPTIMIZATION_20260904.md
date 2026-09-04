# Full 数学等价代码优化与调度研究（2026-09-04）

本轮从已验证的 Full `size / batch 64 / lr=0.001` 候选继续优化。Full 的一阶场、
二阶场、谱核、层数、宽度和 437,579 个参数全部保留；公共 `none/sparse` 默认值、
历史 checkpoint、失败记录和既有运行目录均不覆盖。

修改前恢复点：
`../../backups/main-pre-math-code-opt-20260904-120727.zip`，53 个已逐项读取文件，
129,196 字节，SHA256：

```text
F270C4DC1153E256F424CCAACA848F8F2085D1F90A23973472869E7F5E2D940D
```

## 瓶颈复核

在 RTX 5070 Laptop、Full、batch 64、BF16 上，用 20 个预热后的训练步和 CUDA event
分解得到以下中位数。它用于确定优化方向，不是端到端训练成绩：

| 阶段 | ms/step | 占所列阶段 |
| --- | ---: | ---: |
| 装批 | 0.724 | 2.45% |
| 前向 | 11.566 | 39.15% |
| 反向 | 15.533 | 52.58% |
| 梯度裁剪 | 1.114 | 3.77% |
| fused AdamW step | 0.603 | 2.04% |

因此本轮不继续微调 optimizer 或缓存读取，而是减少谱场网络的重复执行。
完整采样、选图/数据/源码/脚本哈希保存在
`runs/math-code-opt-20260904/step-breakdown.json`。

## 接受：共享 ψ 的一阶/二阶场融合

Full 的两个读出头独立，但其 `psi` 是同一个共享 GIN。该 GIN 把场维当作独立批维，
不在不同场之间做归一化、注意力或聚合，所以对一阶场集合 `F1`、二阶场集合 `F2`：

```text
psi(concat(F1, F2)) = concat(psi(F1), psi(F2))
```

先拼接全部 `+/-` 场、只调用一次共享 `psi`，随后按原边界拆开并送入各自的 `rho`
读出，得到同一个数学函数。实现同时覆盖单图、packed sparse 和 dense 路径；若关闭
`share_second_order_psi`，自动回到原分开执行路径。没有合并两个 `rho`，也没有删除场。

CPU float64 上，新旧路径的输出与全部参数梯度做严格对照；符号、基旋转、节点置换和
二阶梯度测试继续覆盖默认融合路径。BF16 可能因矩阵批次形状和累加顺序产生舍入差异，
因此新的训练研究使用独立目录和源码归档，不续写旧 checkpoint。

独立整模型计时使用相同权重、图选择、BF16，7 次轮换重复，每次 5 warmup + 20 批；
训练包含装批、前向、反向、梯度裁剪和 AdamW：

| 模式 | B32 原路径 | B32 融合 | 降低 | B64 原路径 | B64 融合 | 降低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 推理 ms/batch | 11.6247 | 10.8414 | 6.74% | 11.8158 | 11.1390 | 5.73% |
| 训练 ms/batch | 29.4503 | 27.8802 | 5.33% | 32.1864 | 30.6857 | 4.66% |

原始采样保存在 `runs/math-code-opt-20260904/field-fusion.json`；表中数值由 JSON
机械读取，而不是从单个最快重复中选择。

## 保留但不默认：输入投影与首层聚合交换顺序

对标量场 `x`、投影 `h=xW+b` 和首层 GIN，有严格恒等式：

```text
(1+eps) h_i + sum_j h_j
= ((1+eps) x_i + sum_j x_j) W + ((1+eps) + degree_i) b
```

因此可以先聚合标量再投影，避免在第一层按边搬运 `hidden_dim=32` 个值。float64 的
新旧值和梯度对照通过，重复边由 `degree` 的重数保留，dense 使用邻接行和。

但整模型实测显示，相对“仅场融合”，它的延迟反而增加约 0.3%–2.2%；B32/B64 训练
峰值显存从 247.56/316.79 MiB 降到 241.84/304.44 MiB。故默认保持关闭，仅把实现与
测试保留为显存受限备选，不能称作速度优化。完整证据为
`runs/math-code-opt-20260904/field-linear-fusion.json`。

## 暂不实现：全数据集 Bernstein 核基缓存

谱核对 Bernstein 系数线性，理论上可预存每个 basis 的 `U diag(B_d(lambda)) U.T`。
但当前 ZINC 训练缓存为 10,000 图、最大 37 节点、degree 8；float32 全缓存约需额外
470 MiB，且 profiler 没有显示谱核足以支配整步。没有先分离测出净收益前，不用显存
换取一个推测中的加速，也不降低精度来保存成 BF16。

## 精度研究：同预算 Plateau 对 Cosine

预先固定 `runs/math-code-opt-20260904/schedule20/plan.json`：完整 Full、场大小调节、
batch 64、lr=0.001、sparse、20 epoch，seeds 42/43/44；唯一变量是原
`ReduceLROnPlateau(patience=10)` 与 `CosineAnnealingLR`。每个种子轮换执行顺序，
checkpoint 仅按验证 MAE 选择，测试集不评估。

Cosine 的动机是保证所有种子在固定短预算内退火，而不是依赖波动验证曲线连续 11 次
不改善；此前 safe30 中只有 seed 44 触发 Plateau 降率，并在之后明显改善。
这是经验假设，不是精度保证。CUDA scatter 非逐位确定，样本仅 3 个种子；最终须同时
报告所有运行、配对差值、区间和训练+验证时间。

全部 6 次运行已完成：

| Full 调度器 | 验证最佳 MAE（3 seeds） | 训练均值 (s) | 训练+验证均值 (s) |
| --- | ---: | ---: | ---: |
| Plateau 对照 | 0.362255 ± 0.010203 | 100.164 | 103.878 |
| Cosine | **0.300039 ± 0.003266** | 99.898 | 103.610 |

| Seed | Plateau | Cosine | Cosine−Plateau |
| --- | ---: | ---: | ---: |
| 42 | 0.364900 | 0.303294 | -0.061606 |
| 43 | 0.370877 | 0.296761 | -0.074115 |
| 44 | 0.350990 | 0.300062 | -0.050928 |

Cosine 在相同 20 epoch、样本、更新次数与架构下，把平均 MAE 降低 17.17%，
训练+验证时间反而小幅少 0.26%。配对差值均值 -0.062216，描述性 95% t 区间
[-0.091046, -0.033387]；n=3、近似正态、未做多重比较修正，不能扩展成跨数据集定理。
三个 Cosine 运行在 60.85–63.90 秒达到 MAE≤0.35；三个 Plateau 对照在完整 20 轮内
均未达到，不外推其所需时间。

相对上轮原始 `none / B32 / Plateau / 20 epoch`，最终联合候选的平均验证 MAE
0.382143→0.300039（误差降低 21.49%），训练+验证计算 215.913→103.610 秒
（减少 52.01%）。这项跨轮次研究的联合差异还包含大小调节、batch、代码执行顺序及
调度器；精度归因以本轮同源码 Plateau/Cosine 配对为准，速度归因以独立代码基准为准。

Cosine 作为显式候选而非静默默认：训练器默认仍是原 Plateau，复现时必须写
`--scheduler cosine`。所有 checkpoint 的 `test_mae` 均为 null；没有用测试集选择配置。

## 旧权重的 BF16 兼容检查

把上轮三个 `safe30` 最佳 checkpoint 逐个加载，在相同验证集上分别运行旧的分开场路径
与新的共享 ψ 融合路径。平均预测绝对差为 0.00312–0.00339，单点最大差 0.03125–0.0625；
验证 MAE 的绝对变化为 0.000033–0.000100。该差异来自 BF16 GEMM 形状/累加顺序；
float64 值和梯度严格对照通过。检查只读原权重，未覆盖 checkpoint，也未访问测试集。
记录位于 `runs/math-code-opt-20260904/field-fusion-checkpoints.json`。

## 复现

从 `main/` 使用项目内部环境：

```powershell
.\.venv\Scripts\python.exe -m ast_boost.experiments.train `
  --methods full --field-scaling size --batch-size 64 --lr 0.001 `
  --scheduler cosine --epochs 20 --seeds 42 43 44 `
  --output runs/my-full-cosine20
```

若要重做调度器配对而不是只运行候选：

```powershell
.\.venv\Scripts\python.exe -m ast_boost.experiments.schedule_compare `
  --output runs/my-schedule20 --epochs 20 --seeds 42 43 44
```

`schedule_compare` 在父计划后和每次运行后验证全部 runnable source SHA256；未来若源码
中途变化将拒绝继续。异常会另存类型、消息、最后提交 epoch 与 last.pt 哈希，不改写
checkpoint；不完整重复不进入汇总。训练指标现在分别记录本轮实际 `lr` 与 `next_lr`，
避免把调度后、下一轮才使用的学习率误标成本轮学习率。

## 最终验证

- 项目内部 `.venv`：后续新增测试前为 111 tests passed；最新总数见
  [继续优化记录](NEXT_OPTIMIZATION_20260904.md)。Ruff、format check、`pip check` 通过。
- 新调度研究：6/6 次完整运行、每次 20 epoch；12 个 best/last checkpoint 均可读取，
  权重有限，best.pt 与 last.pt 原子提交中的 `best_model` 逐张量完全一致。
- 6 份运行源码归档逐文件 SHA256 与预先冻结的父计划一致；分析文件的 plan SHA256
  与磁盘计划一致；所有 `test_mae=null`，测试集评估次数为 0。
- `git diff --check` 无空白错误（仅 Windows 工作树已有的 LF→CRLF 提示）。
