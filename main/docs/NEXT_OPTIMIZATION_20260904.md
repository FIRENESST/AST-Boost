# AST-Boost 继续优化与验证（2026-09-04）

## 结论

当前验证集候选更新为 Full、`field_scaling=size`、B128、lr=0.001、Cosine、35 epoch。
Full 的一阶、二阶、核偏置和独立读出均保留，参数量仍为 437,579；没有裁掉字段、pair
或分支。三个种子的最佳验证 MAE 为 0.281574、0.282162、0.284420，平均
**0.282719 ± 0.001502**。

相对上轮 B64/20 Cosine 候选的 0.300039，新候选平均误差降低 **5.77%**；平均
训练+验证时间从 103.610 秒降至 97.770 秒，降低 **5.64%**。逐种子差值均为负，
平均 -0.017321，描述性配对 t 区间为 [-0.026872, -0.007769]。基线来自较早冻结源码，
所以这不是严格的同源码随机化对照；新源码中的裁 padding 和 BMM 归约均默认关闭。

相对最初 Full `none / B32 / Plateau / 20 epoch` 的 0.382143 与 215.913 秒，当前
联合候选的误差低 26.02%，训练+验证时间少 54.72%。该跨轮次差异包含多项配置，
不能全部归因于 batch size。

## 为什么 B128/35 同时更快、更准

B64/20 每轮 `ceil(10000/64)=157` 次更新，共 3,140 次；B128/35 每轮 79 次，
共 2,765 次，更新数反而少 11.9%，但完整数据遍历从 20 次增至 35 次。实测 B128
Full 训练步中位约 33.44 ms，B64 约 29.00 ms；每步只慢约 15%，一次处理的图却翻倍。
这使模型在略少的总时间里看到 75% 更多的数据轮次，并将 Cosine 衰减分布到更充分的
数据覆盖上。三次新运行的最佳点位于 epoch 32–34，35 epoch 没有明显浪费预算。

代价是峰值已分配显存由上轮约 340 MiB 增至 **482.6 MiB**。显存受限时，B64/20
仍是保留的低显存候选。

## 代码与数学路径筛查

CUDA 算子 profiler 显示，当前训练时间分散在大量小矩阵、BatchNorm、归约、复制和
注意力反向；没有一个单独算子可以安全地删除。已有阶段 profiler 中前向+反向合计占
91.7%，因此继续优化目标应是减少等价计算和 kernel launch，而不是牺牲 Full 字段。

本轮实现并保留了两个默认关闭的等价实验路径：

- minibatch-local padding 裁剪只删除全零尾部。B64 训练快 0.63%，但 B32 推理慢
  0.57%，收益不足且形状变化会改变有限精度累加，故 `trim_padding=False` 为默认。
- 字段掩码求和可改写成批量矩阵乘法。Float64 值和参数梯度对照通过；B64 训练快
  1.51%，但 B64 推理慢 0.42%。旧 Cosine 权重上的 BF16 预测平均差约 0.0031–0.0034，
  MAE 变化不超过 0.00034；没有从头训练证据前，`bmm_field_reduction=False` 为默认。

共享 ψ 融合也存在尺寸交叉点：B128 训练比旧分开路径快约 5.39%，B160 快 3.29%，
B192 仅快 0.93%，B224 慢 0.44%，B256 慢 4.30%。因此当前推荐 B128；不能把小批量
上的融合收益外推到任意大 batch。分开路径和 `fuse_shared_fields=False` 继续保留。

以下候选被证据否决或暂缓，不进入默认配置：

- CUDA FP32 matmul `high`/TF32：B64 训练只快约 0.33%，有限精度预测变化相对收益过大。
- `torch.compile`：当前项目内环境缺少可工作的 Triton，首次编译明确失败；未安装不确定
  的额外运行时，也未把编译写进训练器。
- best/last 权重插值：只改善 seed 43，seed 42/44 变差。
- attention dropout 0.5→0.2：seed 42 为 0.282656，差于同种子保留候选 0.281574。
- B256：单位样本吞吐更高，但更新次数下降过多，且共享场融合跨过有利尺寸区间；在完成
  收敛对照前不升级为训练候选。

## 证据与复现

聚合审计结果：`runs/next-opt-20260904/comparison.json`。GPU 配对原始记录包括：

- `batch-trimming.json`
- `field-reduction.json`
- `matmul-precision.json`
- `batch-scaling-field-fusion.json`
- `field-fusion-crossover.json`
- `cuda-operators.json`
- `runtime-options-checkpoints.json`
- `weight-interpolation.json`

复现当前候选：

```powershell
.\.venv\Scripts\python.exe -m ast_boost.experiments.train `
  --methods full --field-scaling size --batch-size 128 --lr 0.001 `
  --scheduler cosine --epochs 35 --seeds 42 43 44 `
  --output runs/my-full-b128-cosine35
```

三次候选运行均保存完整 35 轮历史、best/last checkpoint 和源码快照；两个候选 study
的源码逐文件 SHA256 完全相同，并与当前 runnable source 相符。6 个 checkpoint 均可
读取且权重有限，best.pt 与 last.pt 中原子提交的 `best_model` 逐张量一致。
所有 `test_mae=null`，本轮测试集评估次数为 0。

修改前备份：`../../backups/main-pre-next-opt-20260904-135803.zip`，59 个文件，
148,236 bytes，SHA256：

```text
BA70C72780F7A9F2273F0092DB7A1351EA27A0B0F3AB3A6D65327056BC1C8BCD
```

## 最终工程验证

- 项目内部 `.venv` 完整测试：**120 passed**。
- Ruff lint 与 format check 全部通过；`pip check` 无损坏依赖。
- `git diff --check` 无空白错误，仅报告 Windows 工作树已有的 LF→CRLF 提示。
- 候选三次运行、两份源码归档、6 个 best/last checkpoint 均通过聚合审计。
- 所有新训练与只读兼容检查均未访问测试集，也未覆盖旧 checkpoint。
