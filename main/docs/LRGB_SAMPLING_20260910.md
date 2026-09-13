# LRGB 随机采样修正实验

Peptides-struct 完整 train/val，35 epoch，种子 [42]，batch 128，bf16。测试集评估为 0。

历史参考按节点数固定分组、每轮只打乱 batch 顺序；新运行逐轮随机置换图。两组模型容量、谱设置、优化器和训练轮数相同。

## 同一模型的采样比较

- signnet_local / seed 42：最佳验证 MAE 0.514281 → 0.249446，相对下降 51.50%。
- kern / seed 42：最佳验证 MAE 0.523943 → 0.250108，相对下降 52.26%。
- full / seed 42：最佳验证 MAE 0.526006 → 0.249662，相对下降 52.54%。

## 新采样下的模型比较

- signnet_local / seed 42：MAE 0.249446，epoch 35（最后一轮 MAE 0.249446），525,501 参数，训练中位 6.37 秒/epoch，峰值分配 3018.9 MiB。
- kern / seed 42：MAE 0.250108，epoch 35（最后一轮 MAE 0.250108），525,541 参数，训练中位 9.54 秒/epoch，峰值分配 4233.1 MiB。
- full / seed 42：MAE 0.249662，epoch 35（最后一轮 MAE 0.249662），530,837 参数，训练中位 10.73 秒/epoch，峰值分配 5154.9 MiB。
- lappe_graphgps / seed 42：MAE 0.257621，epoch 33（最后一轮 MAE 0.258080），504,459 参数，训练中位 5.16 秒/epoch，峰值分配 2244.9 MiB。

配对差值（负值有利于前者）：

- kern − signnet_local / seed 42：+0.000662。
- full − kern / seed 42：-0.000446。
- kern − lappe_graphgps / seed 42：-0.007513。

## 解释边界

采样改变同时影响 batch 组成、BatchNorm 统计和优化轨迹。跨采样的 MAE 改善不能归因于谱核或二阶场，也未单独识别 BatchNorm 的贡献。H2/H1 应看新采样内部的配对模型差值，随后补多种子和更长训练。

公开 LapPE 是固定 GraphGPS 源码的独立移植，当前仍不是 GraphGym 运行时复现。本轮没有扩充谱模块；核对角仍关闭，Kern/Full 保持相同特征值标签和谱核设置。

AP 修正合并同分预测的阈值贡献；断点恢复以 last.pt 为唯一提交点，可重建 best.pt 与指标日志。新增目标和谱缓存哈希、运行脚本快照。这些工程修正未改变本次结构回归的 MAE 定义。

同一次运行的 epoch 耗时存在明显阶段性变化。报告保留中位数和范围，仅作资源记录，不把本轮单次计时当作严格吞吐基准。

精确数据、完整协议及源码哈希见 [LRGB_SAMPLING_20260910.json](LRGB_SAMPLING_20260910.json)。

## 采样分布诊断

以下为种子 42、第 1 轮的训练图节点数统计，不使用验证或测试标签。

- fixed_length：各 batch 平均节点数的标准差 84.26；batch 内节点数标准差的均值 1.44；填充后节点对数量 / 实际节点对数量 1.04。
- sortish：各 batch 平均节点数的标准差 82.21；batch 内节点数标准差的均值 14.22；填充后节点对数量 / 实际节点对数量 1.50。
- random：各 batch 平均节点数的标准差 6.59；batch 内节点数标准差的均值 83.68；填充后节点对数量 / 实际节点对数量 5.05。

填充比仅描述注意力矩阵规模，不等于整模型耗时倍数。本轮没有训练 sortish 消融，不能据此给出其 MAE 结论。

## 复现实验

使用记录对应的源码/运行脚本快照和环境，选择新的输出目录。同一运行续跑时使用原目录并加 `--resume`；源代码或数据指纹改变会被拒绝。

```powershell
.venv/Scripts/python.exe benchmarks/experiment_lrgb_matrix.py --dataset Peptides-struct --output runs/lrgb-random-reproduction --methods signnet_local kern full lappe_graphgps --seeds 42 --epochs 35 --warmup-epochs 10 --batch-size 128 --sampler random --sign-hidden 32 --sign-layers 2 --precision bf16
```

本轮仅有 seed 42；下一步先补配对种子和公开 SignNet 对照，再决定是否扩展至 200 epoch。继续冻结谱模块结构，不评估测试集。
