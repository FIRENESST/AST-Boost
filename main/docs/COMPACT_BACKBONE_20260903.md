# 紧凑骨干与恢复可靠性优化（2026-09-03）

## 结论与边界

本轮不改变 Full 的理论结构：一阶编码、二阶乘积场 GIN、可学习相对谱核全部保留，
没有减少合法 pair、隐藏宽度、骨干层数或参数。修改的是执行布局与实验记录机制。
两轮独立交替测速中，Full 的完整训练步耗时分别降低约 15.4% / 14.9%。
这是相对本项目上一版的工程收益，**不是相对 RWSE/公开模型的速度优势，也不是精度提升**。

## 接受的改进

1. GINE、BatchNorm、前馈层直接处理有效节点，避免每层多次取出/写回填充位置；
   只有注意力需要填充 Q/K/V。参数名称和数量不变，旧模型权重可严格加载。
2. 同一谱核的 key-padding 掩码构造一次，在所有骨干层共享，梯度仍累加到同一谱核。
3. GraphBank 利用 CPU 已知的节点/边位置生成索引，去掉 CUDA 布尔压缩为获取动态
   输出形状产生的同步。仍有索引上传等工作，并非声称完全没有 CPU/GPU 同步。
4. 原骨干保留为 `--node-layout padded`；默认新布局为 `compact`。
   GraphBank 也保留 `static_layout=False` 的布尔压缩参考入口。
5. `--k`、`--pairs`、`--rw-steps` 暴露为显式实验参数，默认仍是 8/4/20。
   `pairs` 是 k0 截止而非字段数量；不修改预计算数学或原缓存格式。
6. 自动保存源码快照；epoch 的模型、最佳权重和指标历史在同一个原子 checkpoint
   中提交，再发布日志和 best.pt，避免中断导致指标重复或最佳权重与轮次不一致。

数学上，去掉无效节点上的计算不改变有效节点的消息或 BatchNorm 统计；无效 key
被屏蔽，无效 query 的结果不参与后续计算。该等价性不依赖关闭 Full 的任何分支。
FP32/BF16 的矩阵形状变化会改变舍入；节点 dropout 非零时随机数消耗也不同，
所以不能承诺新旧训练轨迹逐位相同，更不能把新旧运行混入同一研究续跑。

## 完整模型测速

设备：RTX 5070 Laptop；PyTorch 2.13.0+cu130；4 CPU 线程；BF16 AMP。
共同模型：10 层、宽 64、4 头，PE 宽 16、SignGIN 宽 32，batch=32。
从真实 ZINC 训练集按固定随机序列选择 30 个 batch，填充上限 37。
每路径预热 5 步，5 轮交替执行，每轮重新加载相同初始权重。
参考模型直接从本轮修改前的可信源码备份读取，不只是给新代码贴上“旧版”标签。

| 方法 | 旧训练步 ms | 新训练步 ms | 耗时降低 | 旧推理 ms | 新推理 ms | 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RWSE | 30.959 | 25.243 | 18.5% | 11.766 | 9.528 | 19.0% |
| AST-Kern | 36.094 | 30.911 | 14.4% | 14.323 | 11.688 | 18.4% |
| AST-Full | 37.972 | 32.316 | 14.9% | 15.044 | 12.635 | 16.0% |

训练步包括 GPU bank 装批、完整模型前向、反向、梯度裁剪和 fused AdamW；
推理包括装批和完整模型前向。二者均不含预计算、验证循环或 checkpoint 写盘。
Full 的训练轮次范围为旧 37.884–38.229 ms、新 32.003–33.579 ms。
训练峰值已分配显存约从 252.1 降至 250.3 MiB，主要收益是时间而非显存；
该数字包含训练 split 的共享 GPU bank，不是设备总占用。
新布局同时加速基线，Full 本身仍比同布局 Kern/RWSE 慢。

首轮探索测速（每轮 20 batch）也保留在 `timing.json`，正式复测（30 batch）在
`timing-verified.json`；两轮没有挑选单个最快值，均取 5 轮中位数。
不要把本表与上轮编码模块约 22% 的布局入口收益相加。

## 已训练模型的完整验证集核对

使用首轮试验已有的 RWSE/Kern/Full × seeds 42–46，共 **15 个 checkpoint**。
每个 checkpoint 在完整的 1,000 张验证图上分别运行旧/新布局和 FP32/BF16，
不重新训练、不改变权重，不评估测试集。所有模型均严格加载成功。

| 模型 | FP32 最大逐图预测差 | FP32 最大验证 MAE 差 | BF16 最大验证 MAE 差 |
| --- | ---: | ---: | ---: |
| RWSE | 9.54e−7 | 0 | 0.0002472 |
| AST-Kern | 9.54e−7 | 2.98e−8 | 0.0003015 |
| AST-Full | 1.19e−6 | 1.19e−7 | 0.0002259 |

BF16 并非逐图相同：RWSE 最大逐图差为 0.0625，Kern/Full 为 0.03125。
完整验证 MAE 的变化较小，但这个核对不能证明新训练轨迹或最终收敛误差相同。
各 checkpoint SHA256、全部预测/标签数组、30 条方法/种子/精度汇总均单独保留。
测速脚本额外检查未训练模型的相对数值误差；未校准 BatchNorm 的输出尺度很大，
因此精度判断以此处的已训练模型核对和高精度梯度测试为依据。

## 可靠性与回归测试

修改前 55 项测试通过；本轮增加到 **82 项**，全部通过。
新增覆盖包含重复/乱序图选择、无边图、真实节点计数校验、7 种模型在训练/评估
模式下的高精度输出/参数梯度/BatchNorm 状态等价、CUDA FP32/BF16、Full 二阶
梯度以及非默认 k/k0/RWSE 预算。原谱符号、简并、排列不变性测试保持通过。

四种模拟中断位置：提交 last.pt 前、提交后、发布 best.pt 前、发布后。
恢复后的 CPU 训练最终权重、最佳权重、随机状态和逐轮学习指标均与连续训练
逐位一致，日志不重复。冲突日志会先另存备份；源码快照内容变化会拒绝恢复。
这是进程中断恢复验证，不是断电、磁盘损坏或多进程同时写同目录的保证。

全量训练集短程运行另存于 `../runs/compact-smoke-20260903/`：RWSE/Kern/Full，
seeds 42/43，各 3 epoch。**6 组全部完成、18 条 epoch 记录、12 个 best/last
checkpoint**；未发生非有限梯度或 OOM，`test_mae` 全部为 null。
训练秒/epoch 中位数约为 RWSE 7.74、Kern 9.36、Full 10.03；这些不是与旧实验
跨时段比较的加速证据，加速以同进程交替表格为准。
其用途是集成冒烟检查，不作为方法排名或 Full 去留证据。
实际执行原命令加 `--resume` 后，28 个已完成实验文件的内容 SHA256 全部未变，
没有重复训练或添加 epoch；Ruff、格式检查与依赖一致性检查均通过。

## 复现与备份

从 `main/` 使用项目内部环境运行，输出路径必须是尚不存在的新路径：

```powershell
.\.venv\Scripts\python.exe benchmarks/benchmark_backbone.py --reference-archive ../backups/main-pre-packed-backbone-20260903-221136.zip --iterations 30 --output runs/my-compact-timing.json
.\.venv\Scripts\python.exe benchmarks/check_checkpoint_layout.py --study runs/zinc-pilot-5x5-20260903 --reference-archive ../backups/main-pre-packed-backbone-20260903-221136.zip --output runs/my-checkpoint-layout
.\.venv\Scripts\python.exe -m ast_boost.experiments.train --methods rwse kern full --seeds 42 43 --epochs 3 --output runs/my-compact-smoke
```

参考归档会执行其中的 Python 模型源码，只允许使用自己信任的本地源码备份。
原始测速及逐图核对记录位于 `../runs/compact-backbone-20260903/`。
当前新研究会自动产生 `source_snapshot.zip`，恢复时验证源码指纹与归档 SHA256。
本轮短程运行源码快照包含 14 个文件，SHA256 为
`676fd41be8775010eed12d75cfe27ca39dd62e2e4ab59077964f74ea4580a32c`。
正式测速、已训练模型核对、短程运行三个记录的源码指纹均与当前源文件一致。
旧试验、原 Full 配置、50 个既有 checkpoint 及以前的备份均未删除或改写。

本轮修改前备份：`../../backups/main-pre-packed-backbone-20260903-221136.zip`，
33 个可读源码/文档文件，100,286 字节；不包含 `.venv`、数据或运行权重。
SHA256：

```text
9463F6E84EF33C21E2501B8CF5760E3AEB61B1A270D69949B44B2C9B2901E2AE
```

若恢复旧协议，请先解压到独立目录检查，勿覆盖当前工程。下一步的效果证据仍需
长预算、多种子配对训练和可信的公开基线复现；本轮没有产生删除 Full 的证据。
