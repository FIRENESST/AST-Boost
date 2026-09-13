# LRGB 接入与 Peptides-struct 短程筛查（2026-09-07）

本轮把实验从 ZINC 推进到真实 LRGB Peptides 任务，并保持测试集不参与模型选择。
当前结论是：完整数据上的 35 epoch、seed 42 筛查没有支持谱核 H2，也没有支持
二阶场 H1。它已经足以阻止继续增加谱模块，但不能替代冻结计划中的 200 epoch、
五种子验证。

## 已接入的可信路径

- PyG 2.7 `LRGBDataset` 的 Peptides-func 与 Peptides-struct train/val。
- OGB 风格的 9 列原子类别与 3 列键类别嵌入，每列独立 Xavier 初始化后求和。
- 公开 GraphGPS commit `2801570` 的 `CustomGatedGCN+Transformer` 数据流：
  4 层、宽 96、4 heads、attention dropout 0.5、mean pooling、线性任务头。
- 官方 Peptides LapPE 口径：组合拉普拉斯、保留零模、10 个频率、DeepSet、16 维。
- Peptides-func 使用 BCE-with-logits 与 macro average precision；Peptides-struct 使用
  L1 与 MAE。
- AdamW 3e-4、weight decay 0、10 epoch warmup、余弦衰减、梯度范数裁剪 1。

这仍是 standalone 源码移植，不宣称运行了历史 GraphGym trainer，也不把短程分数
当作公开论文复现分数。

## 可运行性和缓存

ZINC 的整 split 常驻 GPU 缓存不适合 Peptides。新 `LRGBGraphBank` 将每张图的谱和
拓扑存成 CPU shard，只对当前 minibatch 填充并传到 GPU。本报告的历史运行先全局按
节点数分组，每轮只打乱 batch 顺序，组内成员固定；验证也按长度排序。
2026-09-10 核对发现，此训练采样偏离随机 mini-batch，可能影响 BatchNorm。
当前训练默认已改为随机置换，窗口内排序的 `sortish` 仅保留作显式消融。
下列旧数值没有使用新采样器，不应据此评价修正后的实现。

Peptides-func 与 Peptides-struct 使用完全相同的分子及 split 顺序。缓存 v2 将谱和
拓扑放到 `peptides-common`，每次载入另一任务时逐 shard 重算拓扑 SHA256 后再复用，
标签始终从对应任务重新读取。完整 10,873/2,331 train/val 缓存计算约 63 秒；另一
任务经哈希验证后复用约 12 秒。测试数据在 PyG 首次处理数据包时被自动解包，但本轮
代码没有实例化 test split，也没有训练或评估测试标签。

## 35 epoch 完整数据筛查

环境为 RTX 5070 Ti 16 GiB、PyTorch 2.13 CUDA 13.0、BF16。三种方法使用 seed 42、
相同训练顺序和 128 batch。AST 一阶编码固定为此前容量检查选择的 hidden 32、2 层；
一次额外的 hidden 64、8 层压力测试不作为默认结论。

- `signnet_local`：最佳验证 MAE **0.514281**（epoch 35），525,501 参数，
  中位训练 3.700 秒/epoch，峰值 3,486.6 MiB。
- `kern`：最佳验证 MAE **0.523943**（epoch 35），525,541 参数，
  中位训练 4.362 秒/epoch，峰值 4,632.0 MiB。
- `full`：最佳验证 MAE **0.526006**（epoch 35），530,837 参数，
  中位训练 4.850 秒/epoch，峰值 5,639.8 MiB。

配对差值为 Kern − matched first-order = **+0.009662**，Full − Kern =
**+0.002064**；MAE 越低越好。这个种子上 H2 与 H1 都没有成立。训练损失相近而
验证差异存在，说明不是某个分支完全没有参与学习；但验证曲线波动明显，仍需要
更长训练和更多种子判断差异是否稳定。

精确数值见 [`LRGB_SCREEN_20260907.json`](LRGB_SCREEN_20260907.json)。运行目录包含
逐 epoch 指标、best/last checkpoint、协议 manifest 和源码快照；目录位于 `runs/`
且不进入版本控制。

## 决策

1. 保持 `signnet_local`、Kern、Full 的数学结构和默认容量冻结，不再增加谱模块。
2. 2026-09-10 修订：先完成随机采样的同预算对照，再扩展配对种子和训练时长。
3. 官方 LapPE 与公开 SignNet 已完成整 split 单 epoch 容量检查，但尚未跑完 35/200
   epoch；论文比较必须补齐它们，不能用本地一阶对照替代公开基线。
4. Peptides-func 的旧烟雾实验 AP 未正确合并并列分数。2026-09-10 已修正指标，
   旧 AP 仅保留为工程记录，精度比较须用修正后的指标重新计算。
5. 在协议和超参数冻结前继续不评估测试集。
