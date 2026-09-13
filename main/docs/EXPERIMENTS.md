# ZINC 对照实验与 Full 保留策略

## 原则

Full 保留为正式可选模型，代码、配置、最佳 checkpoint 和运行记录都保留。
一次短跑、更高的每步耗时或个别种子不占优，都不能构成删除 Full 的证据链。
训练器没有自动淘汰模型、删除旧运行或回退到 Kern 的功能。

后续精度/吞吐联合试验见 [2026-09-04 记录](ACCURACY_SPEED_20260904.md)。
`experiments.compare` 预先固定三个 Full 配置，按种子轮换次序，保留所有结果；
`field_scaling=size` 与 `signal_backend=dense` 均为可选项，原始 none/sparse 默认值不变。
大 batch 同时调整学习率时，必须在报告中列明，不将其效果冒充为单一数学改动的收益。
共享 ψ 的等价场融合与同预算 Cosine 对照见
[数学/代码优化记录](MATH_CODE_OPTIMIZATION_20260904.md)。
B128/35 的后续吞吐、精度与反例筛查见
[继续优化记录](NEXT_OPTIMIZATION_20260904.md)。
频率标签、全谱核和核对角的同机配对实验见
[谱信息补全实验](MATH_FEATURES_20260905.md)。本轮包含等参数的无频率标签控制组。
2026-09-07 已新增 GraphGPS 兼容骨干、公开 RWSE/LapPE/SignNet 编码和预热余弦协议，
并完成首轮可信基线及 H1/H2 筛查，见
[GraphGPS 筛查报告](GRAPHGPS_SCREEN_20260907.md)。五种子强 RWSE 对照未验证谱核收益，
因此当前默认转为 `rwse_graphgps`，停止继续搜索谱模块。

LRGB Peptides 数据、指标、OGB 风格多列特征编码和公开
`CustomGatedGCN+Transformer` 现已接入。变长图使用逐图 CPU shard、minibatch 动态
填充；训练默认随机置换，`sortish` 仅作显式消融。func/struct 经逐 shard 拓扑哈希后共享谱缓存。完整
Peptides-struct train/val 的 35 epoch、seed 42 筛查见
[LRGB 报告](LRGB_SCREEN_20260907.md)：matched first-order / Kern / Full 最佳 MAE
分别为 0.514281 / 0.523943 / 0.526006。旧运行使用固定长度分组，组内成员没有逐轮变化；
它未支持 H2 或 H1，但不能替代修正采样后的比较，也不是 200 epoch 多种子结论。
测试集评估仍为 0。

2026-09-10 的[随机采样修正实验](LRGB_SAMPLING_20260910.md)保持相同的 35 epoch、
seed 42、B128 和模型容量。一阶对照 / Kern / Full 的最佳验证 MAE 分别降为
0.249446 / 0.250108 / 0.249662。主要收益来自采样修正；三种模型之间的小差值尚不支持
稳定的谱模块收益。AP 并列分数、断点一致性和数据指纹检查也已补齐。

2026-09-11 已完成冻结的[三种子复核](LRGB_REPLICATION_20260910.md)：
公开 SignNet 移植 / 一阶对照 / Kern / Full 的平均验证 MAE 为
0.254732 / 0.250739 / 0.250774 / 0.250180。
Kern 相对一阶对照平均差 +0.000035，只有 1/3 配对获胜，H2 尚未获得支持。
Full 相对 Kern 平均差 -0.000594，2/3 配对获胜；保留 Full，但不能称为稳定 H1 收益。
三组 AST 比公开 SignNet 的均值低，不能单独证明谱核有效，因为一阶对照也有同样趋势。
下一阶段优先验证训练预算：固定同一配置，从头运行更长的配对训练，再判断 MAE 与模块收益。
不能把 35 epoch 余弦已衰减结束的 checkpoint 直接续到 200 epoch，并称为原定 200 epoch 协议。
本轮不新增谱模块、不触碰测试集，也不把三个种子的标准差解释为显著性证据。

2026-09-13 补强完成态恢复检查：除协议哈希外，核对运行身份、完整 epoch 历史、
最佳验证分数及对应 epoch、发布日志和最佳 checkpoint 权重；缺失或不一致即拒绝复用，
不自动覆盖历史证据。已有 13 次完整运行通过只读检查，文件哈希均未变化。
训练公式、模型及已有 MAE 未变；runner 源码哈希已变化，历史实验仍应使用各自
保存的 runner 快照复现，新实验应生成新 manifest，不能修改旧 manifest 来绕过检查。

当前任务是建立可重复的实际训练比较，并完成短程、多种子筛查；不是宣称
已经复现论文的收敛结果，也不是一次性证明或否定二阶场假设 H1。

## 实验设计

- 官方 ZINC-subset：10,000 train / 1,000 val / 1,000 test；不混合划分。
- 输入：离散原子类别和键类别，图回归，L1 损失、验证 MAE。
- 旧 `compact` 骨干仍用于复现 2026-09-03 至 09-05 的本地试验。
  新 `--backbone graphgps` 使用公开 GraphGPS commit `2801570` 的 GINE 与
  `MultiheadAttention` 并行数据流、BatchNorm、残差、2d 前馈网络和 SAN head；
  训练循环仍为本项目 standalone runner，不冒充历史 GraphGym 运行。
- `rwse`：完整随机游走转移矩阵的第 1–20 次幂对角；不是截断谱近似。
- `lappe`：前 8 个正特征值对应的列，训练期每图、每频率随机反号。
- `signnet_local`：本项目的一阶 SignNet + 简并投影对角，关闭相对核；
  仅用作分离核偏差作用的内部对照，不能冒充公开 SignNet 基线。
- `kern` / `full`：沿用原谱模块；Full 的一阶、二阶分支都正常参与反向传播。
- PE 输出宽 16、共享 GIN 宽 32、两层；Full 仍有两个 16 维读出。
  所有模型总参数约 429k–438k，最大差约 2%，不是通过闲置参数凑预算。
- 优化器：AdamW，lr=0.001，weight_decay=1e-5，梯度范数裁剪 1，
  ReduceLROnPlateau（patience=10）；CUDA 使用 fused optimizer 和 BF16 AMP。
- 谱/核维持 FP32。试验统一用 `kernel_eps=1e-6`，以避免近常量核放大
  舍入噪声；原公共 API 的默认 `1e-8` 没有修改。这个差异在 manifest 中显式记录。
- 首轮预定种子 42–46，各方法各 5 epoch，全量训练集。取各自验证最佳 checkpoint。
  不以测试集决定 checkpoint、超参或 Full 去留；首轮 `test_mae` 为 null。

各方法的原子/键嵌入、骨干与预测头按相同种子初始化，训练样本顺序由独立
NumPy generator 决定，LapPE 的随机反号不会扰乱 shuffle。不同种子轮换方法
执行顺序，减少温度/先后顺序系统偏差。CUDA scatter 不承诺逐位确定性。

## 工程改进与测试

预计算实测（k=8、相对近简并阈值 0.01，仅统计保留谱）：训练集中 3,038/10,000
张图包含高维块，1,296 张图因完整保留边界块而保留多于 8 个频率。
合法二阶 pair 数分布为 10 个：9,838 张；6 个：118 张；3 个：44 张；
没有完全无合法 pair 的训练图。不能把这个保留谱口径当成全谱重根比例。

`GraphBank` 一次准备 split 内的填充频谱、字段、拓扑和 RWSE，常驻 GPU。
每个 minibatch 用索引选择，避免每步逐图 Python 构建和 CPU↔GPU 谱传输。
GINE、注意力及 SignNet 复用同一份节点/边布局；BatchNorm 只计算有效节点，
有效节点索引每次前向只构造一次。注意力只屏蔽无效 key，之后清零填充 query。

后续紧凑布局优化：当前默认 `--node-layout compact`，每个 batch 从 CPU 缓存的
已知节点/边位置生成索引，避免 CUDA 布尔压缩为获取输出形状而同步。局部消息、
归一化和前馈层直接操作有效节点，仅注意力填充 Q/K/V；相同的谱核掩码在层间
共享。`--node-layout padded` 保留原骨干路径，参数名称/数量不变。
这不删除任何 Full 分支，也不减少频率或合法 pair 数。它不承诺 BF16 逐位相同；
非零节点 dropout 的随机数消耗也会变化，因此新版本必须独立登记研究。
测速和旧 checkpoint 的完整验证集核对见 [本轮记录](COMPACT_BACKBONE_20260903.md)。

CLI 现在显式提供 `--k`、`--pairs`、`--rw-steps`，默认仍为 8/4/20。
`pairs` 指 k0 频率截止，不是最终 pair 数；例如 k0=4 最多有 10 个含对角项的 pair。
不同预算使用独立 cache 文件，保留原缓存。2026-09-07 起训练缓存为 v3，
在 v2 字段频率标签和可选完整核谱之外，增加官方基线专用的组合拉普拉斯谱
（包含零模）；AST 谱与公开基线谱不混用。原有 v1/v2 缓存仍保留。
主训练入口默认只加载 train/val；`--evaluate-test` 才加载和评估 test。

新 `forward_packed` 与原 padded API 的值和梯度一致性均受回归测试保护，
没有用关闭二阶场或减少合法 pair 数量来“加速” Full。

测试覆盖：完整 RWSE 的解析小图、图/标签装批对应、真实节点 BatchNorm 等价、
各方法预测的图间隔离、有限梯度、Full 二阶参数更新、相同种子骨干初始化、
按 seed 配对汇总、CUDA BF16 优化器步骤，以及标签不进入前向计算。
另有模拟中断测试：保存第一轮 checkpoint 后中断，再恢复训练；在 CPU 确定性
小样本上，恢复优化器与随机状态后的最终权重和连续训练逐位一致。
对二阶字段本身求导的测试也确认 Full 的预测确实依赖这些字段，且填充字段梯度为零。

## 运行、记录与恢复

从 `main/` 内使用 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m ast_boost.experiments.train --output runs/zinc-pilot-5x5-20260903 --epochs 5 --seeds 42 43 44 45 46
```

冻结的 GraphGPS 兼容矩阵使用：

```powershell
.\.venv\Scripts\python.exe benchmarks\experiment_graphgps_matrix.py --output runs\graphgps-zinc-frozen --epochs 2000 --warmup-epochs 50 --batch-size 32 --seeds 42 43 44 45 46
```

冻结的 LRGB 矩阵使用：

```powershell
.\.venv\Scripts\python.exe benchmarks\experiment_lrgb_matrix.py --dataset Peptides-struct --output runs\peptides-struct-frozen
.\.venv\Scripts\python.exe benchmarks\experiment_lrgb_matrix.py --dataset Peptides-func --output runs\peptides-func-frozen
```

`manifest.json` 保存参数、运行环境、源文件 SHA256 和研究范围；`dataset.json`
保存每个划分的摘要及数据指纹。每个 method/seed 目录包含：

- `metrics.jsonl`：逐 epoch 训练/验证 MAE、学习率和墙钟时间。
- `best.pt`：仅由验证集选出的模型。
- `last.pt`：模型、优化器、调度器、AMP scaler 与随机状态，支持原协议续跑。
- `result.json`：最终汇总及 Full 保留策略。

当前 checkpoint 格式 v2 将 `last.pt` 作为单一 epoch 提交点，额外保存完整的
epoch 指标历史与最佳模型权重，以及 Python/NumPy 随机状态。先原子替换
`last.pt`，再发布 `metrics.jsonl`、`best.pt`。进程在提交前中断会重跑未提交轮次，
在提交后中断则由已提交内容修复衍生文件，不重复优化器步骤。
发现冲突日志时先另存为 `metrics-uncommitted-*.jsonl`，不静默丢弃记录。
这针对进程中断恢复，不是磁盘损坏或断电持久性的保证。
四种提交/发布边界的中断测试均验证 CPU 恢复训练与连续训练逐位一致。

`summary.json` 汇总各方法的均值/样本标准差与 Full−Kern 的逐种子配对差。
负值偏向 Full。已有目录默认拒绝覆盖；中断后以相同命令加 `--resume`。
代码或训练预算变动应开新目录，不能把不等预算结果混入原试验。
首轮实际运行的源码另存为 `source_snapshot.zip`，逐文件与 manifest 校验一致；
源码快照 SHA256 为 `A9000E359CFB66A03ECBDA14F8676632A5E1442DC4C0C8D5200E7E23C1AAD3E8`。

新研究会自动生成、逐文件校验 `source_snapshot.zip`，并将归档 SHA256 写入
manifest。恢复时再次校验；已有快照不覆盖。首轮 v1 checkpoint 和全部结果原样
保留，若需在原协议中恢复旧运行，应使用其原源码快照；不能用新代码强行续入旧研究。
旧模型权重仍可用 `load_state_dict` 严格加载到两种布局做独立评估。

耗时包含 GPU bank 取样、完整模型前向、反向和 optimizer；验证耗时单独记录。
预计算、下载、checkpoint 写盘不计入训练 epoch 时间。显存为该进程的峰值
已分配显存，包含共同数据 bank，不是整块显卡占用，也不与论文异机数据直接比。

## 结果解释边界与后续证据链

5 epoch 只能判断早期学习行为和工程可运行性，不能推断收敛 MAE；BatchNorm
和不同 PE 的适应速度也会影响短程排名。不能将它与论文训练数百 epoch 的测试
分数直接比较。首轮结果另外记录在 `ZINC_PILOT_20260903.md`。

在考虑是否限制 Full 的使用前，应至少完成：

1. 公开 SignNet 和 GraphGPS+RWSE 的可信复现，排除较弱的本地对照。
2. 足够长、相近参数和相同调参预算的 Kern/Full 配对多种子比较。
3. 固定训练轮数与固定墙钟预算两种口径，检查收敛速度和最终误差。
4. 检查二阶分支梯度、pair 规模、近简并比例、核归一化敏感性和失效图分组。
5. ZINC 之外的长程任务验证；区分“该数据集无收益”与“方法普遍负面”。

## 备份和参考

修改前完整工作区源代码备份（含此前未提交改动）：
`../../backups/main-pre-experiments-20260903-204010.zip`，24 个文件。SHA256：

```text
21F82439E93F1E0ED15F9A62BDEDED668B24426DC12D57129E942CCF13C5EB65
```

原有备份未删除。恢复请解压到独立目录核对，不直接覆盖工程。

数据划分沿用 [PyG ZINC](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.ZINC.html)。
骨干设计参考 [GraphGPS 的 GINE/Transformer 层](https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py)，
但 CLI 训练器和填充优化是本项目独立实现。
