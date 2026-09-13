# GraphGPS 兼容骨干与核心假设筛查（2026-09-07）

## 结论

本轮停止增加谱模块，完成了可信骨干接入和两项核心假设的首轮筛查。35-epoch
ZINC 结果表明：公开 RWSE 是当前最强基线；AST 的主要短板在一阶字段编码；谱核在
强 RWSE 上的五种子平均收益未成立；Full 相对 Kern 只有单种子约 0.00175 MAE 的
小幅优势，尚不足以支持 H1。

因此当前推荐默认是 `rwse_graphgps`。核和 Full 代码继续保留用于更长训练及 LRGB，
但不作为已经验证的默认改进。

## 接入范围

实现对齐公开 GraphGPS 仓库 commit
`28015707cbab7f8ad72bed0ee872d068ea59c94b`：

- `GINE + Transformer` 并行分支、各自残差与 BatchNorm、求和融合、2 倍宽前馈层；
- ZINC 的 TypeDict 原子/键嵌入、add pooling 和 SAN 两层缩宽回归头；
- 公开 RWSE、随机反号 LapPE 和 SignNet-MLP 编码结构；
- AdamW 及线性预热后的半余弦调度。

公开 LapPE/SignNet 的缓存使用组合拉普拉斯最小 8 个模态并包含零模，与官方 ZINC
配置一致。AST 继续使用其定义所需的对称归一化拉普拉斯、正谱和简并安全块处理；
两套统计分别缓存，避免基线继承 AST 的预处理选择。

这是在当前 PyTorch 2.13 / PyG 2.7 上对官方层和编码器数据流的可审计移植，训练循环
仍是本项目的 standalone runner，不宣称运行了旧 GraphGym 环境或复现论文分数。

## 筛查协议

- ZINC subset：10,000 train / 1,000 validation；测试集未加载、未评估。
- 10 层、宽 64、4 头；BF16；batch 128；35 epoch；前 5 epoch 预热。
- 选择各方法最佳验证 MAE。该预算用于筛查，不等于官方 batch 32 / 2000 epoch 协议。
- Kern/Full 使用相同频率标签、低频核、训练预算和种子；核对角关闭。

## 单种子最小矩阵

| 方法 | 参数量 | seed 42 最佳验证 MAE |
| --- | ---: | ---: |
| GraphGPS + RWSE | 423,717 | **0.17968** |
| 公开 GraphGPS SignNet | 486,957 | 0.23634 |
| AST-Full | 441,931 | 0.27915 |
| AST-Kern | 437,147 | 0.28090 |
| AST 一阶控制（关闭核） | 437,107 | 0.28155 |
| 随机反号 LapPE | 423,833 | 0.28616 |

Kern 相对同构的一阶控制只改善 0.00065；Full 相对 Kern 改善 0.00175。Kern 相对
公开 SignNet 差 0.04456，相对 RWSE 差 0.10122。说明当前 AST 整体差距主要不能由
“再加强核或二阶场”解决。

将 AST 的字段 GIN 从 2 层/32 宽提高到 8 层/64 宽后，Kern 参数增至 509,441，
MAE 反而从 0.28090 变为 0.28298；这条纯容量路线不保留为默认。

## H2：在强 RWSE 上隔离谱核

只在 GraphGPS+RWSE 上增加已有谱核注意力偏置，公共权重的同种子初始化逐位相同，
参数从 423,717 增至 423,757。负差值表示核更好。

| seed | RWSE | RWSE + kernel | 差值 |
| ---: | ---: | ---: | ---: |
| 42 | 0.17745 | 0.17617 | -0.00128 |
| 43 | 0.18645 | 0.18660 | +0.00015 |
| 44 | 0.18224 | 0.18062 | -0.00162 |
| 45 | 0.18552 | 0.18487 | -0.00065 |
| 46 | 0.17216 | 0.18034 | +0.00818 |
| 均值 | **0.18076** | 0.18172 | +0.00095 |

核在 3/5 个种子上略好，但 seed 46 的退化使均值变差。配对差的描述性 95% t 区间
为 `[-0.00413, +0.00604]`，明显跨零。当前应记录为 H2 未验证，而不是继续调核形状。

## 冻结后的执行顺序

1. 默认强基线使用 `rwse_graphgps`；`rwse_kernel_graphgps` 只保留为 H2 对照。
2. ZINC 后续只做预先登记的更长训练，不继续搜索谱结构；测试集继续不碰。
3. H1 需要 Kern/Full 多种子长训练；目前单种子差值不能形成结论。
4. 下一项实现工作转向 Peptides-func / Peptides-struct 数据与指标接入，再在相同骨干上
   运行 RWSE、公开 SignNet、Kern、Full。

可执行入口：

```powershell
.\.venv\Scripts\python.exe benchmarks\experiment_graphgps_matrix.py `
  --output runs\graphgps-zinc-frozen `
  --epochs 2000 --warmup-epochs 50 --batch-size 32 `
  --seeds 42 43 44 45 46
```

完整数值见 [机器可读结果](GRAPHGPS_SCREEN_20260907.json)。运行目录中的 manifest、
源码快照、逐 epoch 指标和 checkpoint 均保留；本报告没有使用测试集结果。
