# 理论—实现审计（2026-09-03）

后续已新增本地 GPS-style 训练入口和短程多种子结果，见 [实验协议](EXPERIMENTS.md)
与 [首轮实测](ZINC_PILOT_20260903.md)。下文的实现范围表记录的是训练接入前的状态；
目前仍未宣称完成官方 GraphGPS/SignNet 复现或收敛精度验证。
后续 [幅度条件化与吞吐试验](ACCURACY_SPEED_20260904.md) 只调整绝对输入场的
图级幅度；原始 U、简并处理和相对核不变，没有新增严格表达力/扰动稳定性结论。

本轮以根目录 `READMEv0.3.md` 为输入规格；原规格书保持不变。
下面区分已经验证的实现、需要收窄的理论表述，以及尚未完成的实验。
不将模块单测或合成图 GPU 吞吐率当作 ZINC/LRGB 的精度证据。

## 已修复的实现偏差

| 约束 | 本轮处理 | 回归证据 |
| --- | --- | --- |
| 一个图的编码不能由同批其他图是否存在合法字段决定 | `readout_padded` 在每图字段全空时返回零，并阻断虚假的 `rho(0)` 梯度 | 空谱、只有简并块、正常 singleton 图混合批次与单图一致 |
| 求解器分支不能改变图 | CSR 与密集路径均对重复有向边取最大权重，再无向对称化；密集构图改为 `maximum.at` | sym/comb/rw 三种拉普拉斯一致 |
| 必须取最低频，不是任意内部频率 | 默认 shift 改为 `-1e-5`；最低谱 API 拒绝非负 shift | 低频和正 shift 附近密集谱共存的对角 PSD 反例 |
| 每图、每头按非对角元计算总体标准差 | 先屏蔽对角/填充，减去公共参考值，再计算中心化二阶矩 | 大对角、大公共偏移、小真实方差，与 float64 NumPy 参考一致；double gradcheck |
| AMP 不应先量化冻结谱再恢复 FP32 | 冻结谱及谱核乘法保留至少 FP32，神经 PE 仍用 AMP | 半精度输入 token、简并旋转、CUDA 前后向 |
| 填充 query 无合法 key 时不应生成 NaN | 新增 `attention_softmax`，softmax 之前处理全 `-inf` 行，之后返回零概率 | 输出和反向梯度有限，空行梯度为零 |
| 缓存不能因设备别名隐式往返 CPU | 统一 `cuda` 与当前 `cuda:N`，保留 `.to()` 对象身份 | 已在 GPU 的频谱装批不经过 CPU |

单图、旧 disjoint batch 和向量化 padded batch 使用同一套非对角标准化逻辑。
`contiguous=True` 要求按 graph ID 升序连续排列；CPU/CUDA 均检查这个契约。
一般交错排列继续用默认的稳定排序路径。整批为空受支持，但非空批次中的
零节点图被明确拒绝，因为 PyG `batch` 向量无法表示该图的位置。

### 为什么负 shift 更合适

SciPy 的 normal shift-invert 将特征值映射为 `1 / (lambda - sigma)`。
当 `lambda >= 0`、`sigma < 0` 时，它随 `lambda` 严格递减，`which='LM'`
因此优先选择最低谱，而且不在精确零根处求逆。正 shift 落在谱内部时只保证
选择距离该 shift 最近的根，可能漏掉更低的根。这是对 v0.3 正 shift 建议的
收窄，不是在原拉普拉斯上加随机扰动。[SciPy 官方 eigsh 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)

CLI 和三个示例 YAML 已同步。旧的显式 `--sparse-sigma 1e-5` 应改成
`--sparse-sigma=-1e-5`。示例缓存目录改为 `spectral_k8_v2`，已有缓存没有删除；
受重复边或正 shift 影响的谱需要重新预计算。

## 原规格需要限定的数学表述

### 1. 一般四阶矩不是符号不变量

令 `M_pqrs = sum_i u_pi u_qi u_ri u_si`。在独立列反号下：

```text
M_pqrs -> s_p s_q s_r s_s M_pqrs
```

只有每个频率下标的出现次数都是偶数时，才能普遍保证该符号因子为一。
例如 `<w_00, w_01>` 在只反转第 1 列时变号，回归测试包含这个反例。
`||w_pq||^2 = sum_i u_pi^2 u_qi^2` 确实符号不变，但能由逐节点 `|U|`
再做全图求和得到，不能独立证明 H1 的表达力优势。

H1 可保留的动机是：图上的乘积场消息传递能接触跨节点相关性。
例如 `psi(w)=(w+Aw)^2` 的对称化包含跨节点乘积项，而逐点函数不能访问邻居。
这不是“Full 严格强于所有 SignNet”的证明，更不是任务精度结论。
一般近简并块内，满足反号不变也不等于满足 O(m) 基不变。

### 2. 投影矩阵的普通幂没有新增信息

正交投影满足 `P_B^2=P_B`，故所有正整数次幂的对角仍是 `diag(P_B)`。
不能把“投影的若干次幂对角”说成更强的特征。不同频率权重的热核统计是
另一件事；它不是单个投影矩阵的普通幂。

### 3. 跨块 pair 也可能不安全

当 `p` 位于一个高维块、`q` 是其他块的 singleton 时，`u_p * u_q`
仍随 O(m) 旋转混合。当前实现正确地排除所有触及非 singleton 块的 pair，
而不只是“两个端点同属一个简并块”的 pair。块内信息保留在投影对角与相对核中。

### 4. 截断核、归一化与稳定性边界

- `U diag(g) U.T` 的原始核在块内响应为常数时基不变；近简并时这是主动采用
  块均值后的近似模型，并非不等特征值原算子的精确旋转对称。
- 硬分块阈值、截断边界仍可能对谱扰动不连续；不能据此声称具有 SPE 式稳定性保证。
- 保留低频且删除零空间的热核对角不等于完整热核/RWSE；缺少被截断的频率和零模贡献。
- 标准化并清零对角后的 bias 不再保证 PSD，即使原始响应全为正。
- 标准差接近零时，任何单位方差归一化都敏感。新的中心化计算消除了公式相消，
  但不能消除输入谱本身的舍入误差。`kernel_eps` 可配置，兼容默认仍为 `1e-8`；
  高对称、近常量核应额外检查 epsilon 消融或用 float64 诊断，不能把舍入噪声当结构。
- 当前 `I-D^(-1/2) A D^(-1/2)` 实现对孤立点保留对角 1，因此孤立点的谱值为 1；
  “零根个数等于全部连通分量数”在这个约定下不包括孤立点。原有约定未静默更改。

## 已实现范围与仍然存在的缺口

| 模块 | 当前实际实现 | 不能据此声称 |
| --- | --- | --- |
| Lite | 每节点谱范数和特征值加权平方和这两个摘要的 MLP | 等价保留全部 `|x_i|` |
| 一阶 PE | singleton 用带符号列，高维块用投影对角；共享 GIN 和求和 readout | 完整 BasisNet；带频率标签的完整公开 SignNet 基线 |
| 二阶 PE | 安全 singleton 的低频乘积场与 SignNet，空集合返回零 | H1 已验证、匹配预算下优于 Kern |
| 谱核 | 每头 Bernstein 响应、块均值、非对角标准化 | Chebyshev/MLP 响应已实现；标准化后仍是 PSD 核 |
| 集成 | 返回 token、bias、mask 的可微适配器及安全 softmax 工具 | 完整 GraphGPS 训练器、ZINC/LRGB 复现实验已完成 |

示例 YAML 是配置草案，不是已连接的 GraphGym 配置加载器。原来宣称开启的
`include_eigenvalue_embedding` 改为 false，Lite 输入字段也改为实际摘要名称。
Dropout、边特征骨干、频率标签、RWSE/SignNet 公平基线等仍需在真正训练集成中落实。
组合拉普拉斯的谱可以超出 `[0,2]`；新增 `kernel_domain_max` 供该消融设置合适的
统一谱域，不能直接沿用归一化拉普拉斯的 2 并误以为高频仍可区分。

## 后续更优解的选择依据

1. 优先补 GraphGPS 基线和频率标签消融，而不是继续增加不经验证的网络宽度。
2. 变长图按节点数/字段数分桶，避免单个大图让整个 batch 过度填充；先测真实数据分布。
3. 保持低秩 `U` 缓存为默认。缓存全部投影或 Bernstein 核基会把谱存储从
   `O(B*N*k)` 提升到 `O(B*k*N^2)` 或 `O(B*(degree+1)*N^2)`；默认 k=8、degree=8
   时不自动带来更少乘加，需按复用次数和显存预算验证后才采用。
4. 对已经不变的投影对角/对角 pair，去掉负号分支可能是合理消融，但改变了现有
   `psi(w)+psi(-w)` 函数及 checkpoint 行为，不应当作完全等价的提速静默替换。

## 实机记录与回退

RTX 5070 Laptop / PyTorch 2.13.0+cu130 / seed 42。多轮交替顺序，报告中位数；
合成路径图，非端到端数据训练。旧路径和新路径均为本轮代码，不把倍率冒充本轮相对
上一版向量化代码的提升。

| 条件 | disjoint 参考 | vectorized padded | 峰值显存：参考 / padded |
| --- | --- | --- | --- |
| 32 图 x 64 节点，推理，5 轮 x 20 次，复用装批缓存 | 84.341 ms | 3.811 ms | 101.3 / 58.9 MiB |
| 同规模，3 轮 x 10 次，包含每步 GPU 频谱装批 | 84.375 ms | 9.427 ms | 101.3 / 59.1 MiB |
| 16 图 x 32 节点，前向+反向，3 轮 x 10 次，无 optimizer | 91.695 ms | 7.526 ms | 113.2 / 98.8 MiB |

本轮修改前已在较空闲 GPU 上测得旧向量化路径约 4.019 ms（32 x 64）。本轮目标
主要是理论正确性和稳定性，不依据单次旧测量宣称显著版本间提速。
装批计时仍不包括磁盘 IO、DataLoader 和骨干；`--backward` 不包含 optimizer/GradScaler。

### 验证与复现

在 `main/.venv` 内完成：36 项测试通过（含 20 项新增理论/数值回归及真实 CUDA
检查），Ruff 检查与格式检查通过，`pip check` 无依赖冲突。从 `main/` 执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -W error -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache src tests benchmarks
.\.venv\Scripts\python.exe benchmarks\benchmark_gpu.py --batch-size 32 --nodes 64 --variant full --iterations 20 --warmup 5 --repeats 5
.\.venv\Scripts\python.exe benchmarks\benchmark_gpu.py --batch-size 32 --nodes 64 --variant full --iterations 10 --warmup 3 --repeats 3 --include-preparation
.\.venv\Scripts\python.exe benchmarks\benchmark_gpu.py --batch-size 16 --nodes 32 --variant full --iterations 10 --warmup 3 --repeats 3 --backward
```

### 备份

本轮源代码备份：`../../backups/main-pre-theory-20260903-181334.zip`，包含 `main/`
和原 `READMEv0.3.md`，不含虚拟环境或数据缓存。SHA256：

```text
92127CF185B323EFF10C50F904FD0A3B61C9944934B7EFBC33AB3B9D2E66FA25
```

恢复时先解压到独立预览目录，不直接覆盖当前工程。上一轮备份也继续保留。
