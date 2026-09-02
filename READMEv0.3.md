# AST-Boost：面向 Graph Transformer 的符号不变高阶谱编码

> Adaptive Spectral-encoding Boost：把拉普拉斯位置编码从「带符号噪声的坐标」改成「良定的相对谱核 + 可学习的一阶/二阶谱场」。
>
> 本文档是项目说明书与开发基线（**v0.3**）。v0.2 完成了可行性收缩与战略聚焦；v0.3 在其基础上**恢复了完整数学证明**（v0.2 压缩掉了命题 2 证明、不变环与四阶矩）、**新增命题 3（相对谱核的不变性与简并安全条件）**、修正引用与若干工程细节。逐条增量见〈相对原方案改了什么〉末节。

---

## 目录

- [文档状态](#文档状态)
- [可行性结论](#可行性结论)
- [相对原方案改了什么](#相对原方案改了什么)
- [问题与主张](#问题与主张)
- [数学框架](#数学框架)
- [推荐架构](#推荐架构)
- [实现规范](#实现规范)
- [安装与接口草案](#安装与接口草案)
- [实验协议](#实验协议)
- [复杂度](#复杂度)
- [相关工作](#相关工作)
- [风险与缓解](#风险与缓解)
- [路线图](#路线图)
- [引用](#引用)
- [License](#license)

---

## 文档状态

- 仓库目前是研究规格书，**尚无已复现的基准数字**。下文不出现任何声称的精度。
- 代码接口是规格而非现成实现；实现时以本文件为准。
- 成功的第一标准不是「模块是否齐全」，而是：**在相同骨干与参数预算下，能否稳定超过 SignNet-PE 与 RWSE**。

---

## 可行性结论

### 总判

**静态图上的谱位置编码增强：可行，且值得做。**
前提是把项目收成一件事：给 Graph Transformer 提供一套**符号/基不变、可预计算、可插入 GPS 的谱 PE/SE**。不要同时做图像骨干、交通预测和从零训练新 Transformer。

原草案（v0.1 及更早）的数学诊断（外积不能吸收符号、局部层只能看到 `|x_i|`、简并会把歧义群升级成 `O(m)`）是对的，必须保留。原草案的产品形态不可行：创新点被拆成三件已发表工作的拼接，主实验选错数据，图像与动态图会稀释唯一站得住的假设。

### 模块可行性

| 模块 | 可行性 | 对最终效果的作用 | 结论 |
|---|---|---|---|
| 相对谱核注意力偏差 `K = U g_θ(Λ) Uᵀ` | 高 | **最大性价比**。天然符号不变（命题 3），简并时退化为谱投影，直接给注意力结构先验 | **默认开启** |
| SignNet 一阶谱场（沿用） | 高 | 绝对 PE 的合格下限；不复现它就无法声称改进 | **必做基线，内嵌于完整模型** |
| 简并分块 / 子空间投影 | 高 | ZINC 相当比例的图存在重/近重特征值（内部口径约 64%，**待 v0.1 统计脚本实测后落档**）；丢掉块内信息会伤害主实验 | **默认开启** |
| 二阶谱场 `w_pq = u_p ∘ u_q` 的 SignNet | 中 | **本项目唯一清晰的新假设**；有表达力缺口，但计算贵、增益未验证 | **主创新，做成可开关模块** |
| 特征值门控 `g_θ(λ)` | 高 | 便宜、无歧义；单独拿出来不算创新 | **作为核与 PE 的标量调制，不单独立项** |
| `svec(\|S_i\|)` 局部二次基 | 高 | 理论信息不超出 `\|x_i\|`（命题 1 推论），MLP 可学；最多当轻量变体 | **Lite 可选，不当主贡献** |
| 规范固定 `D*` | 低 | 过零不连续；`r=𝟙` 对非平凡特征模整体失效 | **静态图禁用** |
| 多项式滤波 `Y = g(L)X` 当「可学习谱基」 | 高但跑题 | 这是 ChebNet/BernNet，不是 PE。塞进来只会造成对照不干净 | **可作 GPS 的局部 MPNN，不进 PE 故事** |
| 图像 DCT/FFT 对照 AFNO | 低收益 | 规则网格解析基无符号歧义，打的是别人的主场 | **v1 不做** |
| PEMS04/08 上的「动态谱流」 | **当前设定下无意义** | 路网拓扑固定，`L` 不随时间变，谱流恒为 0 | **换数据或降为 Track B** |
| GROUSE 式增量跟踪 | 中低 | 实现与数值坑多，须先证明离散帧主角度有用 | **不进 v1** |
| 磁拉普拉斯 / 真相位 | 另开一篇 | 复 Hermitian 才有规范的幅角 | **远期** |

### 可证伪的主假设

> **H1.** 把每个特征向量当图信号做 SignNet（一阶）之后，再对截断的二阶场 `w_pq = u_p ∘ u_q` 做同一个构造，能提供 **GNN(u_p ⊙ u_q) 不能由 mix(GNN(u_p), GNN(u_q)) 代替** 的结构信息，并在 GPS 预算内降低 ZINC MAE 或提升 LRGB 指标。
>
> **H2.** 相对谱核 `K_ij = Σ_p g_θ(λ_p) u_pi u_pj` 作为加性注意力偏差，在长程任务上优于「只把 PE 拼进 token」以及 Graphormer 最短路偏差的谱替代。

若 H1 在参数匹配后不成立，项目仍可退化为 **不变谱核偏差 + 简并感知 SignNet** 的工程组合（论文贡献变弱，但系统可发布）。若 H2 也不成立，则不宜继续堆模块。

### 明确不做（v1）

- 不从零写一个新的 Graph Transformer 骨干；**插入 GraphGPS**（GatedGCN + Transformer）。
- 不以 PCQM4Mv2、图像分类、PEMS 交通预测为 v1 主表。
- 不把「可学习谱滤波替代特征分解」讲成核心创新。
- 不用 `L + εI` 或随机扰动去「拆简并」——那只会在子空间里随机选基。
- 不把 `svec(|S_i|)` 写成「恢复了频率间交互」；局部层恢复不了符号相干，只是 `|x_i|` 的二次展开。

---

## 相对原方案改了什么

原方案要改的不是「符号问题不存在」，而是**叙事、默认模块和实验对象**。

1. **创新口径收窄。** SignNet、多项式谱滤波、Graphormer 加性偏差都是已有工作。本项目的新东西只保留：二阶谱场的不变编码，以及把相对谱核写成默认的、简并安全的注意力偏差。
2. **先做相对核，再做二阶场。** `u_pi u_pj`（同一频率、两个节点）在符号翻转下不变，不必 SignNet。原方案把力气花在更难的 `u_pi u_qi`（同一节点、两个频率）上，却漏掉了这个更稳的相对 PE。对 Transformer 效果，核偏差通常比局部二次基更直接。
3. **拉普拉斯用对称归一化 `L_sym`。** 组合拉普拉斯 `D-A` 的特征向量会偏向高度数节点，GPS / SignNet 主实验用的是归一化谱。默认改 `L_sym`，组合拉普拉斯只作消融。
4. **骨干固定为 GPS，PE 预计算。** 否则无法与 GraphGPS、SignNet、RWSE 公平对比，也最容易把调参开销当成方法收益。
5. **动态图从主线拿掉。** PEMS 的邻接来自固定路网，`L` 不变则没有谱流。动态分支改为 Track B，且必须换「边真的在变」的数据。
6. **简并从「丢掉块内 pair」改为「用投影」。** 在 ZINC 这类分子图上，丢掉块内信息会系统性地削弱主表。块内用 `P_B = U_B U_Bᵀ`，需要更强块内结构时再上精简 BasisNet。
7. **对照补上随机反号增强。** 这是 LapPE 的标准便宜基线；只打「未反号的 LapPE」没有说服力。

### v0.3 相对 v0.2 的增量（本轮评审合并）

1. **恢复完整证明链**：命题 2 的不变性证明、命题 1 的多项式版不变环 `ℝ[x]^{ℤ₂ᵏ} = ℝ[x₁²,…,xₖ²]`、二阶场的四阶矩谱不变量——v0.2 压缩掉了，导致「为什么逐点 ψ 不算 H1」只有断言没有依据。
2. **新增命题 3 与推论**：相对谱核的符号不变性（一行证明），以及**简并安全条件——`g_θ` 必须在分块内取常值**，否则核是基依赖的。v0.2 把「近简并时对块用 `g(λ̄_B)`」当作工程技巧，实际它是必要条件，应写成命题并进单测。
3. **注意力偏差逐图标准化**：`K` 的元素量级随图尺寸与 `g_θ` 谱漂移，直接加进 logits 会造成跨图尺度不一致。新增 `bias.standardize`（每图对非对角元 zero-mean / unit-std 后再乘 `α_h`）。
4. **引用修正**：PEG 与 EquivStableLapPE 是同一篇工作（Wang, Yin, Zhang, Li, ICLR 2022），相关工作表合并；SPE 确认为 Huang et al., ICLR 2024。
5. **统计口径诚实化**：「ZINC 约 64% 图有高维特征空间」暂无公开来源，标注待验证，改为 v0.1 阶段先跑统计脚本落档。
6. **工程细节**：`eigsh` shift-invert 在 `L_sym` 精确奇异时的退化写法；修正 `skip_zero` 说明笔误与 `pip install` 版本号引号。
7. **负对照预期锐化**：逐点 `ψ` 的二阶场退化为 `|x_i|` 的二次展开（Lite 水平，无跨节点信息），因此预期**不超过 Kern**；若显著更好，先查 bug 与参数匹配。

---

## 问题与主张

### 痛点

拉普拉斯特征向量是图 Transformer 最常用的全局 PE 之一，但求解器返回的是等价类而不是函数：

- 简单特征值：`u ↦ -u`；
- 重特征值：`U_B ↦ U_B Q`，`Q ∈ O(m)`；
- 批次里两个同构图、两次分解，编码可以不一致，训练目标因此不良定。

业界常用两条路：训练时随机反号（不保证不变，不管简并），或 SignNet / BasisNet（一阶、按特征向量分别处理）。RWSE 没有这套歧义，因此在分子图上经常比生 LapPE 更稳。谱方法要赢，必须先把不变性做对，再证明它比 RWSE 多提供了谱几何信息。

### 主张（按优先级）

1. **相对谱核是默认的结构偏差。** 节点 `i,j` 的低频坐标内积（可学习频率响应加权）是符号不变、块内基不变的相对 PE，适合加到 softmax 之前。
2. **一阶 SignNet 是绝对 PE 的下限。** 完整模型包含它，消融时关掉二阶场即得到 SignNet 对照。
3. **二阶谱场补上「按频分别做 GNN 再混合」的缺口。** `GNN(u_p ⊙ u_q) ≠ mix(GNN(u_p), GNN(u_q))`。这是 H1，也是真正要验证的贡献。
4. **任务自适应只通过 `g_θ(λ)` 与后续网络发生，不通过对 `U` 做无约束调制。** 后者在符号意义下不是良定的可学习对象。

### 非主张

- 不声称发明了符号不变谱网络（SignNet）。
- 不声称发明了谱注意力（SAN / Specformer）或谱滤波（ChebNet、BernNet）。
- 不声称局部 `|S_i|` 恢复了相位。
- 不声称本方法在图像或固定拓扑的交通图上自动成立。

---

## 数学框架

### 1. 谱与记号

图 `G=(V,E)`，`|V|=N`。**默认**用对称归一化拉普拉斯

$$L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2} \in \mathbb{R}^{N\times N}.$$

孤立点令 `D_{ii} ← D_{ii} + δ`（`δ ≈ 10^{-6}`），避免 `D^{-1/2}` 失败。`L_{\mathrm{sym}} = U Λ Uᵀ`，`0 = λ_1 ≤ ⋯ ≤ λ_N`。连通图上 `λ_1 = 0` 的模是 `D^{1/2}𝟙`，几乎不含位置信息，**默认丢掉零空间，取最小的 k 个正特征值**。多连通分支时零空间维数等于分支数，按简并块处理，不要假装它是一组独立的 PE 坐标。

记 `U = [u_1,…,u_k] ∈ ℝ^{N×k}`，节点 `i` 的谱坐标

$$x_i = (u_{1i},\dots,u_{ki})^\top \in \mathbb{R}^k.$$

消融才使用组合拉普拉斯 `L = D-A`。

### 2. 良定义性

特征分解不唯一：对 `D = \mathrm{diag}(s_1,…,s_k)`、`s_j ∈ {±1}`，

$$L = (UD)Λ(UD)^\top.$$

特征值互异时（除列排序外）这就是全部离散自由度。编码 `E` 作为节点特征必须满足

$$\boxed{\,E(UD)=E(U)\quad\forall D\in\{\pm 1\}^k\,},\qquad E(ΠUD)=ΠE(U).$$

原草案的 `S_i = x_i x_iᵀ` 两条都不满足。

简并时，块 `B` 上 `U_B ↦ U_B Q`、`Q ∈ O(m_B)`，不变量必须是谱投影 `P_B = U_B U_Bᵀ` 的函数，而不是某一组坐标的函数。

### 3. 原外积编码的协变

`U → UD` 时 `x_i → D x_i`，故

$$S_i = x_i x_i^\top \;\longmapsto\; D S_i D.$$

对角元 `u_{pi}^2` 不变；非对角 `u_{pi}u_{qi}` 乘上 `s_p s_q`。同一节点的 `vec(S_i)` 可落入 `2^{k-1}` 种符号模式。外积只是把「逐分量反号」变成「逐行列联合反号」，**并不吸收歧义**。

### 4. 局部不可能性（仍然成立）

**命题 1.** `ℤ₂^k` 以 `x ↦ Dx` 作用在 `ℝ^k` 上。连续映射 `f` 满足 `f(Dx)=f(x)` 对所有 `D` 成立，当且仅当 `f(x)=g(|x_1|,…,|x_k|)`。

**证明.** 取 `D_x = \mathrm{diag}(\mathrm{sign}(x))`（零分量任意），则 `x = D_x|x|`，故 `f(x)=f(D_x|x|)=f(|x|)`；令 `g = f|_{[0,∞)^k}`。反向显然。∎

**多项式版（v0.3 恢复）.** 单项式 `x^α = ∏_j x_j^{α_j}` 的变换因子是 `∏_j s_j^{α_j}`，不变当且仅当每个 `α_j` 为偶数。故不变环为

$$\mathbb{R}[x]^{\mathbb{Z}_2^k} = \mathbb{R}[x_1^2,\dots,x_k^2].$$

因此任何只依赖单节点 `x_i`（或只依赖 `S_i`）的连续不变特征，信息量不超过 `|x_i|`。`|S_i|_{pq} = |u_{pi}||u_{qi}|` 是 `|x_i|` 的二次单项式，对线性层是显式核、对 MLP **不是新信息**。Lite 变体可以把它当特征展开；主论文不要把这一项写成「频率交互的修复」。

### 5. 三类良定对象（按性价比）

把「要编码什么」按变换律分开，而不是先做一个大外积再修补。

| 对象 | 公式 | 符号翻转 | 简并 `O(m)` | 用法 |
|---|---|---|---|---|
| 相对谱核（优先） | `K_{ij}^{(g)} = Σ_p g(λ_p)\, u_{pi} u_{pj}` | 每项 `s_p^2=1`，不变（命题 3） | `g` 块内取常值时 = `g(λ̄_B)(P_B)_{ij}` | 注意力偏差 |
| 一阶谱场 | `u_p ∈ ℝ^N` | 全局乘 `±1` | 块内坐标无意义，改用 `P_B` 的列/对角 | SignNet 绝对 PE |
| 二阶谱场 | `w_{pq} = u_p ∘ u_q` | 全局乘 `s_p s_q` | 块内 pair 无定义，必须丢弃或改投影 | 本项目 H1 |

#### 5.1 相对谱核（推荐写成滤波核）

$$K^{(g)} = U\,\mathrm{diag}\big(g_\theta(\lambda_1),\dots,g_\theta(\lambda_k)\big)\,U^\top,\qquad b_{ij} = \alpha\, K^{(g)}_{ij}.$$

**命题 3（核不变性，v0.3 新增）.** 对任意 `D ∈ {±1}^k`，`K^{(g)}` 不变。

**证明.** `(UD)\,g_\theta(Λ)\,(UD)^\top = U\,D\,g_\theta(Λ)\,D\,U^\top = U\,g_\theta(Λ)\,U^\top`，因为对角阵 `D` 与对角阵 `g_\theta(Λ)` 可交换且 `D² = I`。∎

**推论（简并安全条件，v0.3 新增）.** 在简并块 `B` 内，若 `g_θ` 非块内常值，则 `Σ_{p∈B} g_θ(λ_p)\,u_p u_p^\top` 随 `Q ∈ O(m_B)` 变化（基依赖）；取块常值 `g_θ(λ̄_B)` 时它等于 `g_θ(λ̄_B)\,P_B`，对一切 `Q` 不变。**因此实现必须在分块后把 `g_θ` 的输入钳到块均值 `λ̄_B`**，这进 `kernel_invariance` 单测。

近简并时不要对块内单个 `u_p` 建核，改为

$$K^{(g)}_{ij} \leftarrow \sum_B g_\theta(\bar\lambda_B)\, (P_B)_{ij},\qquad P_B = U_B U_B^\top.$$

这同时解决符号与基。

**与 RWSE/SPE 的关系（v0.3 新增注记）.** 核的对角元 `K^{(g)}_{ii} = Σ_p g_θ(λ_p)\,u_{pi}^2`：取 `g = e^{-tλ}` 即截断热核对角（SPE 使用的统计量家族）；取 `g` 为随机游走级数的截断则与 RWSE 同族。即 **Kern 的对角部分已覆盖热核/RW 类 PE，非对角部分是超出它们的相对几何**——这解释了 Kern 的合理下限，也让「打赢 RWSE」的归因必须靠非对角项（消融时单独报对角-only 版本）。

`g_θ` 用 Bernstein / Chebyshev 基或对 `λ` 的小 MLP，输出不强制为正。需要 PSD 解释时再 `softplus`。多头注意力让每头一套 `g_θ^{(h)}`，不同头看不同频段（与 SAN 的「按频率学注意力」同族，但这里作用在**已经不变的核**上，而不是对带符号的 `u_{pi}` 做 token）。

#### 5.2 一阶场：沿用 SignNet

$$\big(z^{\mathrm{(1)}}\big)_i = \rho\Big(\big[\psi(u_p)+\psi(-u_p)\big]_{p=1}^{k}\Big)_i$$

`ψ` 必须是**置换等变**的（GIN / GatedGCN 作用在以 `u_p` 为节点信号的图上）。逐点 MLP 的 `ψ` 退回命题 1，只是 `|u_{pi}|` 的另一种基。

#### 5.3 二阶场：H1

$$w_{pq} := u_p \circ u_q \in \mathbb{R}^N,\qquad (w_{pq})_i = u_{pi}u_{qi}.$$

变换律：`w_{pq} ↦ s_p s_q w_{pq}`，与单个特征向量同构，故

$$z^{\mathrm{(2)}} = \rho\Big(\sum_{(p,q)\in\mathcal{P}} \psi(w_{pq})+\psi(-w_{pq})\Big)$$

仍不变。

**命题 2（不变性，v0.3 恢复证明）.** `w_{pq}(UD) = s_p s_q\,w_{pq} = ± w_{pq}`，故 `ψ(w_{pq})+ψ(-w_{pq})` 不变。
**证明.** `w → -w` 时，和式 `ψ(w)+ψ(-w)` 的两项恰好交换位置，逐点相等。∎

`𝒫` 必须避开简并块内 pair。

**四阶矩谱不变量（v0.3 恢复）.** 场族 `{w_{pq}}` 的内积

$$\langle w_{pq}, w_{rs}\rangle = \sum_{i=1}^{N} u_{pi}u_{qi}u_{ri}u_{si}$$

是图的经典谱不变量（四阶矩），**不**能由任何逐节点 `|x_i|` 特征表达——它是跨节点的四次结构。跨节点 `ψ` 的 GNN 恰好能逼近这类量的图信号版本，这是二阶场表达力缺口的具体形态。

**表达力（必须写清楚边界）：**

- 逐点 `ψ`：`ψ(w)+ψ(-w)` 是 `|w_{pq}[i]|` 的函数，与 `|x_i|` 信息等价，**不要作为 H1 的实现**。
- 跨节点 `ψ`（GNN）：`GNN(w_{pq})` 看到的是乘积场在图上的振荡，一般**不能**由「分别对 `u_p`、`u_q` 做 GNN 再在节点上拼接」实现。这是 H1 的全部理论内容，不是万能逼近器意义上的「比 SignNet 更强」——若把所有频率当作多通道一次性送进一个未做符号设计的 GNN，表达力可以更大，但会重新引入符号歧义。二阶场是在**保持 SignNet 不变性构造**的前提下，显式加入跨频率的图信号。

场数 `k(k+1)/2` 不可全开。默认只取**低频之间的 pair**（见实现规范），不要按 `‖w_{pq}‖²` 做与标签有关的筛选（泄漏）。

### 6. 简并与近简并

以相对间隙分块：

$$|\lambda_p-\lambda_q| \le \varepsilon \cdot \max(\bar\lambda, \tau),\qquad \tau=10^{-6},\ \varepsilon=10^{-2}.$$

绝对阈值 `|λ_p-λ_q|<ε` 会随拉普拉斯种类和图谱尺度漂移，不作为默认。

块内：

- 相对核用 `P_B`（命题 3 推论）；
- 绝对 PE 用对角 `(P_B)_{ii} = Σ_{j∈B} u_{ji}^2`，需要块内方向时用 BasisNet（对 `P_B` 的等变网络），v1 可先只上对角 + `P_B` 的若干次幂对角（热核对角，基不变）；
- **禁止**对块内 `(p,q)` 建 `w_{pq}`。

注意：`L ← L + εI` 只平移谱，**不拆简并**，对条件数几乎无帮助。给矩阵元素加随机抖动会拆简并，但基变成噪声，**禁止**。

### 7. `g_θ` 用在哪里

两处合法、一处不要和 PE 混写：

| 用法 | 公式 | 角色 |
|---|---|---|
| 核滤波 | `K = U g_θ(Λ) Uᵀ` | 相对 PE，主路径 |
| 特征值嵌入 | 把 `g_θ(λ_p)` 或 `MLP(λ_p)` 拼进频率 token | 给 SignNet 的 `ρ` 看频率标签 |
| 图卷积 | `Y = g_θ(L) X` | 这是局部 MPNN，不是 PE。若用，只作为 GPS 的 GatedGCN 替代，单独消融 |

对称矩阵特征分解几乎处处可微；真正的问题是重根处梯度病态。v1 **预计算** `(Λ,U)` 并断开反传，不把 `eigh` 放进训练图。可学习部分全在 `g_θ、ψ、ρ、α`。不要把 Power Iteration 当可微性方案。

### 8. 规范固定为什么不当主方案

`D^\ast = \mathrm{diag}(\mathrm{sign}(U^\top r))` 有两处硬伤：

1. `⟨u_j, r⟩` 过零时 `D*` 跳变；
2. 连通图上 `λ_j>0` 的模与 `D^{1/2}𝟙`（或组合拉普拉斯下的 `𝟙`）正交，取常数参考时内积恒为 0。

只允许在 Track B 里做**相邻时间帧**的符号 Procrustes（变化小、间隙大时 `|(Φ_tᵀ Φ_{t-1})_{jj}| ≈ 1`）。静态图只用不变式。

### 9. Track B：动态谱流（不进 v1 主表）

仅当 **L_t 真随 t 变**（边增删、重连、时序交互图）才有定义。PEMS04/08 的 `adj` 来自固定路网，整段轨迹 `L` 为常数，主角度恒为 0，不能用来验证本分支。

帧间子空间的主角度

$$\theta_j = \arccos \sigma_j\big(\Phi_t^\top \Phi_{t-1}\big)$$

对左右正交（含反号）乘法不变，这是正确的不变量。它是**图级**量，应作为快照的全局特征或时间 token，不要写成 `λ_i(t)` 那种节点下标混乱的拼接。

Davis–Kahan：`|sin Θ| ≲ ‖L_t-L_{t-1}‖_2 / gap`。间隙小时主角度本身不稳定，需按简并块比较子空间而不是逐列。

符号 Procrustes `D_t^\ast = \mathrm{diag}(\mathrm{sign}(\mathrm{diag}(Φ_tᵀ Φ_{t-1})))` 仅用于「需要把 `ũ_t` 当连续轨迹画出来」的可视化或辅助特征；数值上仍以主角度为准。GROUSE 类增量更新放到该 Track 的第二阶段，先用每帧独立 `eigsh` + 主角度证明任务有信号。

候选数据（必须拓扑可变）：合成边重连、时序链接（Bitcoin-OTC、UCI 消息）、或 SLATE 类 supra-Laplacian 设定。不要在固定 `A` 的交通图上硬做谱流。

### 10. 编码方案汇总

| 编码 | 符号 | 简并 | 信息 | v1 |
|---|---|---|---|---|
| 原始 `x_i` / `S_i` | 否 | 否 | 混入规范噪声 | 只作反例 |
| LapPE + 随机反号 | 训练期近似 | 否 | 一阶坐标 | 必做基线 |
| `svec(\|S_i\|)` | 是 | 块内需改投影对角 | 不超过 `\|x_i\|` | Lite |
| SignNet 一阶 | 是 | 需分块 / BasisNet | 一阶场的跨节点结构 | 完整模型的一部分 |
| 滤波谱核 `K^{(g)}` | 是（命题 3） | `g` 块内常值则是 | 相对几何 / 热核（对角）+ 非对角 | **默认开** |
| 二阶场 SignNet | 是 | 丢块内 pair | 跨频率图信号 | H1 开关 |
| `D*` 规范固定 | 近似 | 近简并处坏 | 名义全 | 禁用 |

---

## 推荐架构

目标不是新骨干，是 **GPS 的 PE/SE 插件**。这样数字能对照公开配置，效果来自编码而不是来自「换了一个更大的 Transformer」。

```
Data ─预计算─ L_sym 的 top-k (λ, U)，简并分块
                │
                ├─ 相对 SE:  K^{(g)}_h = U g_θ^{(h)}(Λ) Uᵀ   →  逐图标准化 → 每头注意力偏差
                │
                ├─ 绝对 PE:   SignNet^{(1)}(U)                 →  z⁽¹⁾
                │
                └─ 可选 H1:  SignNet^{(2)}({w_pq}_{(p,q)∈𝒫})  →  z⁽²⁾

token = [x_node ‖ z⁽¹⁾ ‖ z⁽²⁾ ‖ MLP(λ 摘要)]
GPS layer: GatedGCN（局部） + Transformer（全局，attn += α_h · std(K_h)）
```

### 三个交付变体

| 名称 | 内容 | 何时用 |
|---|---|---|
| **AST-Lite** | 预计算 U + `MLP(\|x_i\|, λ)` + 谱核偏差 | 最快出数、跑通管线 |
| **AST-Kern** | Lite 的 PE 换成 SignNet 一阶 + 谱核偏差 | 主对照，应不低于公开 SignNet+GPS |
| **AST-Full** | Kern + 二阶场（共享 `ψ`，低频 pair） | 验证 H1；参数/FLOPs 与 Kern **匹配**后再比 |

完整模型默认 = **AST-Full**，论文主表同时报 Kern 与 Full。若 Full 不优于 Kern，发表口径退回 Kern，H1 记为否定结果（仍然有信息）。

### 二阶场截断（默认）

`k=8` 时全 pair 为 36，负担过大。默认：

- 只用前 `k_0=4` 个低频之间的 pair（含对角场 `u_p ∘ u_p`，即 `u_p^{⊙2}`），`|𝒫|=10`；
- `ψ` **跨场共享**，按 batch 维把场叠成 `m` 个图信号一次前向；
- 简并块内 pair 从 `𝒫` 删除；
- `k_0` 与 `k` 分开设：一阶 SignNet 仍可用 `k=8` 或 `min(16, N-1)`。

不要用任务标签选 pair。需要消融再比较：仅 Fiedler 乘积 `{(1,j)}` vs 低频全 pair vs 能量最大的 m 个（无监督 `‖w_{pq}‖²`）。

### 注意力偏差

已有全局注意力时，偏差的额外显存与 `O(N²)` 注意力同阶，ZINC / Peptides 可接受。实现：

```
K_h = U g_h(Λ) Uᵀ                       # 命题 3：符号不变；分块处 g 输入钳到 λ̄_B
K_h = standardize(K_h, offdiag=True)    # v0.3：每图 zero-mean / unit-std（非对角元）
attn_h += α_h * K_h                     # α_h 可学习标量，初值 0.1
```

要点（v0.3 修订）：

- **逐图标准化**再乘 `α_h`：`K` 的元素量级随图尺寸 `N` 与 `g_θ` 的谱响应漂移，不标准化会让 logits 的有效温度逐图不同，小图被淹没、大图被放大。
- `K_h` 不进计算图对 `U` 的反传；梯度只流向 `g_θ` 与 `α_h`。
- `g_θ` 参数量保持很小（每头 < 32）。
- 若发现 `⟨z_i, z_j⟩` 与拼进 token 的 PE 高度冗余，以 **滤波核** 为准，不要再叠一层学到的 `⟨z_i,z_j⟩`，以免不可解释的双计数。

### 骨干与预算

- 骨干：GraphGPS，`GatedGCN + Transformer`，层数/宽度跟从对应数据集的公开 GPS 配置。
- ZINC-subset / LRGB：参数预算对齐文献常用的 **~500k**。
- 加二阶场时用减小 `ψ` 宽度或 GPS 隐藏维做 **参数匹配**；同时记录一步的 FLOPs，避免「慢三倍但参数相同」的假公平。

---

## 实现规范

### 谱计算

- 小图（`N ≤ 256`，ZINC 几乎都在此）：密集 `eigh` 一次，取最小正谱。
- 更大图：`scipy.sparse.linalg.eigsh`，对最小代数谱用 **shift-invert**（`sigma=0, which='LM'`）而不是 `which='SM'`。PyTorch 侧可用 `torch.lobpcg`，但批次图大小不一，**优先离线预计算写入 `Data`**。
- **shift-invert 退化写法（v0.3）**：`L_sym` 在连通图上有精确零特征值，`sigma=0` 时移位矩阵奇异，LU 分解可能报警或失败；改用 `sigma=1e-5`（或对 `L+δI` 分解后把谱平移回去），结果一致。
- 不存在 `torch.scipy.sparse.linalg.eigsh` 这一接口，文档与代码都不要写它。
- 训练期默认 **冻结 U**。符号不变性单测在测试夹具里对 `U` 乘随机 `D`，不在训练循环里重分解。

### 模块划分

- `spectral.precompute`：拉普拉斯、分块、缓存 `λ, U, block_ids, λ̄_B`。
- `spectral.kernel`：`g_θ`（含块内钳位）、`K_h` 构造与逐图标准化。
- `spectral.signnet`：共享的 `ψ/ρ`，一阶与二阶共用实现，二阶只换输入信号。
- `spectral.fields`：按 `𝒫` 与分块规则建 `w_pq`。
- `spectral.gps_adapter`：把 PE 写入 `x`、把 `K` 写入注意力 bias 的钩子。

不要在 PE 模块里再实现一层完整 Transformer。

### 必须落地的测试（比新模块更优先）

```
pytest tests/ -m sign_invariance
pytest tests/ -m degeneracy
pytest tests/ -m permutation
pytest tests/ -m kernel_invariance
```

| 测试 | 断言 |
|---|---|
| 符号翻转 | 同图 `U` 与 `UD`，Lite/Kern/Full 的节点编码与 `K` 在 `1e-5` 相对误差内一致 |
| 简并 | 构造 `m=2` 的块，随机 `Q∈O(2)` 作用后，`K` 与 `(P_B)_{ii}` 一致；Full 不读取块内 pair |
| **核简并安全（v0.3）** | 块内 `g_θ` 未钳位时 `K` 随 `Q` 漂移（负例）；钳位后一致（正例） |
| 置换 | `Π` 同时作用于 `A,X,U`，节点输出等变 |
| 同构批次 | 两个同构拷贝在同一 batch 中编码一致（这是原方案最容易炸的点） |
| 零空间 | 丢掉 `λ≈0` 后，连通图上常数平移不改变 `K` 的非对角结构（按实现定义检查） |

先红灯测试、再写模块。没有 `sign_invariance` 通过，不准跑 ZINC。

### 超参默认值

| 参数 | 默认 | 说明 |
|---|---|---|
| `laplacian` | `sym` | `sym` / `rw` / `comb`（后两者仅消融） |
| `k` | 8 | 一阶场个数；小图可 `min(16, N-1)` |
| `k0_pairs` | 4 | 二阶场用的低频个数 |
| `skip_zero` | true | 丢掉 `λ≈0` 零空间模（连通图即常数模，无位置信息） |
| `degeneracy.eps` | 1e-2 | 相对间隙 |
| `kernel.heads` | 与 Transformer 头数相同 | 每头独立 `g_θ` |
| `kernel.basis` | bernstein | 阶数 8 |
| `kernel.block_clamp` | true | `g_θ` 输入钳到块均值 `λ̄_B`（命题 3 推论） |
| `bias.alpha` | 可学习，初值 0.1 | 可 per-head |
| `bias.standardize` | true | 每图对 `K` 非对角元 zero-mean/unit-std 后再乘 `α_h` |
| `sign.psi` | GIN × 2 | 禁止默认逐点 MLP |
| `sign.hidden` | 64 | Full 与 Kern 参数匹配时优先改这里 |
| `precompute` | true | 写入磁盘缓存 |

---

## 安装与接口草案

规格阶段的依赖（实现时再钉版本）：

```bash
# Python 3.10+
pip install "torch>=2.1" torch-geometric numpy scipy pytest
# 对齐 GraphGPS 实验时，按其仓库安装 dgl / performer-pytorch 等可选件
```

谱求解用 `scipy.sparse.linalg.eigsh` 或密集 `numpy.linalg.eigh` / `torch.linalg.eigh`。

```python
# 规格示意：预计算 + 冻结 U，训练只走 g_theta / SignNet / GPS
from ast_boost import precompute_spectrum, ASTBoostPE, gps_forward

data.spectral = precompute_spectrum(data.edge_index, n=data.num_nodes, k=8)
pe = ASTBoostPE(k=8, k0_pairs=4, variant="full")   # lite | kern | full
x_pe, bias = pe(data.x, data.spectral)              # bias: (H, N, N) 或稀疏
out = gps_forward(x_pe, data.edge_index, attn_bias=bias)
```

---

## 实验协议

### 原则

1. **先复现，再改进。** 在 ZINC-subset 上复现 GPS+RWSE 与 GPS+SignNet 到文献量级（允许实现细节误差，但数量级必须对）。复现失败则禁止加二阶场。
2. 开发用 **4 seeds**，主表再用 4–10。一上来 10 seeds × 6 个模块是在烧机器，不是更科学。
3. 每步只开一个开关。Full vs Kern 必须参数匹配。
4. 报 mean±std，并报相对 Kern 的配对差值。H1 只看这组差值。

### 数据集（v1 主线）

| 阶段 | 数据 | 为什么 |
|---|---|---|
| 开发 / 复现 | ZINC-subset（12k） | 标准、便宜、SignNet 有数；分子图短程，**不是** H2 的最佳舞台，只作可比主表 |
| 主验证（H2 / 长程） | LRGB：Peptides-func、Peptides-struct | 结构先验更可能进注意力；用公开 GPS 预算 |
| 可选 | PascalVOC-SP | 超像素上 LapPE 历史上有用，作为谱核的第二现场 |
| 不做（v1） | PCQM4Mv2、PEMS、ImageNet 式图像 | 太大或与假设无关 |

Track B 另列数据，不进 v1 主表。

### 必做基线

| 名称 | 说明 |
|---|---|
| GPS，无 PE | 下限 |
| GPS + LapPE + **随机反号** | 标准便宜谱基线 |
| GPS + RWSE | 分子图上很强，必须打 |
| GPS + SignNet | 即 AST-Kern 关掉核偏差的版本；再单独报 Kern |
| GPS + Graphormer 最短路偏差 | 与谱核偏差对照（H2） |
| **Kern 对角-only**（v0.3） | `K` 只留对角（热核/RW 类统计量），隔离「非对角相对几何」的净贡献 |
| AST-Lite / Kern / Full | 本方法 |

不是主贡献、但容易被审稿人问的：SAN / Specformer 用公开实现或「同骨干 + 其 PE」在 ZINC 上点一下即可，不要为它们重做全部 LRGB。

### 消融顺序（按信息增益 / 成本）

1. 复现 GPS+RWSE、GPS+LapPE（反号）、GPS+SignNet。
2. 加谱核偏差 → AST-Kern。看 H2。
3. Lite：`MLP(|x|)` ± 核，确认管线与不变性测试，不当主贡献。
4. 开二阶场 → AST-Full（`ψ` 必须是 GNN）。看 H1。
5. 二阶场的负对照：逐点 `ψ`（预期退化为 Lite 水平、**不超过 Kern**——它没有跨节点信息；若显著更好则实验有 bug 或参数没匹配）。
6. 简并：关掉分块 vs 默认分块（ZINC 上应能看出差异）；同时报 `kernel.block_clamp` 开关。
7. `L_sym` vs `L_comb`。
8. 不要把 `Y=g(L)X` 塞进这条链。

### 成功 / 失败

- **发布为方法论文（完整故事）：** Full 在参数匹配下稳定优于 Kern，且 Kern 不低于 SignNet、并至少在 LRGB 之一上优于或持平 RWSE。
- **发布为短文 / 技术报告：** H1 失败但 Kern 在 LRGB 上相对最短路偏差或纯 SignNet 有稳定增益——贡献改为「简并安全的滤波谱核偏差」。
- **停止加模块：** Kern 无法进入 SignNet / RWSE 的复现带。回头查预计算、是否误用组合拉普拉斯、是否把零空间当 PE、以及 GPS 超参是否跑偏。

---

## 复杂度

| 步骤 | 成本 | 备注 |
|---|---|---|
| 预计算 top-k | 小图 `O(N³)` 一次；大图 Lanczos / shift-invert ≈ `O(T · nnz)` | 训练期摊销为零 |
| 谱核 `K_h` | `O(H N² k)` 或预计算后 `O(H N²)` | 与满注意力同阶 |
| SignNet 一阶 | `O(k · \|E\| · d · L_ψ)` | 与现有 SignNet 相同 |
| 二阶场 | `O(\|𝒫\| · \|E\| · d · L_ψ)`，默认 `\|𝒫\|=10`、共享 `ψ` | Full 的主要额外开销 |
| 满特征分解每步反传 | 不要 | v1 冻结 U |

Nyström / 随机 SVD 只在 `N` 大到 `eigsh` 也贵时再引入，不作为 ZINC 路径的一部分。

---

## 相关工作

| 模块 | 最接近的工作 | 本项目怎么用 |
|---|---|---|
| LapPE | Dwivedi & Bresson 2020 | 加随机反号后作为基线 |
| 符号 / 基不变 | SignNet / BasisNet，ICLR 2022 | 一阶沿用；二阶场是同一构造换信号；简并用投影 / 精简 BasisNet |
| Graph Transformer 配方 | GraphGPS | **唯一骨干** |
| 结构注意力偏差 | Graphormer | 对照；我们用谱核而不是最短路 |
| 谱注意力 | SAN、Specformer | 同族；我们把可学习部分放到不变核 `g(λ)`，不对带符号坐标做 token |
| 谱滤波 GNN | ChebNet、BernNet、GPR-GNN | 不进 PE 故事；可作 GPS 局部组件的额外消融 |
| 相对 / 稳定 PE | PEG（即 Equivariant & Stable PE，Wang et al., ICLR 2022）、SPE（Huang et al., ICLR 2024） | 近亲；谱核是滤波后的截断热核，对角部分与 SPE 的热核统计量同族 |
| RWSE | Dwivedi et al. / GPS | 必须打赢或讲清何时谱优于随机游走；对角-only 消融隔离差异来源 |
| 动态谱 | SLATE（supra-Laplacian） | Track B 的对照，不是 PEMS |
| 复谱 | MagNet | 真相位走这条，远期 |

---

## 风险与缓解

| 风险 | 为什么会发生 | 缓解 |
|---|---|---|
| 组合拳没有新贡献 | 审稿人把 Kern 看成 SignNet+Graphormer | 主文只强调 H1 与「不变滤波核」；Full 必须有独立增益 |
| H1 为假 | 共享 GNN 的低频 pair 不够，或 ρ 已能混频 | 接受否定结果；不要靠加宽网络硬造出差距 |
| 复现失败 | GPS 超参、PE 维度、是否用边特征 | 先对齐公开 yaml，再改 PE |
| 简并处理过度 / 不足 | ZINC 大量近简并 | 相对间隙 + 投影；单测随机 `Q`；`block_clamp` 负例测试 |
| 二阶场过拟合 | `\|𝒫\|` 过大 | 默认 `k0=4`；Dropout；参数匹配 |
| 核偏差与 PE 双计数 | token 已含 z，注意力再加 `⟨z,z⟩` | 只用 `K^{(g)}`，不加 `⟨z_i,z_j⟩` |
| 偏差尺度随图漂移 | `K` 量级依赖 `N` 与 `g_θ` | `bias.standardize` 逐图标准化（v0.3） |
| 把 PEMS 当动态图 | `L` 不变 | v1 删除；Track B 换数据 |
| 特征分解进训练图 | 近简并梯度爆炸、又慢 | 预计算冻结 |
| 注意力显存 | `O(N²)` | 主数据规模可接受；更大图改 Performer/Exphormer 时核偏差要改随机特征，不在 v1 |

---

## 路线图

- [ ] **v0.0** 简并统计脚本：ZINC-subset 上实测重/近重特征值比例，落档替换「64%（待验证）」
- [ ] **v0.1** 预计算 `L_sym` + 符号/置换/简并/核不变单测（无训练）
- [ ] **v0.2** GPS 接上，ZINC-subset 复现 RWSE 与 SignNet
- [ ] **v0.3** 滤波谱核偏差 → AST-Kern；H2 的第一次读数
- [ ] **v0.4** 简并分块 + 投影对角；对比「丢掉块 vs 投影」
- [ ] **v0.5** 二阶场 AST-Full，参数匹配消融；逐点 `ψ` 负对照
- [ ] **v0.6** LRGB Peptides 两任务；决定论文口径（Full 还是仅 Kern）
- [ ] **Track B（可选）** 拓扑可变的动态图 + 主角度；不做 GROUSE 直到离散帧有正信号

时间分配建议：约一半给 v0.0–v0.2 的复现与测试。没有这条，后面的模块没有解释权。

---

## 引用

实现与实验设计应对齐并致谢：

- Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*, 2020.
- Lim et al., *Sign and Basis Invariant Networks for Spectral Graph Representation Learning*, ICLR 2022.
- Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS)*, NeurIPS 2022.
- Kreuzer et al., *Rethinking Graph Transformers with Spectral Attention (SAN)*, NeurIPS 2021.
- Ying et al., *Do Transformers Really Perform Badly for Graph Representation? (Graphormer)*, NeurIPS 2021.
- Dwivedi et al., *Long Range Graph Benchmark (LRGB)*, 2022.
- He et al., *BernNet*, NeurIPS 2021.
- Bo et al., *Specformer*, ICLR 2023.
- Wang, Yin, Zhang & Li, *Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks (PEG)*, ICLR 2022.
- Huang, Lu, Robinson, Yang, Zhang, Jegelka & Li, *On the Stability of Expressive Positional Encodings for Graphs (SPE)*, ICLR 2024.
- Zhang et al., *MagNet*, 2021.（仅远期复谱）

---

## License

研究原型，尚未发布。内部验证并完成 v0.2 复现后再定开源许可。
