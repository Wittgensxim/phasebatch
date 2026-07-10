# 组会汇报：我们怎么压缩 Phase-Ordering 搜索空间

> 我们不是在暴力搜索 pass 顺序，而是在每个 IR state 上先找出哪些顺序决策是假的，把它们折叠成 batch，从而压缩搜索空间。

---

## 1. 先讲问题

我们现在研究的是 LLVM phase ordering 问题。传统做法会把优化过程看成：

```
当前 IR state
  -> 选择下一个 pass
  -> 得到新 IR
  -> 再选择下一个 pass
  -> ...
```

如果当前有 $n$ 个 active passes，最朴素的顺序空间就是：

$$n!$$

这个空间很快爆炸。

但我们的观察是：很多 pass 的相对顺序其实没有意义。比如两个 pass 分别修改不同区域，或者虽然都修改 IR，但 `A;B` 和 `B;A` 最终得到同一个 canonical IR，那么我们就不应该把 `A before B` 和 `B before A` 当成两个不同搜索分支。

导师最早的核心想法也是这个：**如果几个优化之间没有 dependency，就不应该让 ML 或搜索算法猜"下一个做哪个"，而应该把它们一起做；只有真的有依赖或冲突的地方，才需要搜索或选择。**

---

## 2. 我们的方法一句话

我们的方法可以概括成一句话：

> **在每个 IR state 上，先动态判断 active passes 之间哪些可交换，哪些顺序敏感；把可交换的 pass 合成 certified batch，只在剩下的 conflict choices 上搜索。**

也就是说，传统搜索是：

```
state -> pass -> state -> pass -> ...
```

我们变成：

```
state -> certified batch -> state -> certified batch -> ...
```

这里 batch 不是随便拼出来的，而是有证据的。

---

## 3. 每个 state 上具体怎么做

假设当前 state 是 $S$，系统做四步。

### 第一步：找 active passes

我们对当前 IR 运行 pass profiling，判断哪些 pass 当前会改变 IR：

- **active pass**：执行后 IR 改变
- **dormant pass**：执行后 IR 不变

注意，pass 不会被永久消耗。每到一个新 state，我们都会重新分析所有 pass。所以如果某个 pass 在 $S_0$ 上 dormant，但在某个 batch 执行后在 $S_1$ 上 active，它会在 $S_1$ 被重新发现。

---

### 第二步：做 pairwise commutation test

对每一对 active pass $(A, B)$，我们比较：

```
A;B(S)
B;A(S)
```

如果两个结果的 canonical IR hash 一样，就说明：

> $A$ 和 $B$ 在当前 state 上 **commute**（可交换）。

如果不一样，就标成：

> **order-sensitive**

如果失败或工具无法判断，就保守标成：

> **unknown / conflict**

注意：这里 objective（比如 IR instruction count）只用于最后评价，不用于证明可交换。可交换证据来自 **IR-level certificate**。我们的报告也明确写了 objective 不作为 commutation proof。

---

### 第三步：构造 conflict graph

我们把 active passes 当成节点：

- 如果两个 pass **可交换**，就不加冲突边。
- 如果它们 **order-sensitive**、**unknown**、**validation failed**，就加冲突边。

然后这个图会分成一些 component。

直观理解：

> 没有冲突边的一组 pass → 放进同一个 batch
> 有冲突的一组 pass → 需要保留不同选择

**举例**：当前有 $10$ 个 active passes：`{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}`

如果 `1、2、3` 之间互相冲突，而 `4–10` 都和它们可交换，那么我们**不会**枚举 $10! = 3,628,800$ 种顺序。

我们只保留：

```
batch A = {1, 4, 5, 6, 7, 8, 9, 10}
batch B = {2, 4, 5, 6, 7, 8, 9, 10}
batch C = {3, 4, 5, 6, 7, 8, 9, 10}
```

也就是：**3 个 batch choices**。

这就是最核心的搜索空间压缩。

---

### 第四步：验证 batch

pairwise commute 还不够保险，所以对生成的 batch，我们会做 **permutation validation**。

例如 batch 是 `{A, B, C}`，我们检查不同排列：

```
A B C
A C B
B A C
B C A
C A B
C B A
```

如果所有 tested permutations 都得到同一个 canonical IR hash，我们把它标成：

```
all_permutations_same
certified_batch
strong certificate
```

只有这种 **strong-certified batch** 才能作为 hard fold 来执行。

- **sampled batch**：只是 heuristic，不默认执行
- **rejected、failed、unvalidated batch**：都不能执行

---

## 4. 举一个具体例子

以 `n-body` 为例，root state 上有 $9$ 个 active passes。

如果按传统 pass ordering 看，局部顺序空间是：

$$9! = 362,880$$

我们先测 active pass pairs，一共有：

$$C(9,2) = 36 \text{ 对}$$

其中有一部分 pass pairs 最后被证明 commute，一部分是 order-sensitive。然后系统基于这些关系构造 batch，最后 root state 只剩下 **5 个 batch candidates**。

也就是说，在这个 state 上，局部候选空间从：

> **362,880 个顺序 → 5 个 batch choices**

局部压缩比例大约是：

> **72,576 倍**

更关键的是，这 5 个 batch 不是 heuristic，它们都有 **validation evidence**。

---

## 5. 当前实验数据怎么支撑这个说法

我们在 5 个程序上跑了 exact r4，全部 exact complete：

| 指标 | 数值 |
|------|------|
| programs | 5 |
| total states | 807 |
| total transitions | 936 |
| selected path steps | 16 |

在这些 reached states 上，我们一共测试了 **8068** 个 active pass pairs：

| 类别 | 数量 |
|------|------|
| commute pairs | 5321 |
| order-sensitive pairs | 2688 |
| unknown | 0 |

也就是说，很多 pair 确实是可交换的，但也有大量 pair 是顺序敏感的。我们的系统没有把所有 pass 粗暴合并，而是区分了两类。

更重要的是，所有 executed batches 都是 **strong certificates**：

| 类别 | 数量 |
|------|------|
| executed batches | 936 |
| strong certs | 936 |
| weak / rejected / failed / unknown | 0 |

并且：

> **dropped active passes = 0**

这说明我们不是靠丢 pass 来缩小空间，而是把 active pass 都 accounted for，要么进入 certified batch，要么被标记为 heuristic / conflict / terminal。

---

## 6. 压缩效果有多大？

在 exact r4 的 reduction summary 里，每个程序都有明显的 local reduction。

最大局部压缩（数量级）：

| 程序 | 压缩倍数（对数） |
|------|------------------|
| fannkuch | $10^{4.71}$ |
| n-body | $10^{4.86}$ |
| nsieve-bits | $10^{6.56}$ |
| partialsums | $10^{7.60}$ |
| puzzle | $10^{7.30}$ |

也就是说，在某些 state 上，原始局部顺序空间被压缩了 **4 到 7 个数量级**。

这就是我们论文想表达的核心：

> Phase ordering 空间巨大，但很多 ordering choices 是假的。我们用 state-local evidence 折叠这些假的选择，只搜索真正需要保留的 conflict choices。

---

## 7. 搜索怎么继续走？

压缩完之后，我们不是停止，而是在 **batch-state graph** 上搜索。

传统是：

```
S0 --pass--> S1 --pass--> S2
```

我们是：

```
S0 --certified batch--> S1 --certified batch--> S2
```

每到一个新 state，都会重新做：

1. active pass profiling
2. pairwise commutation test
3. batch construction
4. batch validation

这样后续被 enable 的 pass 也会被发现。

比如 `n-body` 之前 `max_rounds=2` 时，batch optimizer 结果是 223，落后 greedy 的 212。但继续 exact 到 r4 后，batch optimizer 到了 211，超过 greedy，也超过 random best。这说明问题不是 batch 方法错，而是必须让 **state-aware iterative loop** 继续展开。

---

## 8. exact 很贵，所以我们还有 budgeted mode

Exact mode 是 reference，它完整展开 certified batch-state graph，但成本高。

所以我们实现了 **budgeted mode**，用 beam search 在压缩后的 batch-state graph 上搜索。

结果很好：在这 5 个程序上，budgeted mode **全部匹配** exact r4 的最终 IR instruction count：

| 指标 | 数值 |
|------|------|
| programs matching exact | 5 / 5 |
| average gap to exact | 0 |
| max gap to exact | 0 |

同时它平均减少：

- **64.06% states**
- **53.03% time**

相对 baseline 的结果：

| 对比 | 结果 |
|------|------|
| vs greedy | 2 wins / 3 ties / 0 losses |
| vs random best | 2 wins / 2 ties / 1 loss |
| vs config order once | 3 wins / 2 ties / 0 losses |

所以我们的最终系统是两层：

```
Exact mode:
  小程序 / reference / 证明压缩空间确实有效

Budgeted mode:
  真实程序 / scalable search / 用更少 states 接近 exact
```

---

## 9. 总结一句话

> 我们不是直接在 pass sequence 空间里搜索。我们先在每个 IR state 上动态证明哪些 pass 顺序可交换，把这些 pass 合成 certified batch，从而把 $n!$ 的局部顺序空间压缩成少量 batch choices；然后在压缩后的 batch-state graph 上继续搜索。实验中，所有 executed batches 都有 strong certificate，dropped active passes 为 0，局部搜索空间最高压缩到约 $10^{7.6}$ 倍；budgeted search 还能用更少 states 和时间匹配 exact search 的最终结果。

---

## 附录：一版更口语化的 2 分钟讲法

> 我们现在的核心不是"猜下一个 pass"，而是先判断这个选择到底有没有意义。比如当前 IR 上有 9 个 active passes，传统做法会认为有 $9!$ 种顺序。但我们会逐对测试 pass：如果 `A;B` 和 `B;A` 得到同一个 canonical IR，就说明 A 和 B 的相对顺序没必要搜。我们把所有这种可交换关系合起来，构造 conflict graph。没有冲突的 pass 被放到同一个 batch，有冲突的地方才保留不同选择。这样一个 state 上原本几十万甚至上千万的局部顺序选择，可能只剩几个 batch choices。
>
> 这些 batch 不是随便合并的。我们会验证 batch 内不同排列是否产生同一个 IR。如果所有排列都一样，才标成 certified batch，才能执行。sampled、failed、unknown 都不会 hard prune。
>
> 然后优化过程不是只做一轮。执行 batch 后得到新 IR state，我们重新扫描所有 pass、重新测 pair relation、重新生成 batch。所以后面被 enable 的 pass 会在新 state 里被发现。
>
> 当前 exact r4 在 5 个程序上跑完了 807 个 states、936 条 batch transitions。所有 executed batches 都是 strong certificates，没有 dropped active pass。最大的局部搜索空间压缩大概达到 $10^{7.6}$ 倍。budgeted mode 在这 5 个程序上全部匹配 exact r4，但平均少跑 64% states、少花 53% 时间。所以我们的贡献可以说是：**用 state-local commutation evidence，把 phase ordering 从暴力 sequence search 变成 certified batch-state search。**


> 当前发现的问题和可能得解决方式：1，假冲突（False Conflicts / 假阴性）： 两个 Pass 序列生成的 IR 在语义和结构上完全一致，但因为 SSA 虚拟寄存器编号不同（如 %1 vs %2）、基本块命名不同、或者 Metadata（如 DbgInfo, TBAA 标签）的排布顺序轻微差异，导致 Hash 不一致。如果你用的 Canonicalization 不够深度，就会把大量原本 commute 的 Pass 判定为 conflict，导致搜索空间压缩率大打折扣。

假交换（False Commutativity / 假阳性）： 如果 Canonicalization 剥离了过多的附带信息（例如丢弃了某些看似冗余但在后续 Pass 中会触发不同启发式分支的属性），可能会导致你在 Step 4 验证失败，或者生成错误的 Batch。
2，Wall-Clock Time 悖论：编译开销（Overhead）能否真正被赚回来？这是系统的实用性（Pragmatism）面临的最大挑战。审稿人的算账逻辑： 假设当前状态有 $N=20$ 个 Active passes，计算 pairwise 需要约 $20 \times 19 / 2 = 190$ 次完整编译（或 Pass 运行）。即使压缩了搜索树的深度，但在每一个搜索节点（State）都额外引入了 $O(N^2)$ 的动态测试开销。如果搜索一个序列需要走 10 个步长，仅构建图就要跑 ~1900 次 Pass。用这些额外的开销时间，传统的 RL 或遗传算法是否已经用暴力搜索跑出了更好的结果？建议（实验防守策略）：极高优先级的指标——Memoization Hit Rate（缓存命中率）： 你在第5节提到了“跨 State/跨程序有很多关系可以复用”。你必须在实验中用强力数据支撑这一点！ 如果你能证明：在编译某个特定程序时，90% 的 Pass 交换性关系在经历了几轮迭代后都被 cached 了，实际的 $O(N^2)$ 开销在搜索中后期骤降为 $O(1)$，这个故事就彻底立住脚了。两类对比实验：同等 Search Steps 下： 证明你的最终代码优化效果（运行耗时/指令数）远超 baseline。同等 Wall-Clock Time 下： 给你和 Baseline（如 OpenTuner, CompilerGym, Standard GA）完全一样的搜索时间预算（例如限制只允许搜索 2 个小时），证明在相同的时间开销下，你的方法找到了更好/更稳定的 Pass 组合。

3，为什么 Step 4 (Verify Batch) 必须存在？——理论解释需更深入
你在 Step 4 提到“pairwise commute 并不确定就能生成 multi-pass batch”，这一点你处理积极且真实，没有掩盖问题，非常赞。但审稿人会问：为什么在代数意义上， pairwise 交换不能传递到多 pass 组合？

理论深挖： 在现代编译器中，Pass 不仅修改 IR，还会修改 LLVM Analysis Manager（分析缓存池） 的状态，或者是 Pass 内部自带了特定的启发式阈值/随机种子。例如，Pass A 和 B 在单独面对同一个 IR 时都不会触发深度分析，但当 A 和 B 连用时，累积的 IR 结构变化触及了某个局部优化门槛，使得第三个 Pass C 的行为发生了改变。

建议： 建议把 Step 4 从一个“工程补救措施”提升为一个“理论观察（Empirical Observation/Insight）”：明确指出在 LLVM 体系中，由纯 IR 变换构成的代数性质与带有分析依赖（Analysis Preservation/Invalidation）的实际 Pass 执行体系之间存在 Gap，因此 Certified Batch 的概念不仅是个优化，也是保真（Fidelity）的必需品。这会显示你对 LLVM 底层设计有着极其深刻的理解。

4，“Active Passes” 的选取与修剪（Pruning）
你在第3节指出“先跑一次 Pass 发现 IR 没变就是 Inactive”。

注意点： 很多 LLVM Pass 在默认 Pipeline 里是配套出现的（例如 loop-simplify -> licm -> loop-unroll）。有些 Pass 在当前 IR 上看似是 Inactive 的（跑完 IR hash 没变），但它只是对 IR 做了一层规范化（Canonicalize / Normalize），为下一个 Pass 能够生效打下了基础。

如果你的“选 Active Pass”过早地将这一类没有直接产生明显收益、但处于“使能（Enabling）”位置的 Pass 剔除，可能会导致你在探索阶段错过重要的全局最佳优化路径。

建议： 解释清楚你的 Active Pass 修剪策略是不是仅限当前 step 的“直接发生变化”，还是会保留部分特定的 Enabling Pass，或者证明这种过早剔除不会导致陷入劣质局部最优（Local Optima）。

5，Batch Validation 的复杂度(TheO(K!)Explosion)

问题所在：在“第四步”中，即使你构建了conflictgraph并提取了
independent sets 作为 candidate batches,由于pairwise
commute不能推导出batch commute，你仍然需要做
permutation validation。即便采用了 "Bounded batch size"来控制
O(K!)的爆炸，这种暴力全排列验证依然不够优雅。

优化建议：

能不能把validation过程从全排列弱化为拓扑排序验证？或
者采用增量式的验证：先验证SA，B，如果通过，再将
C加入验证C(AB)和AB(C)，而不是直接铺开算。
l0
启发式截断：当clique很大时，用什么指标来指导
"Bounded batchsize"的截断？是随机选K个，还是根据
历史数据选那些对性能提升贡献最大的passes组成batch?

6，这个Metadata（元数据）是LLVM IR中的一个核心概念，也是在做IR层面字符串哈希(Hash）对比时，极其容易踩坑的隐藏变量。
在你的论文设计中，第三步和第四步的灵魂在于：判断执行序列A→B和B→A产生的CanonicalIR是否等价。如果等价，说明这两个Pass可以交
换，从而压缩搜索空间。
问题在于，许多Pass 对待Metadata 的态度非常“随意”:
假设有两个执行路径产生了逻辑上完全一模一样的指令流：
1. 路径1:某个 Pass 比较保守，在变换指令时顺手丢弃了 TBAA 和 Debug Info。
2.路径2:另一个Pass比较精细，保留了所有的TBAA信息。
如果你此时直接对两份IR文本进行字符串Hash，由于一行带了！tbaa一行没带，Hash值会截然不同。
你的判定系统就会误以为这是两个冲突的(Conflict）状态，把本该合并的搜索空间分支强行拆开。这被称为FalseNegative（假阴性冲突），它会大幅降低你
算法的“压缩率”。

7，“为什么我们必须采用在每个 State 上进行 On-the-fly 的动态交换律测试，而不是依赖离线的静态分析特征？
因为编译 Pass 之间的交互存在深度的迟滞效应（Enabling effects）。即便两个 Pass 在当前代码切片上没有直接的 def-use 冲突，Pass A 的变换可能会改变 CFG 拓扑或暴露出新的优化模式，使得 Pass B 的优化行为发生根本性改变。静态分析受限于其保守性，无法精确预测这种动态衍生出的优化机会，因此必须依赖基于真实 IR State 演化的动态验证来构建 Conflict Graph。”既然静态白名单走不通，还能怎么优化 $O(N^2)$？既然证实了“静态证明它们绝对独立”是个伪命题，我们要降低第二步（pairwise commute test）的开销，可以把思路反过来或者侧推：从白名单转向“黑名单”（Fast Rejection）： 我们很难证明两个 Pass 绝对 commute，但可以通过简单的 Pass Manager 依赖关系快速判断它们绝对 conflict。比如 Pass A 明确宣告它会 Invalidate DominatorTree，而 Pass B 强依赖 DominatorTree 且不具备动态修复能力，那么 A -> B 和 B -> A 的结果大概率不同。遇到这种对子，直接标记为 Conflict，省去跑 IR 测试的开销。数据驱动的“软隔离”： 在 Budgeted mode 下，如果历史搜索记录表明 LoopUnroll 和 GVN 在前 100 个 state 里测试了都是冲突的，那么在接下来的搜索中，对于这对 combination 赋予极低的探索优先级，直接将其视为 Conflict 断开 Batch。
