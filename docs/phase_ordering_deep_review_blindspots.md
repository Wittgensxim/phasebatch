# 深度审稿意见：Phase-Ordering 搜索空间压缩的盲点、修正与升华

---

## 任务 1：关于「Enabling 效应使静态证明不冲突不可能」——裁判结论

**结论：方向正确、可以作为攻击点，但你现在的措辞形式是错的，原样写进 paper 会被一个懂理论的 reviewer 一句话打回。** 需要把它从一个「不可能性断言」改写成一个「不对称性（asymmetry）」论证。

### 你现在的命题有两个层次，你混用了

#### (a) 强形式：「静态无法证明两个 pass 一定 commute」——这是平凡真

Pass 是图灵完备的变换，commutativity 是一个非平凡语义等价性质，由 Rice 定理直接不可判定。Reviewer 会说：

> "This is trivially true by Rice's theorem and therefore not a contribution."

你如果把论文的核心攻击点建在这上面，等于把 novelty 建在一个 textbook 事实上。

#### (b) 弱形式（你真正想说的）：「对真实 LLVM pass，静态分析给不出有用精度的 commutativity 判定」——这是经验真、且有价值

但它需要**机制解释**，而不是「不可能」三个字。

### 正确的、能站住的框架：conflict 与 commute 的不对称性

这是我建议你写进 paper 的核心 insight，它同时替你把攻击点变 solid、又顺手论证了任务 4 里那个 fast-rejection 优化：

> **冲突是存在量词，可换是全称量词。**
>
> - 判定「A 与 B 冲突」只需要**举出一个**下游行为差异 → 存在量词 → 静态分析可以**保守地过近似**（over-approximate），这正是 LLVM PassManager 的 analysis preserve/invalidate 依赖模型在做的事。
> - 判定「A 与 B 可换」需要**证明所有**下游行为都不变 → 全称量词 → 而且它和下游 pass 的**启发式阈值 / cost model** 纠缠在一起（enabling effect 是二阶效应：A 改变的不是 B 读的数据，而是 B 的启发式所面对的**输入分布**）。

所以正确的论文句子不是 "static proof is impossible"，而是：

> *"Static analysis can **conservatively over-approximate conflict**, but it cannot **soundly certify commutativity with useful precision**, because commutativity is a universally-quantified property entangled with downstream heuristic-sensitive enabling effects. We therefore certify commutativity **empirically and state-locally**."*

这个版本 reviewer 无法反驳，而且它自然导出：**静态方法只在「反方向」（fast conflict rejection）有用** —— 这恰好是你任务 4 该用的黑名单优化。任务 1 的正确表述和任务 4 的优化其实是同一枚硬币的两面。

### 一个你必须自己先承认、否则被反杀的点

你的动态测试**也没有证明 commutativity** —— 它证明的是"在这一个具体 IR state $S$、这一套 canonicalization 下，`A;B` 与 `B;A` 落到同一个 canonical IR"。这是一个**单点经验观测**，不是代数意义的可换性。你现在用 "certificate / proof" 这个词，负载过重。Reviewer 会问：

> "Certificate of what, exactly?"

**建议：** 明确定义为 **state-local empirical commutativity**，并诚实声明其 scope。这不削弱贡献 —— 反而，把它和真正的代数可换性区分开，正是你比「静态白名单」更诚实的地方。

### 文献锚点（写 related work / 加固 motivation）

- **Whitfield & Soffa, TOPLAS 1997** — 《An approach for exploring code improving transformations》：最早把 transformation 之间的 **enabling/disabling interaction** 形式化，并指出它是 specification-dependent、context-sensitive 的。这是你 "enabling 效应" 论点的经典出处，一定要引，能立刻让 reviewer 觉得你知道自己在说什么。
- **Newman's Lemma / 抽象重写系统的 confluence**（Baader & Nipkow, *Term Rewriting and All That*）—— 见任务 2/4，这是把你 Step 4 从「工程补丁」升格成「理论必然」的钥匙。

---

## 任务 2：审稿人视角的致命盲点

> 避开你已经发现的 metadata / $O(N^2)$ / $O(K!)$

以下六个盲点按杀伤力排序。**第 1、2、3 条我认为是能直接 kill paper 的等级。**

### 盲点 1（最致命）：你假设 pass 是 IR → IR 的确定性纯函数，但它不是

你的整个 "commute" 定义建立在 `A;B(S)` 与 `B;A(S)` 的 hash 相等上 —— 这隐含假设「同样的输入跑 A 一定得到同样的输出」。但 LLVM 里有一批关键 pass **本身不是 IR 的确定性函数**：

- 历史上 GVN/NewGVN、SLP vectorizer、部分 pass 的选择依赖容器迭代顺序（`DenseMap` / `SmallPtrSet` 的迭代顺序曾与指针地址相关）；
- inliner、unroller 的 cost model 在 tie-break 时可能对相同 IR 做不同选择；
- 带 cost model 的 pass 对「等价代价」的候选做非确定选择。

**后果：** 如果 `A;B(S)` 自己跑两遍都不等于自己，"commute" 这个谓词根本 **ill-defined**。你的 807 states / 8068 pairs 的干净数据，可能只是因为你的 5 个 toy program 恰好没踩到非确定 pass。

> ⚠️ **这是一个 reviewer 会直接要求你做的 ablation：先跑 self-consistency test（每个方向跑两遍），报告有多少 pass 不是纯函数。** 不做，这篇就有一个 open 的 soundness 洞。

### 盲点 2（真正的深水区 — Analysis pass）：AnalysisManager 的状态是「隐藏的 state 的一部分」，你的沙箱 fidelity 不足

你在做 pairwise 测试时，几乎一定是把 A、B **在 fresh / cold 的 analysis manager 上**孤立地跑。但在真实 pipeline 里，B 跑的时候 analysis cache 是**热的**（前序 pass 留下的 DominatorTree / LoopInfo / AA 结果还在缓存且 valid）。而：

> 一个 pass 的行为可以依赖于「某个 analysis 是 cached-and-valid 还是被重算」—— 因为**重算可能给出一个不同但同样合法的结果**（例如一个不同但合法的 loop rotation、一个在不同起点下不同的 BlockFrequency 估计），从而触发不同的 cost-model 分支。

于是：**你在冷缓存下测得的 commute，在真实热缓存的 pipeline 里未必成立。** 你的 commute 关系其实不是 $S$ 的函数，而是 $(S, \text{analysis-manager-state})$ 的函数，而你的模型把后者完全丢掉了。这是一个**沙箱与真实流水线之间的保真 gap**，而且它专门在 analysis-heavy 的 pass（几乎所有循环 / 别名相关 pass）上发作。

这一条正好把你笔记里第 3 点（为什么 pairwise 不能传递到 batch）**升级**：根因不只是 "IR 累积效应"，而是 "analysis provenance 不在你的 state 表示里"。这是能让 reviewer 点头的深度。

### 盲点 3（对整个 premise 最狠的攻击）：你压缩掉的是搜索空间里不值钱的那部分

commute 的 pass 大多是 scalar cleanup / canonicalization 类（instcombine 的子操作、simplifycfg、reassociate 之间的很多组合）。而**真正决定性能的 loop 变换（unroll / vectorize / unswitch / rotate / licm）几乎永远 order-sensitive**：

- 部分 / 运行时 unroll 会产生 **remainder loop**，其结构随顺序而变 → hash 必然不等 → 判为 order-sensitive；
- unroll-then-vectorize vs vectorize-then-unroll 是**真的**不等价（且 cost model 决策相互依赖）。

**后果：** 你 $10^{7.6}$ 的压缩很可能集中在 cleanup 段，而 payoff 最高的 loop 段压缩率接近 0。一个尖锐的 reviewer 会写：

> *"The authors compressed the boring part of the search space; on the transformations that actually move the objective, the conflict graph is nearly complete."*

**防守方式（你必须提前做）：** 把 commute / order-sensitive 的 pair **按 pass 类别 breakdown**（loop vs scalar vs CFG），并**报告压缩发生在哪些 state 深度**。如果你能展示"即使在 loop 段，batch 折叠仍消掉了 X% 的冗余顺序"，这条攻击就化解了；如果不能，至少要主动承认 scope。**这是最需要新实验的一条。**

### 盲点 4：你把阶乘从「搜索」搬进了「验证」，而救你出 $K!$ 的 cache 又把你打回静态白名单的不可靠性

两个子问题咬在一起：

- **strong certificate 要求测全部 $K!$ 个排列**。$K=8$ 就是 40320 次编译 / 每 batch / 每 state。所以要么你的 batch 很小（$K \le 4 \sim 5$，那你要报告 batch size 分布，别只报 "936 strong certs"），要么你在 sample（那 certificate 就弱化了）。你没消掉阶乘，只是搬了家。
- 你（和上一位 AI）用**跨 state / 跨 program 的 memoization** 来摊薄 $O(N^2)$。**但可换性是 state-dependent 的 —— 这正是你全篇论文的立论！** 于是把 "A commute B" 从 state $S$ 复用到 $S'$ 是**不 sound 的**：你如果按 $(A, B)$ 做 cache key，就**重新引入了你花整篇论文攻击的静态白名单的假阳性**。

这是一个漂亮的**内在矛盾**，reviewer 一定会挑：*救你性能的 cache，和逼你上动态测试的 state-dependence，是直接冲突的。*

**出路（写进 paper 会加分）：** 只 cache **单调安全**的东西 —— (i) conflict 方向（黑名单，见任务 4）；(ii) per-pass 的 analysis preserve/invalidate 指纹（这个几乎与 IR 无关，才是真正可复用的）；而 commute 只在「附带足够 local context 指纹」时才 cache。把这一点讲清楚，反而变成一个 contribution。

### 盲点 5：false-commutativity 不是「生成错 batch」，而是「静默剪掉全局最优」，而你没有 oracle 能发现

你的压缩号称 optimality-preserving，前提是「折叠的确实是等价 ordering」。可一旦有假阳性（canonicalization 剥离过度，或盲点 1 的非确定性），你会把两个**下游不等价**的 ordering 折进一个 batch，只执行其一，于是**全局最优被无声剪掉**。而你现在的验证只对比 IR-hash，没有任何独立 oracle 去抓这种 false fold。你 5/5 exactly match exact 的漂亮数据，恰恰意味着**你还没在会暴露它的地方压过力**。

Reviewer：

> *"Your compression is only optimality-preserving up to the soundness of your equivalence oracle, which you never independently validate."*

**建议：** 加一个 **end-to-end sanity**：对一小批 state，把被折叠掉的 ordering 也真跑到底，确认最终 objective 不因折叠而变差。哪怕小样本，也堵住了这个洞。

### 盲点 6：「active pass 集合」本身是 order/batch 依赖的，不是 state $S$ 的良定义属性

你 Step 1 假设可以在 $S$ 上独立地把 pass 分成 active/dormant。但：

- InstCombine / SROA / GVN 本身是**跑到 fixpoint** 的；一个 pass 在 $S$ 上 "dormant" 可能只是因为**同 batch 里前一个 pass 已经替它把活干了** → activity 依赖 batch 内顺序，与你「独立分类」的假设矛盾。
- 还有你笔记第 4 点自己提的 enabling pass（loop-simplify / lcssa 这类"看似 inactive 但为后续 pass 铺路"）。把它们过早剔出 active 集，会在探索期错过全局最优路径。

**建议：** 把 activity 定义为 **相对于当前 batch 构造过程的**属性，并说明 enabling pass 的保留策略 —— 否则 Step 1 的「独立 profiling」是个未证明的假设。

---

## 任务 3：Storyline 与 Motivation 升华（顶会 Intro 骨架）

### 一句话核心 Story

> **相位排序空间里绝大多数的顺序决策是语义上空洞的；而这种冗余是静态不可见的、状态局部的、必须动态确定的性质。我们把 phase ordering 从「序列搜索」变成「对等价类取商之后的搜索」—— 动态地 certify 掉那些无意义的顺序决策，并证明 certify 的开销被它消灭的搜索开销所主导（dominated）。**

这个 framing 的三个杀招：

1. **"equivalence-quotient search"** —— 一个数学上干净、reviewer 熟悉的概念；
2. **"redundancy is statically invisible"** —— 直接把你和静态方法、也和纯 ML policy 方法区分开；
3. **"cost is dominated by the search it eliminates"** —— 提前回应了 Wall-Clock 悖论。

### 六段式 Intro 骨架

#### 1. 钩子 / 问题

Phase ordering 是 $n!$，而过去十年的主流（Autophase、CompilerGym、POSET-RL、MLGO）全都在学一个**「选下一个 pass」的 policy**。但这个框架 bake 进了一个**未经检验的假设：每一个顺序决策都是有意义的**。

#### 2. 反转（The Twist）

大量顺序决策是**语义空洞**的 —— `A;B` 与 `B;A` 落到同一 IR，对它们搜索是纯浪费。这在数学上是搜索空间上的一个**商（quotient）**，理应被 factor out。

> **没有人在做「压缩空间本身并带 soundness 保证」这件事 —— 大家都在压缩后不存在的、或未压缩的空间上学 policy。这就是 whitespace。**

#### 3. 为什么不能静态（树敌）

自然的想法 —— 静态声明哪些 pass 独立 —— **根本性地失败**，因为**冲突是存在量词（可静态过近似），可换是全称量词且与下游 enabling 效应纠缠（不可静态精确 certify）**（把任务 1 的 asymmetry 放这）。静态方法只能保守 → 认出极少 commute → 抓不到冗余。

#### 4. 我们的方法（下注）

我们付出代价**动态观测**每个 state 上的可换性，构造一个 **certified 等价商（batch）**，并在商图上搜索。**诚实地摊牌开销：是的，每个 state 有 $O(N^2)$ 探测。**（主动承认成本，反而取信 reviewer。）

#### 5. 为什么这注划算（Resolution）——三点闭环

| 论点 | 内容 |
|------|------|
| (a) 商巨大 | 局部压缩最高 $10^{7.6}$，dropped passes = 0（不是靠丢 pass） |
| (b) 探测开销被摊薄 | cache 单调安全的 preservation 指纹，且被它**消灭的阶乘搜索所主导** |
| (c) 因为是 certified | 压缩是 **sound / optimality-preserving** 的，不同于 heuristic pruning |

> **一句话交易：用近线性的 overhead，删掉阶乘级的冗余。**

#### 6. Contributions

- (i) 把 phase ordering **重新形式化**为 certified equivalence-quotient search；
- (ii) state-local **动态可换性 certification** 与 batch 抽象，及其与 Newman/confluence 的关系；
- (iii) **exact mode 作 ground-truth + budgeted beam mode** 的两层系统；
- (iv) 大规模经验刻画：commute vs order-sensitive 比例、压缩量级、budgeted 用少 64% states 匹配 exact。

### 两个能拔高段位的定位动作

**1. 对比 Equality Saturation（Tate et al., POPL 2009）/ e-graph / egg**

那是学术界对 phase ordering 的经典「取商」答案，但它用 e-graph 对**值等价**取商，**要求 pass 是可表达为 rewrite rule 的白盒，难以 scale 到完整 LLVM pipeline**。你的差异化一句话：

> *"Equality saturation quotients by value-equivalence over a white-box rule set; we quotient by **observed** commutativity over **black-box** production passes."*

引它 + 划清界限，立刻显出你读过 canon。

**2. 强调正交可组合性**

> "我们不与 learned policy 竞争，我们缩小它们赖以工作的空间 —— 任何 RL/GA 都能跑在我们的商图上。"

这把潜在竞争者变成潜在 downstream 用户，是 accept 友好的姿态。

### 标题候选

1. *Certified Batches: Turning Phase Ordering into Equivalence-Quotient Search*
2. *Don't Search What Doesn't Matter: State-Local Commutativity Certification for the Phase-Ordering Problem*

---

## 任务 4：Step 4（验证 batch）与 Step 5（生成 state）的工程 Hack

按杀伤力排序。头三个是「不做就慢到不可用 / 或压缩率虚低」的必做项。

### 【必做 1】用 `FunctionComparator` 做结构等价，别做文本 hash —— 一举解决你的 false-negative

你笔记里 metadata / SSA 编号 / block 命名导致的假冲突，**LLVM 里已经有现成引擎**：`llvm/Transforms/Utils/FunctionComparator.h`（MergeFunctions 背后的那个）。

`FunctionComparator::compare()` 定义了一个**忽略 value name、规范化 operand 结构**的全序/等价，内部用 `GlobalNumberState` 给虚拟寄存器/全局做同构编号。**两份 IR 只要在同构意义下相等，它就返回 0，与 `%1` vs `%2`、block 名无关。** 这直接消灭你最担心的假阴性。

**对 metadata：在比较前显式规范化** —— 按你的 objective 决定每类 metadata 是否语义相关：

- 指令数目标下 `!dbg` 无关 → strip；
- `!tbaa` 影响后续 AA 派生的优化 → 要么保留并 canonical 排序，要么把它当「语义相关」纳入等价。

> **关键是显式决策，而不是让原始文本 hash 替你隐式决策。**

### 【必做 2】进程内 `CloneFunction` 沙箱，别 shell 出 `opt`

每对测试 fork 一个 `opt` 子进程 = 进程启动 + IR parse + IR print，这三样会**主导**你的 wall-clock，而且正是 reviewer 算账（~1900 次 pass）时假设的昂贵模型。改成：

- 把 `Module` 常驻内存，`CloneFunction`（**函数粒度**，大多数 pass 是 function pass，别整 module clone）到一个临时函数上，用**同一个常驻的 `PassBuilder` + 新 PM** 在进程内跑，跑完丢弃 clone。比文本往返快 **10–100×**。
- pairwise 测试是 embarrassingly parallel：线程池 + 每线程独立 `LLVMContext`，把 $O(N^2)$ 的 wall-clock 变 $O(N^2 / \text{cores})$。

### 【必做 3】两级等价：廉价指纹先过滤，`FunctionComparator` 只在疑似相等时确认

绝大多数 order-sensitive 对的两份 IR "明显不同"。先算一个**顺序无关的廉价指纹**：

- opcode 多重集 + CFG 形状描述子（每个 block 的 `(in-deg, out-deg, terminator-kind)` 排序后的列表）。

指纹不等 → 立即判 order-sensitive，$O(\text{size})$；只有指纹相等才上 `FunctionComparator` 做 $O(\text{size} \cdot \log)$ 的确认。把昂贵比较留给真正可能相等的少数。

### 【高价值】黑名单 fast-rejection：用 analysis preserve/invalidate 免跑大多数对

这就是任务 1 asymmetry 的工程兑现。**必要条件**：A、B 能 commute，只有在「谁都不破坏对方所需的 analysis」时才**可能**成立。所以：

- 先**探测**每个 pass 的 "preserve / invalidate 指纹" —— 在新 PM 里 pass 运行时返回 `PreservedAnalyses`；你可以对每个 pass 跑一次、diff analysis manager，得到它保留/失效了哪些 analysis。**这个指纹几乎与 IR 无关，是真正 sound 且高度可复用的 cache**（直击盲点 4：cache 这个，别 cache commute 判决）。
- 若 A 失效了 B 强依赖的 analysis（如 A invalidate DominatorTree、B 强依赖它且无动态修复）→ **直接标 conflict，省掉 IR 测试**。这是对 $O(N^2)$ 常数的巨大削减，而且是 sound 的（只砍掉「不可能 commute」的对）。

### 【解 $K!$ 爆炸】相邻对换 + confluence，替代全排列 —— 但要诚实标注前提

不要真跑 $K!$。用两个层次：

#### 1. 相邻对换生成 $S_K$（sound 的方向，但带前提）

对称群由相邻对换生成 —— 任意排列可由相邻交换到达。因此若在**固定参考序内、带完整上下文**地验证每个相邻位置的交换都保 IR-不变，则（在 local confluence 成立的前提下）全部排列等价。这把验证从 $O(K!)$ 降到 **$O(K^2)$ 量级的 in-context 相邻检查**。

> ⚠️ **但请诚实：这只在「pairwise-in-context 可换 ⇒ 菱形闭合（局部合流）」成立时才是 certificate**，而这需要终止性/无振荡 —— 这正是 **Newman's Lemma**（local confluence + termination ⇒ global confluence）。把这一句写进 paper，你的 Step 4 就从「工程补丁」升成「理论必然」（直接兑现你笔记第 3 点想要的「理论观察」层级）。

#### 2. 增量式 batch 生长（可用，标为 heuristic）

已 certify {A, B}，加入 C 时只测 C 与 batch 的若干 interleaving，而非全铺开。**明确它是 strong heuristic 而非 proof** —— 诚实标注 certificate 强度（strong / incremental / sampled）本身就是可信度加分。

> **一句话：相邻对换给你 sound-但需 confluence 前提的降阶；增量生长给你更快但更弱的证据；两者都比 $K!$ 好，且都要标清 guarantee 等级。**

### 【解盲点 1】确定性护栏

对被标记为非确定的 pass（或全体，先跑一轮筛），**每个方向跑两遍、要求自洽**才信任 commute 判决；固定随机种子、加确定性 flag。开销只加在真正非确定的少数 pass 上。

### 【Step 5：解盲点 2】带热缓存、按真实嵌套生成后继 state

执行选中的 batch 生成 $S_{i+1}$ 时，**不要用孤立冷缓存跑** —— 要走**真实 pipeline adaptor**（正确的 Loop / Function / CGSCC 嵌套 + 热 AnalysisManager），让你据以继续的 state 有保真度。然后对 $S_{i+1}$ 跑**一次规范化**（instnamer + metadata canonical 排序），让后续所有 hash 稳定。这样 Step 2 测的 commute 才和真实流水线一致，直接堵住盲点 2。

### 【复用】常驻 PassBuilder / 用新 PM 的 pipeline 字符串一进程跑组合

`PassBuilder` + analysis 注册只建一次；新 PM 的 `-p '...'` pipeline 字符串能在**一个进程内**跑任意 pass 组合，省掉反复初始化。

---

## 如果只让你带走三句话

| # | 要点 |
|---|------|
| **任务 1** | 别说「静态不可能」，说「**冲突是存在量词可过近似、可换是全称量词且与 enabling 纠缠不可精确 certify**」—— 这个 asymmetry 既是你的 motivation，也是你 fast-rejection 优化的依据。 |
| **任务 2** | 最可能 kill 你的三件事 —— **pass 不是确定性纯函数**、**analysis-manager 状态不在你的 state 表示里（沙箱保真 gap）**、**你可能只压缩了搜索空间里不值钱的 scalar 段**。前两个各补一个 ablation，第三个补一个 by-pass-category 的 breakdown。 |
| **任务 4** | `FunctionComparator` 杀 false-negative，进程内 `CloneFunction` 杀 per-test 开销，**相邻对换 + Newman's Lemma** 杀 $K!$ —— 并且把「Step 4 为什么必须存在」讲成 local-confluence 在非终止重写系统里不成立的理论后果。 |
