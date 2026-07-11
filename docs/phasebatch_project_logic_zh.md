# Phasebatch 项目完整逻辑说明

> 更新时间：2026-07-11
> 本文描述当前代码的实际行为。
> 当前推荐主线：`full pairwise + certified validation + rolling-exact + strict worker`。

## 1. 项目目标

Phasebatch 是一个 LLVM phase-ordering 研究原型。它围绕“当前 LLVM IR 状态”重复完成以下工作：

1. 找出当前真正有效的 active passes；
2. 测试 active pass 两两交换顺序后是否得到等价 IR；
3. 根据 pair 关系构造 conflict graph；
4. 从 conflict graph 中生成可同时执行的 batch candidates；
5. 验证 batch 的不同排列是否都得到等价 IR；
6. 只执行通过 correctness classifier 的 batch；
7. 把新 IR 作为新状态继续分析；
8. 在完整的两层局部窗口结束后保留最多五个多样化终态，并从这些终态继续滚动；
9. 重建并重放最终 pipeline；
10. 输出 CSV、Markdown 等证据并清理 `.ll` 文件。

当前目标函数只有 `ir-inst-count`，即 LLVM IR 指令数。它只用于搜索排序和最终状态选择，不能证明 pass 可交换，也不能替代 batch validation。

项目不声称找到了 LLVM 全局最优 pass 顺序。`rolling-exact` 表示 pair/batch 正确性证据和每个局部窗口都完整；若窗口边界候选超过 K=5，`global_search_complete=false` 会明确说明全局状态图发生了前沿剪枝。

## 2. 当前推荐主线

推荐配置是：

```text
batch_construction_mode = pairwise
pair_testing_mode = full
batch_validation_mode = auto
validate_batches = true
optimizer_mode = rolling-exact
rolling_window_depth = 2
rolling_frontier_width = 5
max_rolling_windows = 0
```

主线的保守规则是：

- 只有经过动态测试并得到 `final_commute` 的 pair 才被当作可交换；
- `final_order_sensitive` 一定是冲突；
- 未测试、失败、超时和比较器失败都得到 `final_unknown`；
- `final_unknown` 在 batch 构造时也按冲突处理；
- batch 没有得到执行许可时，优化器绝不执行它。
- `rolling_window_depth` 的主线默认值是 2；H=3 只用于深度消融；
- 窗口内部不使用 beam，也不使用 `max_batches_per_state` 截断；
- 只在第二层窗口边界按五个可解释桶保留 K=5；
- 达到 pair、candidate、validation、state 或正数 window 上限时，只能记为 incomplete；
- `max_rolling_windows=0` 表示持续到闭合，不表示零轮。

## 3. 三种图

项目中有三种容易混淆的图：

| 图 | 节点 | 边 | 用途 |
|---|---|---|---|
| pair conflict graph | 当前状态的 active pass | 不能视为 commute 的 pair | 生成 batch candidates |
| batch validation DAG | 已执行 pass 子集上的 IR 等价类 | 追加一个未执行 pass | 验证一个 batch 的全部排列 |
| optimizer state DAG | canonical LLVM IR 状态 | 执行一个允许执行的 batch | 多轮搜索和重复状态合并 |

它们的关系是：

```text
pair 测试
  -> conflict graph
  -> batch candidates
  -> batch validation DAG / 全排列
  -> correctness classifier
  -> optimizer state DAG
```

## 4. 总体流程

```text
输入 C/LL
  -> 准备 root IR
  -> 加载并验证 pass 配置
  -> 创建 S0000
  -> 单 pass profiling
  -> full pairwise AB/BA
  -> IR 等价性判定
  -> conflict graph
  -> maximal independent sets
  -> batch candidates
  -> exhaustive/DAG/bounded/sampled validation
  -> correctness classifier
  -> 执行安全 batch
  -> child state 去重并继续分析
  -> rolling-exact：完整看两层，边界保留最多五个 open 终态
  -> 从全部保留终态重新开始下一窗口
  -> no active / no executable / state graph closed 时结束
  -> 选择 final state
  -> 重建和 replay pipeline
  -> 写报告
  -> 删除 .ll 和空目录
```

下面按实际执行顺序逐步说明。

## 5. 步骤 0：命令和默认配置

端到端入口是 `optimize-batches`：

```powershell
python -m phasebatch optimize-batches `
  --input benchmarks/tiny/branch.c `
  --out outputs/example_branch `
  --passes configs/core_passes_v1.yaml `
  --mode rolling-exact `
  --rolling-window-depth 2 `
  --rolling-frontier-width 5 `
  --max-rolling-windows 0 `
  --objective ir-inst-count `
  --validate-batches `
  --batch-construction-mode pairwise `
  --pair-testing-mode full `
  --batch-validation-mode auto
```

主要默认值：

| 参数 | 默认值 |
|---|---:|
| `--mode` | `rolling-exact` |
| `--rolling-window-depth` | 3 |
| `--rolling-frontier-width` | 5 |
| `--max-rolling-windows` | 0（直到闭合） |
| `--max-rounds` | 5（仅 legacy exact/budgeted） |
| `--beam-width` | 8（仅 budgeted） |
| `--max-batches-per-state` | 20（仅 budgeted） |
| `--budgeted-validation-strategy` | `all` |
| `--max-component-size` | 14（覆盖完整 Core-v1 pass set） |
| `--max-batch-candidates` | 200 |
| `--max-states` | 2000 |
| `--pair-testing-mode` | `full` |
| `--batch-construction-mode` | `pairwise` |
| `--batch-validation-mode` | `auto` |
| `--max-permutation-factorial` | 120 |
| `--max-validation-dag-nodes` | 5000 |
| `--max-validation-dag-edges` | 20000 |
| `--verify-final-pipeline` | true |
| `--keep-ir-artifacts` | false |

CLI 在 `phasebatch/cli.py`，核心编排在 `phasebatch/optimizer.py`。

## 6. 步骤 1：准备 LLVM IR 和工具链

`phasebatch/tools.py` 发现并记录 `clang`、`opt`、`llvm-diff`、`llc`、`llvm-size` 的路径和版本，写入 `metadata.json`。

输入处理位于 `phasebatch/runner.py`：

- `.ll` 输入直接复制；
- `.c` 输入用 clang 转换；
- 其他后缀直接报错。

C 输入使用：

```text
-O0 -Xclang -disable-O0-optnone -S -emit-llvm
```

这样既保留适合逐 pass 研究的 O0 IR，又避免 `optnone` 阻止后续优化。

所有 `opt` 执行都包含：

```text
-S -verify-each
```

因此 LLVM verifier 会在 pass 之间检查 IR 合法性。

## 7. 步骤 2：加载和验证 pass

`phasebatch/pass_config.py` 把 YAML 中的展示名称映射为真实 New Pass Manager pipeline。

当前主要 pass set：

- `configs/core_passes_v1.yaml`：结构化 Core-v1；
- `configs/scalar_passes_v2.yaml`：staged scalar/memory/CFG pool；
- `configs/ipo_inline_v4.yaml`、`loop_passes_v4.yaml`、`cleanup_passes_v4.yaml`：staged v4 pools；
- `configs/vector_cleanup_passes_v5.yaml`：隔离的 vector/cleanup pool。

搜索前，每个配置 pass 都在 root IR 上单独试运行：

```text
opt -S -verify-each -passes=<pipeline> input.ll -o validate/<pass>.ll
```

结果写入：

- `valid_passes.csv`；
- `invalid_passes.csv`。

invalid pass 不进入 profiling、pair testing 和 batch 构造。

## 8. 步骤 3：创建状态

初始状态是：

```text
states/S0000/input.ll
```

每个状态记录：

- `state_id`；
- `depth`；
- `parent_state_id`；
- `transition_pass`；
- `state_hash`；
- `state_dir`。

`state_hash` 是规范化 IR 的 hash。不同路径产生相同 hash 时：

1. 新转移仍写入 state DAG；
2. child 标为 duplicate；
3. 不再重复分析该 canonical state；
4. 必要时保留到该状态的更优路径。

所以 state DAG 压缩的是“不同 batch 路径到达相同 IR”的重复搜索。

## 9. 步骤 4：单 pass profiling

对当前状态 `S`，`phasebatch/profiler.py` 分别执行：

```text
A(S), B(S), C(S), ...
```

输出暂存在：

```text
states/Sxxxx/artifacts/single_pass/<pass>.ll
```

`pass_profile.csv` 记录：

- success；
- active；
- input/output hash；
- 指令数变化；
- changed functions/blocks；
- 执行时间；
- 输出和 stderr 路径。

active 的定义是：

```text
success = true
并且
output_hash != input_hash
```

能运行但不改变 canonical IR 的 pass 是 dormant。失败和 dormant pass 都不进入当前状态的 pair matrix。

每个新状态都必须重新 profiling，因为执行一个 batch 后：

- 原本 dormant 的 pass 可能被 enable；
- 原本 active 的 pass 可能被 suppress；
- pass effect 和 pair relation 都可能变化。

## 10. 步骤 5：完整 pairwise 测试

若当前有 `n` 个 active passes，完整无序 pair 数是：

```text
n * (n - 1) / 2
```

对 pair `(A,B)`，项目比较：

```text
AB(S) = B(A(S))
BA(S) = A(B(S))
```

`pair_testing_mode=full` 会测试所有 active pairs。调度顺序可能把 changed-function 重叠的 pair 放前面，但不会减少 full 模式的 pair 数。

### 10.1 `max_pairs`

设置 `--max-pairs N` 会截断 full 模式。未测 pair 写为：

```text
dynamic_relation = not_tested
failure_kind = max_pairs
final_relation = final_unknown
```

因此设置 `max_pairs` 后，pair matrix 不再完整，exact 会记录 completeness violation。

### 10.2 lazy

lazy 只测试预算内的 pair。优先级可使用：

- `default`；
- `effect-size`；
- `history`；
- `mixed`。

超预算 pair 得到：

```text
failure_kind = lazy_budget
final_relation = final_unknown
can_hard_fold = false
```

lazy 从不把未测试 pair 推测为 commute。当前主线继续使用 full。

## 11. 步骤 6：single-pass reuse 和 pair cache

### 11.1 single-pass reuse

不复用 profiling 时，一个 pair 需要：

```text
opt S -passes=A,B
opt S -passes=B,A
```

这是 2 次 opt 调用、4 次 pass invocation。

profiling 已生成 `A(S)` 和 `B(S)` 后，当前实现改为：

```text
opt A(S) -passes=B
opt B(S) -passes=A
```

仍是 2 次 opt 调用，但只新增 2 次 pass invocation：

```text
pair_test_pass_invocations_baseline = 4
pair_test_pass_invocations_actual   = 2
pair_test_pass_invocations_saved    = 2
reused_single_pass_outputs          = true
```

如果任意 single-pass 输出不存在，就回退到完整 AB/BA pipeline。

### 11.2 pair cache

`PairRelationCache` 的 key 包含：

- state hash；
- pass 名称和真实 pipeline；
- LLVM 版本；
- target triple；
- normalizer 版本；
- equivalence comparator 版本。

命中后：

```text
pair_test_opt_runs = 0
pair_test_pass_invocations_actual = 0
pair_test_pass_invocations_saved = 4
cache_hit = true
```

当前 pair cache 只存在于一次 Python 进程/运行的内存中，不是跨进程持久化 cache。重新启动命令不会自动读取上次 cache。

成本汇总位于：

- `pair_cost_summary.csv/.md`；
- `pair_scheduling_summary.csv/.md`。

## 12. 步骤 7：AB/BA 等价性阶梯

`phasebatch/ir_equivalence.py` 按以下顺序比较 `AB.ll` 与 `BA.ll`。

### 12.1 任意一边失败

```text
AB failed 或 BA failed
  -> dynamic_failed / dynamic_timeout
  -> final_unknown
  -> can_hard_fold = false
```

失败既不是 commute，也不是 order-sensitive 证据。

### 12.2 safe canonical hash 相同

比较器去除 debug metadata、debug attachment 和注释等不稳定文本后计算 hash：

```text
hash equal
  -> equality_tier = canonical_hash
  -> dynamic_commute
  -> can_hard_fold = true
```

### 12.3 hash 不同

继续运行 `llvm-diff`。只有同时满足：

```text
llvm-diff equal
module_safety_fingerprint equal
```

才得到：

```text
equality_tier = structural_diff
dynamic_commute
can_hard_fold = true
```

### 12.4 确认不同

`llvm-diff` 不同，或 module fingerprint 不同：

```text
dynamic_order_sensitive
final_order_sensitive
can_hard_fold = false
```

### 12.5 比较器失败

`llvm-diff` 缺失、超时、读取失败：

```text
equality_tier = failed
final_relation = final_unknown
can_hard_fold = false
```

总原则是：无法证明相同，就不能 hard fold。

## 13. 步骤 8：静态 footprint 和 final relation

根据 changed functions/blocks，`phasebatch/relation.py` 计算：

- `static_disjoint_function`；
- `static_disjoint_block`；
- `static_overlap_function`；
- `static_overlap_block`；
- `static_unknown`。

静态 footprint 只用于解释，不用于独立性证明。最终关系只由动态结果决定：

| dynamic | final |
|---|---|
| `dynamic_commute` | `final_commute` |
| `dynamic_order_sensitive` | `final_order_sensitive` |
| failed / timeout / not_tested | `final_unknown` |

主要产物是 `pair_relation.csv`。

## 14. 步骤 9：构造 conflict graph

图节点是所有 active passes。

- `final_commute`：不连冲突边；
- `final_order_sensitive`：连边；
- `final_unknown`：连边；
- 缺失 pair row：也按冲突处理。

因此只有已证明 commute 的 pair 才可能进入同一 batch。

### 14.1 connected component

例如冲突边为：

```text
A -- B -- C       D -- E       F
```

组件是：

```text
C0 = {A,B,C}
C1 = {D,E}
C2 = {F}
```

A 和 C 没有直接冲突，但因为通过 B 相连，所以属于同一 component。component 只是把局部冲突问题分组，不表示组件内任意两点都冲突。

在完整 pair matrix 下，不同 component 之间没有冲突边，也就是跨 component pair 都已有 commute 证据。

## 15. 步骤 10：组件内挑 maximal independent sets

independent set 内部没有冲突边。

若 `C0={A,B,C}` 的边是 `A-B` 和 `B-C`：

- `{A,C}` 是 independent；
- `{B}` 是 independent；
- `{A,B}` 和 `{B,C}` 不是。

项目枚举 maximal independent sets：

```text
{A,C}
{B}
```

`{A}` 不是 maximal，因为还能加入 C。

项目不是只取节点数最多的 maximum set，因为较小的 maximal set 可能代表另一条必要局部选择，例如单独选择 B。

组件处理规则：

- size=1：唯一 alternative 是自身；
- `size <= max_component_size`：精确枚举所有 maximal independent sets；
- size 超限：只生成 singleton alternatives，并标记 `is_exact=false`。

超大组件会写入 `unresolved_reason`，exact 会因此标记不完整。

## 16. 步骤 11：跨组件组合 batch candidates

每个 component 先产生 alternatives，再做笛卡尔积。

例如：

```text
C0: {A,C} 或 {B}
C1: {D} 或 {E}
C2: {F}
```

得到：

```text
B0000 = {A,C,D,F}
B0001 = {A,C,E,F}
B0002 = {B,D,F}
B0003 = {B,E,F}
```

batch 的 `canonical_order` 按当前 `pass_profile.csv` 中的稳定 pass 顺序生成。它用于验证参考顺序、真实执行和最终 pipeline 重建，但不代表其他排列天然等价。

若笛卡尔积超过 `max_batch_candidates`，停止生成并写：

```text
truncated = true
```

budgeted 可以在截断集合上继续；exact 会记录 `truncated_batch_candidates`。

主要产物：

- `batch_components.csv`；
- `batch_candidates.csv`；
- `batch_summary.csv/.md`。

## 17. 步骤 12：为什么还要验证整个 batch

pairwise 只证明 source state 上的：

```text
AB(S) == BA(S)
```

但 batch 中某个其他 pass 先改变 IR 后，A/B 的关系可能变化。即使 A/B、A/C、B/C 在 S 上分别 commute，也不能直接断言 ABC 的 6 个排列全部等价。

因此：

```text
pair evidence -> 候选生成
batch validation -> 执行许可证
```

candidate 的 `is_exact=true` 只表示 component 枚举没被截断，不表示 batch 已得到 correctness certificate。

## 18. 步骤 13：batch validation ladder

| 模式 | 行为 | hard certificate |
|---|---|---|
| `exhaustive` | 执行全部 `k!` 排列 | 全部相等时可以 |
| `dag` | subset/equivalence DAG 覆盖全部排列 | DAG 完整且最终一类时可以 |
| `bounded` | 有限 insertion orders | 不可以 |
| `sampled` | 固定随机种子抽样 | 不可以 |
| `auto` | 小 batch exhaustive，大 batch先 DAG，DAG 不完整则 bounded | 取决于实际 tier |

auto 当前规则：

```text
如果 k! <= 120:
    exhaustive
否则:
    permutation DAG
```

因此 1 到 5 个 pass 自动全排列，6 个及以上先走 DAG。

### 18.1 exhaustive

先执行 canonical order，再执行其余所有排列。每个结果与 canonical output 使用相同等价性阶梯比较。

只有完整覆盖且全部可 hard-fold equal 才得到：

```text
validation_status = all_permutations_same
validation_hard_certificate = true
```

### 18.2 bounded 和 sampled

bounded 测试有限 insertion orders，sampled 使用固定 seed 0 抽样。即使都相同，也只能得到：

```text
bounded_same / sampled_same
validation_complete = false
validation_hard_certificate = false
```

### 18.3 profiling 输出复用和成本口径

profiling 已经为当前 state 生成每个 active pass 的 `P(S)`。非 canonical validation order 可以从第一个 pass 的 profiling 输出继续执行剩余 pipeline；canonical order 始终完整执行，作为独立锚点。复用只接受 state hash 匹配、profiling 成功且文件仍存在的输出。

每个 validation row 记录：

- `validation_opt_invocations`：实际启动的 `opt` 次数；
- `validation_pass_invocations_baseline`：不复用时需要的 pass invocation；
- `validation_pass_invocations_actual`：实际执行的 pass invocation；
- `validation_pass_invocations_saved`：两者之差；
- `validation_profile_reuse_hits`；
- `validation_state_transition_cache_hits`；
- `validation_state_equivalence_cache_hits`。

因此 `tested_orders` 只表示验证了多少顺序，不再被当作 `opt` 调用数。

并行时，各 candidate row 的 `time_ms` 会重叠，求和表示累计工作时长而不是 wall-clock。根目录 `optimizer_timing.csv` 的 `batch_validation_time_ms` 由顶层计时器记录真实 validation wall 时间。

## 19. validation DAG 的具体过程

一个 DAG 节点表示：

```text
(已经执行的 pass 子集, 当前 IR 等价类)
```

节点保存 deterministic ID、subset bitmask、IR 路径、hash、representative path 和 depth。

从一个节点出发，对每个未执行 pass 建边：

```text
(subset, IR class) --追加 P--> (subset+P, new IR class)
```

每条 root-to-full-subset path 对应一个完整排列。

### 19.1 节点合并

新 IR 只与相同 subset 中已有节点比较：

1. canonical hash 相同，直接合并；
2. hash 不同，走 `compare_ir_equivalence`；
3. 只有 `can_hard_fold=true` 才结构合并；
4. 确认不同则创建新等价类节点；
5. 比较器失败则 validation 失败。

### 19.2 transition cache

key 是：

```text
(source canonical hash, pass name, resolved pipeline)
```

若不同路径已经到达相同 source IR，再追加相同 pass 时不重复调用 opt。cache 由当前 state 的 `ValidationRuntime` 持有，因此同一 state 的不同 batch candidates 也能复用；它不会跨 state 或跨 Python 进程持久化。

profiling 的 `P(S)` 会先注册为 root transition seed，所以 DAG 的第一层可以直接命中已有结果。

### 19.3 equivalence cache

相同 hash pair 的结构比较结果也在当前 state 的 runtime 内缓存，避免同一或不同 candidates 重复执行 `llvm-diff`。失败的 transition 或比较结果不会被当作成功 cache entry。

### 19.4 并行与确定性

`jobs` 是一个 state validation runtime 的全局 `opt` 并发上限，不允许“候选并行 × 候选内部并行”无限放大进程数。

- 不同 batch candidates 可以并行验证，但最终 CSV 按原 candidate index 写回；
- DAG 同一 depth 的 transition 可以并行执行 `opt` 和 hash；
- 等价类合并、node ID 和 edge 顺序按预先确定的 canonical 顺序串行处理；
- 多个并行 transition 同时失败时，报告预分配顺序中最早的失败，而不是最先完成的失败。

所以 `jobs=1` 与 `jobs>1` 的 status、certificate、节点编号、边顺序和 representative path 保持一致；wall-clock 可以不同。

### 19.5 终止结果

full subset 最终只有一个等价类：

```text
all_permutations_same
validation_tier = permutation_dag_exact
validation_complete = true
validation_hard_certificate = true
```

存在多个类：

```text
mismatch
validation_tier = permutation_dag_mismatch
```

超过 node/edge 预算：

```text
incomplete
validation_tier = permutation_dag_incomplete
```

opt、verifier 或比较器失败：

```text
failed
```

### 19.6 DAG 压缩指标

应同时查看：

- `factorial_permutations`；
- `validation_dag_nodes`；
- `validation_dag_edges`；
- transition/equivalence cache hits/misses；
- `compression_vs_permutation`；
- validation wall-clock。

DAG 边数减少表示验证结构被压缩，但 hash、`llvm-diff`、文件 I/O 也有成本，所以不保证 wall-clock 一定更短。

### 19.7 dump

`--dump-validation-dag` 会额外保留 node CSV、edge CSV、DOT 和 DAG IR，并写 `.keep_ir_artifacts`。

dump 只是保留调试细节的开关。不开 dump 时 DAG 仍然会运行。

## 20. correctness classifier

`phasebatch/batch_correctness.py` 把 validation 结果转换为统一资格：

| validation | class | can_hard_fold | 默认 can_execute |
|---|---|---:|---:|
| 完整 `all_permutations_same` | `certified_batch` | true | true |
| `bounded_same` | `bounded_batch` | false | false |
| `sampled_same` | `sampled_batch` | false | false |
| `mismatch` | `rejected_batch` | false | false |
| `failed` | `failed_batch` | false | false |
| incomplete/not validated | `unvalidated_batch` | false | false |

budgeted 可以显式允许 bounded 或 sampled batch 执行，但它们仍不是 hard-fold 证明。exact 直接禁止这两个允许开关。

最重要的门是：

```text
can_execute != true
  -> optimizer 绝不执行该 batch
```

## 21. active pass coverage

`coverage_report.csv` 检查每个 active pass 是否：

- `certified_covered`；
- `heuristic_covered`；
- `unresolved_conflict`；
- `validation_rejected`；
- `unvalidated_covered`；
- `failed_or_unknown`；
- `not_executed_due_to_max_depth`；
- `dropped`。

它防止 active pass 因候选截断、组件异常或验证失败而悄悄消失。

`dropped_active_passes > 0` 会让 exact 结果不完整。

## 22. 执行 batch 并产生 child state

只有 `can_execute=true` 的 candidate 才会按 canonical order 执行：

```text
opt parent.ll -passes=<resolved batch pipeline> -o child.ll
```

成功后：

1. 计算 child canonical hash；
2. 查找是否已有相同 state；
3. duplicate 只记录转移，不重复分析；
4. 新 state 重新进行 profiling、pair testing 和 batch 构造；
5. 记录 child 指令数和 state DAG edge。

状态变化后必须重新分析，因为所有 relation 都是 state-local 的。

## 23. rolling-exact、exact、budgeted 和 auto

### 23.1 rolling-exact（当前主线）

`rolling-exact` 把默认深度 2 解释成滚动窗口，而不是全局停止深度。每个窗口可以有
最多五个根状态 `R1..R5`：

```text
从 R1..R5 开始
  -> local depth 0：完整构造、验证并执行全部 exact-executable batches
  -> local depth 1：对所有不同 canonical children 重复完整展开
  -> 得到 depth 2 open 终态以及提前闭合的终态
  -> closed 终态进入最终候选，但不占继续搜索槽位
  -> open 终态不超过 5 时全部保留
  -> 超过 5 时按 objective/call/memory/branch/novelty 桶保留 5 个
  -> 五个终态共同成为下一窗口的根
  -> 重复到闭合
```

前两层窗口内不调用 batch selection 或 frontier selection。即使命令中存在
`beam_width` 和 `max_batches_per_state`，它们也只属于 budgeted，不影响
`rolling-exact`。`rolling_frontier_width=5` 只在第二层窗口结束后生效，不是逐层 beam。

根状态只要有 exact-executable outgoing edge，就不是本窗口候选终态。因此第一步
即使暂时让 IR 指令数变多，也可以继续走到第二层，避免把 enable 后续 pass 的路径
提前剪掉。

正常闭合条件有三种：

| closure reason | 含义 |
|---|---|
| `no_active_passes` | 当前已提交状态没有 active pass |
| `no_executable_batches` | 完整构造和验证后没有 exact-executable batch |
| `state_graph_closed` | 所有终态都回到已经提交过的 canonical state，再滚动只会形成环 |

主线状态是：

```text
rolling_exact_complete
rolling_exact_incomplete
rolling_exact_incomplete_continued
```

只有前三种正常闭合可以得到 `rolling_exact_complete`。pair 未测全、component
unresolved、candidate 截断、dropped 不为 0、validation/DAG 不完整、batch apply
失败、state cap 或正数 window cap 命中，都得到 incomplete。元数据写入：

```text
exact_scope = rolling_global_exact_to_closure
# 或 rolling_window_exact_frontier_limited
rolling_window_depth
rolling_frontier_width
max_rolling_windows
rolling_windows_completed
rolling_committed_depth
rolling_closure_reason
rolling_frontier_pruned
rolling_frontier_states_pruned
global_search_complete
```

`rolling_windows.csv` 逐窗口记录全部根、完整展开的状态/边、open/closed 终态数、
五个保留状态、选择桶、边界剪枝数量和闭合原因。`frontier_scores.csv` 记录检查点
选择依据，使用 `rolling_checkpoint_selected/pruned`，不与 budgeted 的 beam 标签混淆。

`rolling_exact_complete` 说明展开过的状态都使用完整正确性证据，并且窗口正常结束；
它不自动说明全局状态图未剪枝。只有 `global_search_complete=true` 才表示运行期间
没有 open 边界状态因 K=5 被丢弃。

### 23.2 exact（固定深度对照）

exact 在每个状态执行全部满足以下条件的候选：

```text
correctness_class = certified_batch
can_hard_fold = true
validation_status = all_permutations_same
```

它不使用 beam pruning。

以下情况会记录 exact incomplete：

- lazy budget 或 `max_pairs`；
- candidates 截断；
- unresolved component；
- dropped active pass；
- correctness/validation row 缺失；
- DAG 超预算或不完整；
- 存在 failed、bounded、sampled、unvalidated 或 unknown validation；
- state cap 超限。

`validation_status=mismatch` 是完整负证据：它证明该 candidate 不能执行，因此
记录为 `rejected_batch` 并安全跳过，但不会使 exact scope 变成 incomplete。验证
失败、未完成或只有弱证据时仍然 incomplete。

默认 `exact_fail_on_incomplete=true`，出现问题就停止。`--no-exact-fail-on-incomplete` 可以继续收集数据，但状态是 `exact_incomplete_continued`。

exact 仍受 pass set、`max_rounds`、`max_states` 和 certified batch graph 范围限制，不表示 LLVM 全局最优。

exact 必须使用 `--budgeted-validation-strategy all`；显式请求 `on-demand` 会直接报错，防止未验证候选被误认为完整 exact scope。

固定深度 exact 使用 `exact_scope=fixed_depth_exact`。只有采用 full、没有
`max_pairs` 截断，并且所有 active pair 都有结果时，`pair_matrix_complete` 才是
true；scope 名称本身不能覆盖这个完整性布尔值。

### 23.3 budgeted（性能/消融对照）

每轮：

1. 找出 `can_execute=true` 的 candidates；
2. batch selection 最多选择 `max_batches_per_state` 个；
3. 执行并产生 children；
4. frontier selection 最多保留 `beam_width` 个；
5. 达到轮数、状态上限或没有 executable batch 时停止。

batch selection 和 frontier selection 是两个独立决策。策略包括：

- `score`；
- `largest-batch`；
- `certified-first`；
- `objective`；
- `diverse`。

默认均为 `score`。budgeted 默认仍只执行 certified batches。

budgeted 另有两种 validation scope：

| strategy | 行为 |
|---|---|
| `all` | 默认；验证全部 candidates，再做 correctness classification 和 batch selection |
| `on-demand` | 实验性；按 cheap deterministic rank 逐个验证，找到 `max_batches_per_state` 个 executable candidates 后停止 |

`on-demand` 的预验证排序依次偏好 active-pass coverage 更大、batch 更大、unresolved components 更少的候选，最后用 canonical order 和 batch ID 打破平局。被跳过的候选仍写入 `batch_validation.csv`：

```text
validation_status = not_validated
validation_tier = unvalidated
validation_complete = false
validation_hard_certificate = false
validation_incomplete_reason = budgeted_on_demand_not_selected
```

这些行经过 correctness classifier 后仍是 `can_execute=false`。`on-demand` 可能改变 budgeted 搜索到的 objective，因此报告必须注明 scope，不能把它当作 exact 结果比较。exact 明确拒绝 `on-demand`。

### 23.4 auto（兼容模式）

auto 先审计 root：

- validation 是否开启；
- 是否允许弱证据；
- candidates 是否截断；
- component 是否 unresolved；
- candidates 是否全部 certified；
- batch/state 估计是否在预算内。

条件全部满足时选固定深度 exact，否则选 budgeted，并把原因写入
`auto_reason`。`auto` 不会自动替代当前默认的 `rolling-exact`。

## 24. 最终状态选择

当前 objective 是 IR instruction count，越小越好。

`rolling-exact` 只在当前完整窗口的终态之间比较；选中的终态和整段局部路径被
提交，最终结果是最后一个已提交状态。未提交探索分支即使出现更小的中间 IR，
也不能绕过窗口决策成为 final state。固定深度 exact 和 budgeted 才在各自已到达
范围内使用原有 final-state selection。

目标相同时依次偏好：

1. path 更短；
2. pass invocation 更少；
3. certified 比例更高；
4. state ID 更小。

budgeted 搜索的 frontier 不再只保留“IR 最小”的单一路径。当前还为
direct call、`load + store`、branch、hash novelty 和原有 IR objective
保留确定性桶位。这些桶只决定哪些已到达状态继续留在 beam 中，不改变
pair/batch correctness，也不改变单次 `optimize-batches` 默认仍以 IR 指令数
选择最终状态。

主要状态产物：

- `states.csv`；
- `state_dag.csv`；
- `batch_state_transitions.csv`；
- `leaf_states.csv`；
- `frontier_scores.csv`；
- `optimizer_events.csv`。
- `rolling_windows.csv`（rolling-exact）。

## 25. chosen path、pipeline 和 replay

优化器从 selected final state 回溯到 `S0000`，生成：

- `chosen_path.csv`；
- `chosen_path_summary.csv`；
- `optimized_batches.txt`；
- `optimized_pipeline.txt`；
- `optimized_pipeline_readable.txt`；
- `final_state.txt`；
- `exact_status.txt`；
- `final.ll`。

默认随后进行 replay：

1. 读取 `states/S0000/input.ll`；
2. 读取 `optimized_pipeline.txt` 和 `chosen_path.csv`；
3. 当 chosen path 与完整 pipeline 一致时，保留选中 batch 的边界逐段回放，并让后一段读取前一段输出；
4. 对没有可靠 chosen path 的旧运行，兼容为从 root 一次性执行完整 pipeline；
5. 生成 `replayed_final.ll`；
6. 与 `final.ll` 走同一等价性阶梯；
7. 写 `pipeline_replay.csv` 并更新 summary。

保留 batch 边界很重要：把相同 pass 顺序合成一次 `opt` 调用，会消除各段之间的
IR 输出/重读和 analysis manager 重建边界，不一定能严格复现搜索阶段生成 `final.ll`
时的执行过程。回放使用临时 `.ll` 承接中间段，完成后自动删除。

replay 证明最终 pipeline 能否重现 selected final IR，不替代 batch 排列验证。

### 25.1 staged optimizer

`optimize-staged` 用 manifest 把多个小 pass set 串起来。每个 stage 仍调用原有
`optimize-batches`，因此 full pairwise、single-pass reuse、pair cache、validation
DAG 和 correctness classifier 都没有被绕过。

每个 stage 的过程是：

1. 对当前 handoff IR 运行该 stage 的完整搜索；
2. 只从已经到达的状态中选择 stage 结果；
3. 把选择的 IR 复制为下一个 stage 的输入；
4. 记录该段 pipeline；
5. 最后从第一个 root 顺序 replay 所有 stage segments，并与最终 IR 比较。

对于 IPO 这类“identity 虽然 IR 更小，但不执行就失去阶段意义”的 stage，
manifest 可以设置 `require_transition: true`。如果普通 objective 选择 root，
orchestrator 会改选 objective 最好的、已经到达且 pipeline 非空的终态；如果
不存在这种状态就失败，不会静默跳过必需阶段。

### 25.2 安全的 runtime top-K 重排

manifest 可为某个 stage 设置 `runtime_rerank: true`，并在顶层配置 warmup、
trials、timeout 和命令模板。runtime 候选必须是 leaf 或原静态选择的终态，
IR 和 parent chain 必须可重放，并按 canonical state hash 去重。

候选选择保留静态 objective、direct call、memory op 和 branch 桶。随后：

1. 使用 `llc` 和 `clang` 编译每个候选；
2. 编译或链接失败的候选绝不执行；
3. warmup 不计入统计；
4. 正式运行按 cyclic order 轮换候选位置；
5. timeout 或 exit code 不符的候选标记为 ineligible；
6. 其余候选按 median runtime、state ID 确定性排序；
7. 没有合格 winner 时回退到静态 stage 结果。

runtime 只是一种 objective signal，不是 pair、batch 或 path correctness 证据。
它只能在已经由原 correctness 流程安全到达的状态之间改变选择。

staged 主输出包括 `staged_summary.csv`、`staged_pipeline.csv`、
`staged_replay.csv` 和 `staged_summary.md`。runtime stage 还写
`runtime_candidates.csv`、`runtime_trials.csv`、`runtime_summary.csv` 和
`runtime_selection.md`。

只要 manifest 中有任意 budgeted stage，整体 scope 就是
`mixed_or_budgeted_stages`；不能因为其他 stage 是 exact 或 pair matrix 完整，
就把整个 staged run 称为 exact complete。

## 26. `.ll` 清理

`phasebatch/artifact_cleanup.py` 在以下工作结束后清理：

1. 搜索和 final state 选择；
2. chosen path/pipeline 写出；
3. 可选 baselines；
4. final pipeline replay；
5. final、pair、validation 和 timing summaries。

默认递归删除运行目录中的所有 `*.ll`，再删除空目录。

当前实现按扩展名删除，不区分中间和最终 IR。因此以下文件都会删除：

```text
input.ll
single-pass/pair/DAG IR
child state IR
final.ll
replayed_final.ll
```

hash、objective、pipeline 和 replay 结论保留在 CSV/MD/TXT/JSON 中。某些 CSV 的路径字段会继续指向已经删除的临时文件。

保留全部 IR：

```text
--keep-ir-artifacts
```

保留 validation DAG 调试 IR：

```text
--dump-validation-dag
```

cleanup 只允许删除 resolved run root 内的文件，不删除 run root，不删除非空目录。

## 27. 缓存与复用层

| 层 | 生命周期 | 节省内容 |
|---|---|---|
| single-pass materialization | 当前 state，cleanup 前 | pair 第一段 pass |
| pair relation cache | 单次 Python 进程 | 整个 AB/BA 测试 |
| batchification memo | 单次 optimizer run | 同一 state 重复构造和验证 |
| validation transition cache | 当前 state validation runtime | 同一或重叠 candidates 的重复 opt transition |
| validation equivalence cache | 当前 state validation runtime | 同一或重叠 candidates 的重复结构比较 |
| validation single-flight | 当前 state validation runtime | 并发线程对同一 key 只计算一次 |
| optimizer state dedup | 单次 optimizer run | 重复 state 分析和扩展 |

当前没有跨命令持久化 pair 或 validation cache。validation cache 的 IR 放在 state-local content-addressed 目录；`--dump-validation-dag` 会写 keep marker，普通运行则由最终 cleanup 清理。

## 28. 三层证据边界

### pair-level

回答：

```text
在当前 S 上，AB(S) 与 BA(S) 是否等价？
```

用于 conflict graph。

### batch-level

回答：

```text
一个 batch 的全部排列是否等价？
```

用于 correctness classifier。

### path-level

回答：

```text
chosen path 展开的 pipeline 是否能重现 final state？
```

用于可复现性。

最强主线证据是：

```text
完整 pair matrix
+ certified batch validation
+ successful final replay
```

## 29. 输出阅读顺序

### 根目录

| 文件 | 含义 |
|---|---|
| `metadata.json` | 工具版本、模式和预算 |
| `states.csv` | 所有状态和 duplicate 信息 |
| `state_dag.csv` | 状态 DAG |
| `batch_state_transitions.csv` | batch 转移证据 |
| `leaf_states.csv` | 状态停止原因 |
| `chosen_path.csv` | 最终路径 |
| `optimized_pipeline.txt` | 最终 LLVM pipeline |
| `final_state.txt` | final state/hash/objective |
| `exact_status.txt` | exact 完整性 |
| `pipeline_replay.csv` | replay 结果 |
| `optimizer_timing.csv` | 各阶段耗时 |
| `optimize_summary.md` | 优化器摘要 |
| `final_summary.md` | 最终总览 |

### 每个 state

建议按以下顺序阅读：

```text
per_state_summary.csv
pass_profile.csv
pair_relation.csv
batch_components.csv
batch_candidates.csv
batch_validation.csv
batch_correctness.csv
coverage_summary.csv
```

它们依次回答：

```text
当前状态多大
-> 哪些 pass active
-> 哪些 pair commute
-> 冲突组件是什么
-> 生成了哪些 batch
-> batch 排列是否等价
-> batch 能否执行
-> active pass 是否完整覆盖
```

默认清理后 `.ll` 不存在；要查看具体 IR，运行时必须加 `--keep-ir-artifacts` 或 dump 开关。

## 30. 命令层次

核心分阶段命令：

- `analyze`：profiling 和 pair testing；
- `batchify`：从已有 state 构造、验证、分类 batch；
- `explore-batches`：分析型多状态扩展；
- `optimize-batches`：完整优化主线；
- `optimize-staged`：顺序执行多个小 pass stage，并可对安全终态做 runtime top-K 重排；
- `replay-final-pipeline`：重放已有最终 pipeline。

worker 验收命令是 `verify-opt-worker` 和 `benchmark-opt-worker`；导师数据命令是
`run-advisor-report-zh` 和 `summarize-advisor-report-zh`。

只读报告命令如 `summarize-reduction`、`export-evidence-pack`、`diagnose-paths`、`visualize-dag` 和 `summarize-components` 主要读取已有 CSV，不产生新的 correctness 证据。

## 31. 代码模块索引

| 流程 | 模块 |
|---|---|
| CLI | `phasebatch/cli.py` |
| optimizer | `phasebatch/optimizer.py` |
| staged orchestrator | `phasebatch/staged_optimizer.py` |
| staged manifest | `phasebatch/staged_config.py` |
| runtime top-K | `phasebatch/runtime_rerank.py` |
| 工具链 | `phasebatch/tools.py` |
| LLVM 执行 | `phasebatch/runner.py` |
| pass 配置 | `phasebatch/pass_config.py` |
| IR 标准化 | `phasebatch/normalizer.py` |
| profiling | `phasebatch/profiler.py` |
| pair testing | `phasebatch/pair_tester.py` |
| pair cache | `phasebatch/pair_cache.py` |
| IR equivalence | `phasebatch/ir_equivalence.py` |
| relation | `phasebatch/relation.py` |
| conflict graph | `phasebatch/graph.py` |
| pairwise batcher | `phasebatch/batcher.py` |
| validation DAG | `phasebatch/batch_validation_dag.py` |
| validation worker/cache runtime | `phasebatch/validation_runtime.py` |
| correctness | `phasebatch/batch_correctness.py` |
| coverage | `phasebatch/coverage.py` |
| replay | `phasebatch/pipeline_replay.py` |
| cleanup | `phasebatch/artifact_cleanup.py` |

逐文件职责可继续查阅 `docs/code_file_roles.md`。

## 32. 推荐运行方式

正式主线（窗口内完整两层，边界 K=5，滚动到闭合）：

```powershell
python -m phasebatch optimize-batches `
  --input <input.c-or-ll> `
  --out <output-dir> `
  --passes configs/core_passes_v1.yaml `
  --mode rolling-exact `
  --rolling-window-depth 2 `
  --rolling-frontier-width 5 `
  --max-rolling-windows 0 `
  --objective ir-inst-count `
  --validate-batches `
  --batch-construction-mode pairwise `
  --pair-testing-mode full `
  --batch-validation-mode auto
```

固定深度 exact 只用于 `exact-rN` 对照实验：

```powershell
python -m phasebatch optimize-batches `
  --input <input.c-or-ll> `
  --out <output-dir> `
  --passes configs/core_passes_v1.yaml `
  --mode exact `
  --max-rounds 4 `
  --validate-batches `
  --pair-testing-mode full
```

budgeted 按需验证只在明确接受较小 validation scope 时使用：

```powershell
python -m phasebatch optimize-batches `
  --input <input.c-or-ll> `
  --out <output-dir> `
  --passes configs/core_passes_v1.yaml `
  --mode budgeted `
  --max-batches-per-state 2 `
  --budgeted-validation-strategy on-demand `
  --validate-batches `
  --batch-construction-mode pairwise `
  --pair-testing-mode full `
  --batch-validation-mode auto
```

较复杂程序使用 budgeted：

```powershell
python -m phasebatch optimize-batches `
  --input <input.c-or-ll> `
  --out <output-dir> `
  --passes configs/core_passes_v1.yaml `
  --mode budgeted `
  --max-rounds 4 `
  --beam-width 8 `
  --max-states 500 `
  --max-batches-per-state 20 `
  --validate-batches `
  --batch-construction-mode pairwise `
  --pair-testing-mode full `
  --batch-validation-mode auto
```

Salsa20 staged v5：

```powershell
python -m phasebatch optimize-staged `
  --input E:\llvm-test-suite\SingleSource\Benchmarks\Misc\salsa20.c `
  --manifest configs\staged_salsa20_v5.yaml `
  --out outputs\salsa20_staged_v5 `
  --jobs 8 `
  --timeout 30
```

当前 E5 结果中，三个带 `loop-vectorize` 的终态都产生了 6 条 XMM 指令行，
但 median 比 identity 慢 0.61% 到 2.40%；runtime reranker 因此正确选择 identity。
同 root 的循环交错测量中，E5 winner 为 2756.859 ms，LLVM `default<O2>` 为
2149.908 ms，E5 仍慢 28.23%。这说明当前 vector stage 能产生 SIMD，但尚未
复现 O2 的有利变换组合，不应直接并入默认 flat pass set。

## 33. 常驻 LLVM worker 执行后端

项目现在默认使用严格模式的 `phasebatch-worker.exe`。它不是新的搜索算法，
也不改变 full pairwise、batch 构造、validation ladder 或 correctness
classifier。它只替换原来数千次短生命周期的 `opt.exe` 调用。

构建和启用方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_worker.ps1

python -m phasebatch optimize-batches `
  --input <input.c-or-ll> `
  --out <output-dir> `
  --passes configs\core_passes_v1.yaml `
  --mode exact `
  --validate-batches `
  --opt-backend worker `
  --opt-worker worker\build\phasebatch-worker.exe `
  --opt-workers 8
```

默认值是 `--opt-backend worker`。worker 缺失、启动失败、协议错误或超时都
直接报错，不会静默回退。只有显式指定 `--opt-backend external` 才使用传统
短进程路径；`auto` 允许记录后回退到 external，但不用于默认正式采集。对应环境变量为
`PHASEBATCH_OPT_BACKEND`、`PHASEBATCH_OPT_WORKER` 和
`PHASEBATCH_OPT_WORKERS`。

worker 的具体过程如下：

```text
Python 启动固定数量的常驻 worker
  -> worker 在独立 LLVMContext 中解析输入 IR 并返回 process-local handle
  -> 后续 pass 通过内存 bitcode 克隆到新的 LLVMContext 中运行
  -> 返回完整 IR hash、feature counts 和 child handle
  -> hash 相同时直接得到严格的文本相等证据
  -> hash 不同时才 materialize，并继续 structural diff + fingerprint
  -> borrowed handle 若因并发 LLVM fatal 重启而失效，同一 AB/BA pipeline 直接物化重跑一次
  -> 重试次数写入 materialization_retry 和 pair_test_retry_opt_runs
  -> structural diff 本身由常驻 worker 内的官方 LLVMDiff 完成
  -> transition 生命周期结束后 release 引用
  -> 路径缓存使用有界 LRU，淘汰时释放自己的引用
  -> 路径签名包含文件内容 SHA-256，不只依赖 mtime 和 size
  -> 无诊断 EOF 只重试一次
  -> pass 触发的 LLVM ERROR 记为 llvm_fatal pipeline failure，并重启 worker
  -> worker timeout 或协议/基础设施错误在严格模式直接报错
  -> 上述情况均不回退到 external
```

pair AB/BA、permutation DAG、exhaustive 和 bounded validation 都支持延迟
落盘。`--dump-validation-dag` 和 `--keep-ir-artifacts` 会强制保留调试 IR；
普通运行结束后仍删除 `.ll` 和空目录。

本机 Ryzen 7 9800X3D 的 `crc8.be H=3,K=5` 扩展测试中，8/12/16 workers
分别为 46.795/48.758/48.762 秒；三次搜索图、pipeline 和 replay 一致。
因此本机正式实验保持 `jobs=8`，不把 16 个 SMT 逻辑线程误当成 16 个物理核。

在 2026-07-10 的 Salsa20 Core-v1 exact 单轮 matched 实验中，external
wall-clock 为 87.798 秒，8-worker 为 25.053 秒，端到端加速 3.504 倍。
两边都是 6 个 states、5 个 transitions、final objective 354、同一条 final
pipeline，并且 replay 成功。178 个 pair 的 correctness 关系和 22 个 batch
validation 分类没有差异。100 次 file-compatible 微基准的 aggregate median
加速为 6.847 倍。这里衡量的是 Phasebatch 搜索程序本身的运行时间，不是
Salsa20 编译结果的运行时间。

使用 worker 前应先运行：

```powershell
python -m phasebatch verify-opt-worker `
  --inputs <input.c-or-ll> `
  --passes <passes.yaml> `
  --out <verify-dir> `
  --opt-worker worker\build\phasebatch-worker.exe
```

任何 status、hash、structural diff 或 module fingerprint 差异都会让门禁
失败，且不会建议把该 worker 用于正式实验。

## 34. Advisor Data Report 中文导师汇报层

`run-advisor-report-zh` 在稳定主线上增加多程序实验编排，但不改变任何
correctness 或搜索规则。它从 `SingleSource/**/*.c` 扫描或读取 manifest，
记录每个 clang smoke 结果，确定性选择多个目录下的程序，然后在同一个严格
worker pool 中逐程序调用现有 `optimize_batches()`。

正式采集默认选择 50 个程序，并使用：

```text
mode = rolling-exact
rolling_window_depth = 2
rolling_frontier_width = 5
max_rolling_windows = 0
max_pairs = None
max_states = 2000  # 仅安全上限，命中即 incomplete
```

原 `outputs/advisor_report_zh_20programs` 是 `budgeted + max_rounds=2 +
beam_width=4` 的 pilot 数据。它仍可用于结构和成本预览，但不能改名为 exact，
也不能作为新主线的闭合性证据。正式 50 程序实验必须写入新的输出目录。

每个成功程序仍保留原有结构化证据。离线汇总器把这些证据转换为两套明确区分
的 component：overlap component 只描述 footprint 相关性，conflict component
则由 order-sensitive、unknown 和 missing relation 构成。随后生成 pair 比例、
小 cluster ABBA、coverage、局部 `log10(n!)` reduction、成本、冲突 pass 排名、
state-aware 变化、九组中文图和代表程序 DAG。

```powershell
python -m phasebatch run-advisor-report-zh `
  --test-suite-root E:\llvm-test-suite `
  --out outputs\advisor_report_zh_50programs_rolling_exact `
  --passes configs\core_passes_v1.yaml

python -m phasebatch summarize-advisor-report-zh `
  --study-dir outputs\advisor_report_zh_20programs
```

该命令只读已有输出，不会调用 LLVM。缺失产物进入 `missing_outputs.csv`，对应
值显示为 N/A，不能被误记为 0。最终报告始终声明 relation 是 state-local，
overlap 不是正确性证明，且本报告不主张 runtime 优于 O2/O3。

## 35. 一句话总结

Phasebatch 在每个 LLVM IR 状态上先找出 active passes，用完整 AB/BA 动态测试建立 pair-level independence 证据，再把非 commute 关系构造成 conflict graph，从组件的 maximal independent sets 组合 batch candidates；随后用全排列或 permutation DAG 验证 batch-level correctness，只有 correctness classifier 允许的 batch 才能成为 optimizer state DAG 的边；当前主线在每个局部窗口完整展开两层，只在窗口边界按五个多指标桶保留 K=5，并从这些状态继续滚动到状态图闭合，H=3、固定深度 exact 和 budgeted 作为对照。最终 pipeline 必须 replay，随后保留结构化证据并清理 `.ll` 与空目录。
