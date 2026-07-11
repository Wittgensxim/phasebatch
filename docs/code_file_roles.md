# 当前代码文件作用说明

> 更新时间：2026-07-11
>
> 当前正式主线：
> `full pairwise + reuse/cache + certified validation + rolling-exact + strict worker`。
> 固定深度 `exact` 与 `budgeted` 保留为对照模式。

## 入口与配置

- `phasebatch/__main__.py`：`python -m phasebatch` 入口。
- `phasebatch/cli.py`：命令、参数默认值、wrapper 参数转发与 worker session 边界。
- `phasebatch/config.py`：简单 YAML 配置解析辅助。
- `phasebatch/pass_config.py`：结构化 pass registry、pipeline、stage 与 canonical order。
- `phasebatch/schema.py`：跨模块共享的 CSV 字段。
- `phasebatch/tools.py`：LLVM 工具发现、版本、target 与 `metadata.json`。

## LLVM 执行与 IR

- `phasebatch/runner.py`：准备 root IR、调用外部 opt 或 worker backend、延迟物化结果。
- `phasebatch/opt_worker.py`：常驻 worker 进程池、JSON 协议、超时和重启。
- `phasebatch/opt_backend.py`：module handle、路径缓存、apply/materialize 和进程内 LLVMDiff。
- `phasebatch/normalizer.py`：canonical IR hash 与静态 IR feature。
- `phasebatch/ir_equivalence.py`：canonical hash、LLVMDiff 和 module fingerprint 等价性阶梯。
- `phasebatch/ir_parser.py`：函数、basic block 与 instruction 的轻量解析。
- `phasebatch/artifact_cleanup.py`：正常结束后删除 `.ll` 和空目录；调试 keep marker 除外。

## State、Profiling 与 Pair

- `phasebatch/state_analysis.py`：单个 reached state 的 profiling、pair testing、relation、成本和摘要编排。
- `phasebatch/profiler.py`：单 pass activity、effect footprint 与可复用的 `A(S)` 输出。
- `phasebatch/pair_tester.py`：AB/BA 动态测试、single-pass reuse、IR equality 与保守失败分类；borrowed handle 延迟物化失效时，用同一 pipeline 做一次直接物化重试，避免线程调度改变 pair relation。
- `phasebatch/pair_cache.py`：state-local pair memoization。
- `phasebatch/pair_cost.py`：opt 调用、pass invocation、cache 与 comparator 成本。
- `phasebatch/pair_scheduling.py`：full/lazy 调度和统计；正式主线使用 full。
- `phasebatch/relation.py`：静态 footprint 与动态 equality 合并为 final relation。
- `phasebatch/footprint.py`：coarse overlap 诊断，不构成 correctness proof。
- `phasebatch/graph.py`：active-pass interaction 与 component 图辅助。
- `phasebatch/report.py`：单状态及聚合分析摘要。

## Batch 构造与验证

- `phasebatch/batcher.py`：pairwise conflict graph、connected components、maximal independent sets、candidate 组合和 validation 调度。
- `phasebatch/batch_validation_dag.py`：permutation DAG、prefix/state 合并、transition/equivalence cache。
- `phasebatch/validation_runtime.py`：state-local validation worker 配额、single-flight 和跨 candidate cache。
- `phasebatch/batch_validation_ladder.py`：exhaustive/DAG/bounded/sampled validation 汇总。
- `phasebatch/batch_correctness.py`：validation 到 correctness class、`can_hard_fold`、`can_execute` 的唯一分类边界。
- `phasebatch/coverage.py`：active pass coverage、unresolved/rejected/terminal 和 dropped 检查。
- `phasebatch/batch_objective.py`：已有 transition 的 objective 评价。

## 搜索主线

- `phasebatch/optimizer.py`：三类搜索器、canonical state DAG、chosen path、最终产物和清理。
  - `rolling-exact`：当前默认。每个窗口完整 BFS 两层，窗口内部不剪枝；边界按 objective/call/memory/branch/novelty 桶保留 K=5，再从五个状态继续到闭合。H=3 保留为显式深度消融。
  - `exact`：固定 `max_rounds` 的完整展开，用于 exact-rN 对照。
  - `budgeted`：使用 candidate cap 和 beam，用于性能/消融对照。
- `rolling_windows.csv`：逐窗口记录多个 root、完整展开规模、open/closed 终态、K=5 选择、边界剪枝和 closure reason。
- `phasebatch/explorer.py`：单 pass state exploration。
- `phasebatch/batch_explorer.py`：pairwise batch exploration，不决定最终 objective winner。
- `phasebatch/pipeline_replay.py`：从 root 重放已提交 pipeline 并核对 hash。

## 评价、证据与可视化

- `phasebatch/baselines.py`：配置顺序、greedy、random 和 LLVM defaults 对照。
- `phasebatch/final_summary.py`：optimizer 最终摘要；rolling 模式显示窗口和闭合范围，不显示 beam 为主线控制。
- `phasebatch/reduction_summary.py`：state-local `n!` 到 executable batches 的 reduction。
- `phasebatch/component_summary.py`：overlap/conflict component 汇总。
- `phasebatch/equality_summary.py`：canonical/structural/different/failed tier 汇总。
- `phasebatch/evidence_pack.py`：selected/executed batch certificate 导出。
- `phasebatch/path_diagnostic.py`：selected path 与 baseline path 对比。
- `phasebatch/dag_visualizer.py`：state DAG 的 DOT/CSV/Markdown 与可选 SVG/PNG。

## Staged 与 Runtime

- `phasebatch/staged_config.py`：stage manifest、root IR mode、预算、required transition 与 runtime 参数。
- `phasebatch/staged_optimizer.py`：逐 stage 复用 optimizer，安全 handoff、聚合 pipeline、replay 和 cleanup。
- `phasebatch/runtime_rerank.py`：只在已安全到达且可 replay 的 terminal states 中做 codegen、cyclic trials、median/dispersion 排序。

## Worker 验收

- `phasebatch/worker_differential.py`：external 与 worker 的语义、hash、feature 和 structural comparison 差分。
- `phasebatch/worker_benchmark.py`：startup/no-op/single/pair/validation-shaped microbenchmark。
- `worker/phasebatch_worker.cpp`：C++ 常驻 LLVM New PM worker。
- `worker/CMakeLists.txt`：worker 构建定义。
- `scripts/build_worker.ps1`：本机 LLVM/CMake/Ninja 构建入口。

## Advisor Report

- `phasebatch/advisor_benchmarks.py`：SingleSource C discovery、clang smoke 与确定性选择。
- `phasebatch/advisor_metrics.py`：pair/equality/component/coverage/reduction/cost/cache/state-aware 聚合。
- `phasebatch/advisor_figures.py`：中文 PNG/SVG 图和 figure manifest。
- `phasebatch/advisor_markdown.py`：中文主报告、汇报话术、数据字典、关键数字和 metadata。
- `phasebatch/advisor_report.py`：正式 50 程序 rolling-exact 编排和离线 summarize。

旧 `outputs/advisor_report_zh_20programs` 是 budgeted depth-2/beam-4 pilot，
只能作为已有结构和成本证据，不能改标签为 rolling exact。

## Pass 配置

- `configs/core_passes_v1.yaml`：14-pass Core-v1。
- `configs/scalar_passes_v2.yaml`：staged scalar/memory/CFG pool。
- `configs/ipo_inline_v4.yaml`：staged IPO pool。
- `configs/loop_passes_v4.yaml`：staged loop pool。
- `configs/cleanup_passes_v4.yaml`：staged cleanup pool。
- `configs/vector_cleanup_passes_v5.yaml`：隔离 vector/cleanup pool。
- `configs/staged_salsa20_v5.yaml`：完整 Salsa20 staged manifest。
- `configs/staged_salsa20_v5_smoke.yaml`：小规模 staged smoke manifest。

## Tests

`tests/test_<module>.py` 通常与同名模块对应。关键跨模块测试：

- `tests/test_optimizer.py`：rolling window、闭合、incomplete、fixed exact 和 budgeted 回归。
- `tests/test_cli_pipeline.py`：CLI 默认值、worker session 与参数传播。
- `tests/test_advisor_*.py`：benchmark、metrics、图表、Markdown 与 offline summarize。
- `tests/test_opt_backend.py`、`tests/test_opt_worker.py`、`tests/test_worker_binary.py`：严格 worker。
- `tests/test_staged_*.py`、`tests/test_runtime_rerank.py`：stage handoff、aggregate replay 与 runtime selection。

正式多程序报告使用 `E:\llvm-test-suite\SingleSource`；快速回归使用
`benchmarks/tiny/*.c`。
