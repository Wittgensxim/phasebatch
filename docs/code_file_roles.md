# 当前代码文件作用说明

本文只描述主线精简后仍保留的文件。当前正式路径是 full pairwise、
reuse/cache、validation DAG、strict worker、exact/budgeted、staged/runtime
rerank 和 Advisor Report。

## 入口与配置

- `phasebatch/__main__.py`：`python -m phasebatch` 入口。
- `phasebatch/cli.py`：维护中的 argparse 命令、参数传递和 worker session
  边界。
- `phasebatch/config.py`：legacy/simple YAML 列表解析辅助。
- `phasebatch/pass_config.py`：结构化 pass registry、pipeline candidates、
  category/stage 和 canonical order。
- `phasebatch/schema.py`：当前 CSV 字段定义。
- `phasebatch/tools.py`：LLVM 工具发现、版本、target triple 和 metadata。

## LLVM 执行与 IR

- `phasebatch/runner.py`：准备输入 IR、外部 opt 基线、worker 路由和 deferred
  result materialization。
- `phasebatch/opt_worker.py`：常驻 worker 进程、JSON 协议、restart 和 pool。
- `phasebatch/opt_backend.py`：worker handles、path cache、apply/materialize、
  in-process structural comparison 和严格故障策略。
- `phasebatch/normalizer.py`：canonical IR hash 和静态 IR feature counts。
- `phasebatch/ir_equivalence.py`：canonical hash、LLVMDiff 和 module safety
  fingerprint 等价性阶梯。
- `phasebatch/ir_parser.py`：函数、basic block 和 instruction 的轻量 IR 解析。
- `phasebatch/artifact_cleanup.py`：普通运行结束后的 `.ll` 与空目录清理，
  以及显式 keep marker。

## State、Pair 与关系

- `phasebatch/state_analysis.py`：单个 reached state 的 profiling、完整/预算
  pair testing、关系、footprint、cost 和 summary 编排。
- `phasebatch/profiler.py`：单 pass 有效性、activity、effect footprint 和
  reusable output。
- `phasebatch/pair_tester.py`：AB/BA 动态测试、single-pass reuse、hash fast
  path、IR equality 和保守 unknown/failed 分类。
- `phasebatch/pair_cache.py`：state-local pair result memoization。
- `phasebatch/pair_cost.py`：opt runs、pass invocations、cache 和 comparator 成本。
- `phasebatch/pair_scheduling.py`：full/lazy pair 顺序与预算统计；正式报告使用
  full。
- `phasebatch/relation.py`：静态 footprint 与动态 equality 合并成 final
  relation。
- `phasebatch/footprint.py`：coarse overlap diagnostic，不是 correctness proof。
- `phasebatch/graph.py`：active pass interaction/cluster 辅助。
- `phasebatch/report.py`：单状态和聚合分析摘要。

## Batch 构造与验证

- `phasebatch/batcher.py`：pairwise conflict graph、connected components、
  maximal independent sets、candidate 组合和 batch validation 调度。
- `phasebatch/batch_validation_dag.py`：确定性 permutation DAG、prefix/state
  合并、transition cache 和 equivalence cache。
- `phasebatch/validation_runtime.py`：state-local validation worker queue、
  single-flight 和跨 candidate cache。
- `phasebatch/batch_validation_ladder.py`：exhaustive/DAG/bounded/sampled
  validation summary。
- `phasebatch/batch_correctness.py`：validation status 到 correctness class、
  `can_hard_fold` 和 `can_execute` 的唯一分类边界。
- `phasebatch/coverage.py`：active pass coverage、unresolved/rejected/terminal 和
  dropped 检查。
- `phasebatch/batch_objective.py`：已有 transition 的 objective evaluation。

## 搜索与选择

- `phasebatch/explorer.py`：单 pass state exploration。
- `phasebatch/batch_explorer.py`：pairwise batch exploration，不负责最终
  objective winner。
- `phasebatch/optimizer.py`：exact/budgeted 主优化器、state DAG、frontier、
  selected path、final replay 和 cleanup。
- `phasebatch/baselines.py`：配置顺序、greedy、random、LLVM defaults 等当前
  run 的对照评价。
- `phasebatch/pipeline_replay.py`：从 root 重放 selected pipeline 并核对 hash。
- `phasebatch/final_summary.py`：optimizer 最终摘要。
- `phasebatch/reduction_summary.py`：state-local n! 到 executable batch 的
  log10 reduction。
- `phasebatch/component_summary.py`：overlap/conflict component 汇总。
- `phasebatch/equality_summary.py`：canonical/structural/different/failed tier
  汇总。
- `phasebatch/evidence_pack.py`：selected/executed batch certificate 导出。
- `phasebatch/path_diagnostic.py`：selected path 与当前 baseline paths 对比。
- `phasebatch/dag_visualizer.py`：state DAG DOT/CSV/Markdown 和可选 SVG/PNG。

## Staged 与 Runtime

- `phasebatch/staged_config.py`：顺序 stage manifest、root IR mode、stage 预算、
  required transition 和 runtime 参数。
- `phasebatch/staged_optimizer.py`：逐 stage 复用 `optimize_batches`，安全 handoff、
  aggregate pipeline、replay 和 cleanup。
- `phasebatch/runtime_rerank.py`：只在已安全到达且 replayable 的 terminal
  states 中做 codegen、cyclic trials、median/dispersion 排序。

## Worker 验收

- `phasebatch/worker_differential.py`：external 与 worker 的语义、hash、feature
  和 structural comparison 差分验证。
- `phasebatch/worker_benchmark.py`：startup/no-op/single/pair/validation-shaped
  worker microbenchmark。
- `worker/phasebatch_worker.cpp`：C++ 常驻 LLVM New PM worker。
- `worker/CMakeLists.txt`：worker 构建定义。
- `scripts/build_worker.ps1`：本机 LLVM/CMake/Ninja 构建入口。

## Advisor Report

- `phasebatch/advisor_benchmarks.py`：SingleSource C discovery、source cap、clang
  smoke、确定性选择和 resume manifest。
- `phasebatch/advisor_metrics.py`：pair/equality/component/coverage/reduction/cost/
  cache/state-aware 聚合 CSV。
- `phasebatch/advisor_figures.py`：九组中文 PNG/SVG 图和 figure manifest。
- `phasebatch/advisor_markdown.py`：中文主报告、五分钟话术、数据字典、关键数字
  和 metadata。
- `phasebatch/advisor_report.py`：20 程序执行编排及只读 summarize 入口。

## Tests

`tests/test_<module>.py` 通常与同名主模块一一对应。跨模块重点如下：

- `tests/test_cli_bootstrap.py`：只允许当前命令出现在 CLI help，并确认已删除
  命令不再暴露。
- `tests/test_cli_pipeline.py`：当前 CLI 参数、worker session 和 wrapper 参数
  传播。
- `tests/test_architecture_imports.py`：core modules 不反向依赖 CLI。
- `tests/test_optimizer.py`、`tests/test_batch_explorer.py`：只允许 pairwise
  construction。
- `tests/test_opt_backend.py`、`tests/test_opt_worker.py`、
  `tests/test_worker_binary.py`：严格 worker 生命周期和真实二进制行为。
- `tests/test_advisor_*.py`：benchmark、metrics、图表、中文 Markdown 和
  offline summarize。
- `tests/test_staged_*.py`、`tests/test_runtime_rerank.py`：stage handoff、aggregate
  replay 和安全 runtime selection。

## Pass 配置

- `configs/core_passes_v1.yaml`：当前 14-pass Core-v1。
- `configs/scalar_passes_v2.yaml`：staged scalar/memory/CFG pool。
- `configs/ipo_inline_v4.yaml`：staged IPO pool。
- `configs/loop_passes_v4.yaml`：staged loop pool。
- `configs/cleanup_passes_v4.yaml`：staged cleanup pool。
- `configs/vector_cleanup_passes_v5.yaml`：隔离 vector/cleanup pool。
- `configs/staged_salsa20_v5.yaml`：完整 Salsa20 staged manifest。
- `configs/staged_salsa20_v5_smoke.yaml`：小规模 staged smoke manifest。

## Tiny Benchmarks

`benchmarks/tiny/*.c` 提供 arithmetic、branch、loop、memory 和 select 的快速
回归输入。正式多程序报告使用 `E:\llvm-test-suite\SingleSource`。
