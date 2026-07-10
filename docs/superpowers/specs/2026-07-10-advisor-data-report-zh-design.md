# Advisor Data Report v1 设计说明

## 目标

在不修改 pair correctness、IR Equality、batch classifier、搜索语义、状态合并、batch 构造和 pass set 的前提下，增加一层可复现的多程序实验编排与中文汇报系统。系统既能从 LLVM test-suite 发现并运行 10--20 个 SingleSource C 程序，也能只读取已有 study 目录重建全部 CSV、图表、DAG 和 Markdown。

## 正确性与执行边界

- 数据采集固定使用 `pairwise + full pair testing + auto batch validation + validate_batches`。
- LLVM 执行后端固定继承项目正式默认：严格 `worker`，基础设施失败立即报错且不回退。
- overlap graph 仅由 footprint 信息构造，是诊断图；commute 只读取动态 AB/BA 与 IR equality 的最终关系。
- 缺失产物写入 `missing_outputs.csv`；数值使用空字符串表示 N/A，绝不转换为 0。
- wall-clock 与 cumulative work 分列，不能把并行任务耗时求和冒充 wall-clock。
- 所有输出按 program、depth、state ID、component ID、pass 名确定性排序。

## 模块边界

### `phasebatch/advisor_benchmarks.py`

负责 manifest 读取、SingleSource C 扫描、source-size 限制、clang smoke 编译、跨目录确定性选择，以及三个 benchmark 清单文件。该模块不调用 optimizer。

### `phasebatch/advisor_metrics.py`

负责读取一个或多个既有 optimize 目录，必要时调用现有 `build_footprint_overlap()` 重建可推导的 footprint CSV，并生成全部 advisor 聚合 CSV。它包含通用 CSV/数值/percentile/connected-component helper，但不复制 optimizer 或 correctness 算法。

percentile 使用确定性的 nearest-rank 定义：排序后选择 `ceil(0.90 * n)`（1-based）位置，并在 `data_dictionary_zh.md` 中记录。

### `phasebatch/advisor_figures.py`

只读取聚合 CSV，使用 matplotlib 生成中文 PNG/SVG 和 `figures_manifest.csv`。字体按 Microsoft YaHei、SimHei、Noto Sans CJK SC、Arial Unicode MS 依次选择；缺失时回退并记录 warning。

### `phasebatch/advisor_markdown.py`

只读取聚合结果和 manifest，生成保守的中文主报告、5 分钟话术、数据字典、关键数字和 report metadata。固定正确性边界原样写入主报告。

### `phasebatch/advisor_report.py`

提供 `run_advisor_report_zh()` 与 `summarize_advisor_report_zh()`。前者调用现有 `optimize_batches()` Python API，保持一个 CLI 级 worker pool 跨程序复用；后者绝不调用 LLVM。每个程序输出到 `OUT/programs/<program>/optimize/`。

### `phasebatch/cli.py`

仅新增两个 parser 和薄 dispatch wrapper。运行命令暴露 benchmark、搜索和稳定主线参数；汇总命令只接收 study dir。

## 数据兼容策略

解析器优先使用当前标准文件：`states.csv`、`state_dag.csv`、`chosen_path.csv`、state 下的 `pass_profile.csv`、`pair_relation.csv`、`batch_*`、`coverage_summary.csv`、顶层 `optimizer_timing.csv`、pair/equality/validation summary。旧运行缺列时使用 N/A，并记录缺失文件或字段。

component 节点始终包含全部 active passes，因此 singleton 不会因无边而消失。conflict edge 包含 order-sensitive、unknown 和 missing relation；commute 不连边。overlap edge仅包含 same-function、same-block 和 possible-WW，unknown-overlap 单独统计。

## 图表与 DAG

九组图均由 CSV 驱动并同时输出 PNG/SVG。量纲不同的 state-aware 指标拆成 `08a` 与 `08b`，同时在 figure manifest 中归属于图 08。DAG 复用 `visualize_dag()`；最多选择 reduction 最大、conflict component 最大、states 最多的三个去重程序。Graphviz 不可用时保留 DOT 并记录 warning。

## Resume 与失败

- `--resume` 仅在 optimize 目录同时存在非空 `optimize_summary.md`、`states.csv` 和必要 state 目录时复用。
- `--overwrite` 删除的范围仅限明确指定的 study output 目录；实现前验证 resolved path。
- `--continue-on-error` 只隔离 benchmark 编译/优化失败；严格 worker 基础设施失败仍终止整个 study，避免生成混合后端性能数据。
- 每次汇总都是幂等操作，可覆盖 advisor 自己生成的 CSV/图表/Markdown，但不删除 optimize 原始产物。

## 测试策略

先以合成小型 run 目录测试所有纯函数和缺失值语义，再测试 chart/DAG 降级，最后 mock optimizer 验证 run/summarize 分离。全量测试通过后运行 3-program strict-worker smoke，并检查验收清单和 `.ll`/空目录清理结果。
