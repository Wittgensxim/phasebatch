# Phasebatch 主线精简设计

## 目标

把仓库收敛到当前正式主线：full pairwise、exact/budgeted 搜索、validation
DAG、correctness classifier、strict in-process LLVM worker、staged/runtime
rerank 和 Advisor Report。删除已经放弃的 CEGAR、被 Advisor Report 取代的
一次性 study wrappers，以及不再用于当前证据链的历史输出和本地杂项。

## 保留范围

- 单状态分析、pair testing、batch 构造、batch validation 和状态 DAG。
- `optimize-batches` 的 exact/budgeted 模式与 full/lazy pair testing。
- `optimize-staged`、runtime top-K rerank 和 aggregate replay。
- strict worker、worker differential 和 worker benchmark。
- Advisor Report 的发现、执行、聚合、图表、DAG 和中文报告。
- 当前 run 的 reduction/component/evidence/DAG/replay/baseline 只读工具。
- `core_passes_v1.yaml` 和 staged v2/v4/v5 所需配置。

## 删除范围

### 算法和 CLI

- 删除 `cegar_batcher.py`、`cegar_comparison.py` 和所有 CEGAR CLI/options。
- `--batch-construction-mode` 只接受 `pairwise`，继续写入 metadata。
- 删除旧多程序/消融 wrappers：mainline、method comparison、round/budgeted
  sensitivity、reduction study、Core-v1 study/reference、v2/v3 study、passset
  smoke/summary 和旧 case-study exporter。
- 删除这些模块的专属测试和 CLI 测试段；保留底层能力测试。

### 配置、脚本和文档

- 删除 legacy `core_passes.yaml` 和不再使用的 `middleend_passes_v3.yaml`。
- 删除早期 MVP shell/python 脚本，仅保留 worker build 脚本。
- 删除 CEGAR plan/spec；当前 README、project status、项目逻辑和文件角色说明
  不再宣称被删除的命令可用。
- 保留 staged 所需 `scalar_passes_v2.yaml`、v4/v5 configs 和对应设计记录。

### 磁盘产物

`outputs/` 只保留：

- `advisor_report_zh_20programs`
- `runtime_matched_salsa20_worker_v6_20260710`
- `worker_benchmark_salsa20_final_20260710`
- `verify_opt_worker_salsa20_core_fresh_context`
- `verify_opt_worker_salsa20_middleend_fresh_context_r3`
- `salsa20_staged_v5_fixed_20260710`
- `verify_validation_dag_salsa20_exact`
- `.gitkeep`

删除其他历史 run、root debug logs/pointers、pytest caches、Python bytecode、
`scripts.zip`、旧任务文本和本地论文副本。保留 `worker/build/phasebatch-worker.exe`
及其可重建构建目录，因为 strict worker 是默认执行后端。

## 安全策略

- 删除前将每个路径解析为绝对路径，确认位于 `E:\PO2` 的预期目录内。
- 不修改或删除保留的 20 程序报告。
- 不使用 Git reset/checkout，也不回滚其他已有修改。
- 先完成代码引用清理并通过 focused tests，再执行批量输出删除。
- 最后运行完整 pytest、CLI help smoke、CEGAR/legacy symbol scan、保留目录
  清单和 Advisor 产物完整性检查。

## 完成标准

- 源码、测试、README 和当前文档中没有 CEGAR 或已删除 study command 引用。
- CLI 仅暴露保留的主线命令。
- 完整测试通过。
- `outputs/` 只含白名单目录和 `.gitkeep`。
- Advisor Report 仍有 20 个成功程序、9 组图、0 missing outputs、0 `.ll`、
  0 空目录。
