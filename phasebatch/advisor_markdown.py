from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


CORRECTNESS_BOUNDARY_ZH = (
    "本报告中的 commute、batch 和 reduction 结论均绑定到当前程序、当前 reached IR state、"
    "当前 LLVM 版本、当前 pass set、当前目标架构和当前 IR 等价性策略。Coarse footprint/overlap "
    "仅用于诊断，不构成独立性证明。只有经过动态 IR 等价性判断和 batch correctness classification "
    "允许的 batch 才能执行。目标函数与运行时间只用于评价，不用于证明 pass 可交换。"
)


def generate_advisor_markdown(study_dir: Path, *, metadata: dict | None = None) -> dict:
    study_dir = Path(study_dir)
    programs = _read_csv(study_dir / "program_summary.csv")
    pair_summary = _read_csv(study_dir / "pair_relation_summary.csv")
    overlap = _read_csv(study_dir / "overlap_component_program_summary.csv")
    conflict = _read_csv(study_dir / "conflict_component_program_summary.csv")
    small = _read_csv(study_dir / "small_overlap_cluster_abba.csv")
    unknown = _read_csv(study_dir / "unknown_failure_summary.csv")
    pair_failures = _read_csv(study_dir / "pair_failure_summary.csv")
    coverage = _read_csv(study_dir / "coverage_summary_all.csv")
    reduction = _read_csv(study_dir / "batch_reduction_program_summary.csv")
    cost = _read_csv(study_dir / "cost_breakdown_by_program.csv")
    top_passes = _read_csv(study_dir / "top_conflict_passes.csv")
    depth = _read_csv(study_dir / "state_aware_by_depth.csv")
    figures = {row.get("figure_id", ""): row for row in _read_csv(study_dir / "figures_manifest.csv")}
    missing = _read_csv(study_dir / "missing_outputs.csv")
    facts = _key_facts(programs, overlap, conflict, reduction, cost)

    report_lines = ["# Phasebatch 导师汇报数据报告", ""]
    if facts["dropped_active_passes"] > 0:
        report_lines.extend(["> **警告：发现 active pass 被静默丢弃**", ""])
    if any(_int(row.get("lazy_budget_skipped")) > 0 for row in unknown):
        report_lines.extend(["> **警告：full pair testing 报告中出现 lazy budget skipped，实验配置需要复核。**", ""])
    report_lines.extend(
        [
            "## 1. 本次要回答的问题", "",
            "1. 当前状态下可交换 commute pair 的比例是多少？",
            "2. overlap component 和 conflict component 的 mean / median / p90 / max 是多少？",
            "3. 小型 overlap cluster 中，AB == BA 的比例是多少？",
            "4. unknown、timeout、failure 是否少？",
            "5. dropped active passes 是否为 0？",
            "6. batch 后，局部搜索空间从 n! 缩小到多少？",
            "7. profiling、pair testing、IR equality、batch validation、search 各花多少时间？",
            "8. 哪些 pass 最常出现在大型 conflict component 中？",
            "9. IR state 改变后，active pass、enable/suppress、pair relation 是否变化？",
            "10. 能否生成 CGO 2006 风格的状态 DAG 图，直观展示搜索空间压缩？", "",
            "## 2. 实验配置", "",
            *_configuration_lines(metadata or {}, len(programs)), "",
            "正确性边界：本报告固定使用 pairwise 构造、full pair testing、auto batch validation，并且只执行 correctness classifier 允许的 batch。", "",
            "## 3. 程序级总览", "",
            *_table(programs, ["program", "states", "transitions", "avg_active_passes", "commute_ratio", "unknown_ratio", "dropped_active_passes", "total_wall_time_ms"], limit=30), "",
            "## 4. 当前状态下的可交换 Pair 比例", "",
            _figure(figures, "01"), "",
            *_table(pair_summary, ["program", "total_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "commute_ratio", "canonical_hash_commute", "structural_diff_commute"]), "",
            "这些结论只对 reached state 成立，不是全局 pass independence。报告中的比例应理解为“当前 reached IR state 上的可交换比例”。", "",
            _figure(figures, "09"), "",
            "## 5. Overlap / Conflict Component 分布", "",
            "Overlap component 和 conflict component 分开构造，前者描述 coarse footprint 相关性，后者描述不能安全折叠的关系。", "",
            _figure(figures, "02"), "",
            *_table(overlap, ["program", "mean_component_size", "median_component_size", "p90_component_size", "max_component_size", "singleton_ratio", "size_le_3_ratio", "size_ge_8_count"]), "",
            _figure(figures, "03"), "",
            *_table(conflict, ["program", "mean_component_size", "median_component_size", "p90_component_size", "max_component_size", "singleton_ratio", "size_le_3_ratio", "size_ge_8_count"]), "",
            "## 6. 小 Cluster 中的 AB == BA 比例", "",
            _figure(figures, "04"), "",
            *_table(small, ["size_bucket", "components", "internal_pairs", "tested_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "ab_ba_equal_ratio"]), "",
            "Overlap 只是风险/相关性诊断；真正的 commute 证据来自动态 AB/BA 和 IR equality。", "",
            "## 7. Unknown、Failure 与 Coverage", "",
            *_table(unknown, ["program", "pair_unknown", "pair_timeout", "pair_opt_failed", "comparator_failed", "max_pairs_skipped", "lazy_budget_skipped", "batch_validation_failed", "batch_unvalidated", "invalid_passes"]), "",
            "Pair failure kind 明细：", "",
            *_table(pair_failures, ["program", "failure_kind", "count", "percentage"]), "",
            *_table(coverage, ["program", "total_active_passes", "certified_covered", "unresolved_conflict", "validation_rejected", "failed_or_unknown", "terminal_due_max_depth", "dropped_active_passes"]), "",
            "缺失 coverage 数据显示为 N/A，不会被解释为 0。", "",
            "## 8. Batch 搜索空间压缩", "",
            _figure(figures, "05"), "",
            *_table(reduction, ["program", "avg_active_passes", "avg_naive_orderings_log10", "avg_batch_candidates", "avg_executable_batches", "avg_local_reduction_log10", "median_local_reduction_log10", "max_local_reduction_log10"]), "",
            "这是每个 reached state 上的局部 reduction；不能把不同 state 的 reduction 简单相乘后声称全局搜索空间。", "",
            "## 9. 成本和工程开销", "",
            _figure(figures, "06"), "",
            *_table(cost, ["program", "total_wall_time_ms", "profiling_wall_ms", "pair_testing_wall_ms", "batch_validation_wall_ms", "state_search_wall_ms", "other_wall_ms", "ir_equality_cumulative_work_ms"]), "",
            *_bottleneck_lines(cost), "",
            "wall-clock 与 cumulative work 使用不同字段。并行任务的累计耗时不会被求和后冒充 wall-clock。", "",
            "## 10. 最常出现在大型 Conflict Component 的 Pass", "",
            _figure(figures, "07"), "",
            *_table(top_passes, ["pass", "programs_present", "states_present", "non_singleton_component_memberships", "max_component_size", "order_sensitive_degree", "unknown_degree", "weighted_conflict_score"], limit=20), "",
            "该排名表示 pass 经常出现在大型冲突结构中，不等于它是冲突的单一因果来源。", "",
            "## 11. State-aware 变化", "",
            _figure(figures, "08"), "",
            *_table(depth, ["program", "depth", "states", "avg_active_passes", "avg_commute_ratio", "enable_count", "suppress_count", "effect_changed_count", "true_relation_flip_count", "pair_availability_change_count"]), "",
            "True relation flip 只统计 pair 在父子状态均可用时的关系变化；仅一侧存在的 pair 单独记为 pair availability change。", "",
            "## 12. DAG 可视化", "",
            *_dag_lines(study_dir), "",
            "节点是 canonical IR state；边是经过 correctness classifier 允许的 batch transition；多条路径到达相同 canonical state 时合并为 DAG；DAG 图本身不产生新的 correctness 证据。", "",
            "## 13. 面向导师的初步结论", "",
            *_conclusion_lines(facts), "",
            "## 14. 限制", "",
            "- 数据只针对当前 LLVM 版本。",
            "- 数据只针对当前 pass set。",
            "- Pair relation 是 state-local。",
            "- Coarse overlap 只是 diagnostic，不是独立性证明。",
            "- Objective 不是 commutation proof。",
            "- 当前报告不主张 runtime 优于 O2/O3。",
            "- exact / budgeted 各有范围边界。", "",
            CORRECTNESS_BOUNDARY_ZH, "",
        ]
    )
    report_path = study_dir / "advisor_report_zh.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    talking_path = study_dir / "advisor_talking_points_zh.md"
    talking_path.write_text(_talking_points(facts), encoding="utf-8")
    dictionary_path = study_dir / "data_dictionary_zh.md"
    dictionary_path.write_text(_data_dictionary(), encoding="utf-8")
    key_rows = _key_number_rows(facts, len(programs), len(missing))
    key_path = study_dir / "advisor_key_numbers.csv"
    _write_csv(key_path, ["metric", "value", "unit", "scope", "description_zh"], key_rows)
    metadata_payload = dict(metadata or {})
    metadata_payload.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "programs_summarized": len(programs),
            "missing_outputs": len(missing),
            "correctness_boundary_zh": CORRECTNESS_BOUNDARY_ZH,
        }
    )
    metadata_path = study_dir / "report_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_program_reports(study_dir, programs, pair_summary, overlap, conflict, reduction, cost)
    return {
        "advisor_report_zh": str(report_path),
        "advisor_talking_points_zh": str(talking_path),
        "data_dictionary_zh": str(dictionary_path),
        "advisor_key_numbers_csv": str(key_path),
        "report_metadata_json": str(metadata_path),
    }


def _key_facts(programs: list[dict], overlap: list[dict], conflict: list[dict], reduction: list[dict], cost: list[dict]) -> dict:
    commute_values = [_float(row.get("commute_ratio")) for row in programs if row.get("commute_ratio") not in {None, ""}]
    small_values = [_float(row.get("size_le_3_ratio")) for row in overlap if row.get("size_le_3_ratio") not in {None, ""}]
    validation = sum(_float(row.get("batch_validation_wall_ms")) for row in cost)
    total = sum(_float(row.get("total_wall_time_ms")) for row in cost)
    return {
        "median_commute_ratio": statistics.median(commute_values) if commute_values else 0.0,
        "overlap_size_le_3_ratio": statistics.mean(small_values) if small_values else 0.0,
        "max_conflict_component_size": max((_int(row.get("max_component_size")) for row in conflict), default=0),
        "dropped_active_passes": sum(_int(row.get("dropped_active_passes")) for row in programs if row.get("dropped_active_passes") not in {None, ""}),
        "batch_validation_share": validation / total if total else 0.0,
        "avg_local_reduction_log10": statistics.mean([_float(row.get("avg_local_reduction_log10")) for row in reduction]) if reduction else 0.0,
        "total_wall_time_ms": total,
    }


def _conclusion_lines(facts: dict) -> list[str]:
    lines = []
    if facts["median_commute_ratio"] >= 0.5:
        lines.append("- 在本次 reached states 中，超过一半的 active pair 可被当前证据判为 commute，说明存在显著的局部顺序折叠机会。")
    else:
        lines.append("- 本次 reached states 的 commute 比例中位数低于 50%，局部折叠机会存在但并不占主导。")
    if facts["overlap_size_le_3_ratio"] >= 0.7:
        lines.append("- 多数 interaction component 较小，适合局部动态验证。")
    if facts["max_conflict_component_size"] >= 8:
        lines.append("- 部分状态存在较大冲突组件，需要进一步分析其来源或细化 effect 粒度。")
    if facts["dropped_active_passes"] == 0:
        lines.append("- 本次数据中没有 active pass 被静默丢弃。")
    else:
        lines.append(f"- 本次数据记录到 {facts['dropped_active_passes']} 个 dropped active pass，必须先处理覆盖缺口再使用 reduction 结论。")
    if facts["batch_validation_share"] > 0.5:
        lines.append("- 当前主要工程瓶颈是 batch validation，而不是搜索 frontier 管理。")
    return lines


def _bottleneck_lines(rows: list[dict]) -> list[str]:
    fields = [
        ("profiling", "profiling_wall_ms"), ("pair testing", "pair_testing_wall_ms"),
        ("batch validation", "batch_validation_wall_ms"), ("search", "state_search_wall_ms"),
        ("其他", "other_wall_ms"),
    ]
    totals = {label: sum(_float(row.get(field)) for row in rows) for label, field in fields}
    total = sum(_float(row.get("total_wall_time_ms")) for row in rows)
    if not totals or total <= 0:
        return ["当前缺少足够的 wall-clock 数据，无法自动判定主要瓶颈。"]
    label, value = max(totals.items(), key=lambda item: item[1])
    return [f"本次运行中，{label} 占总 wall-clock 的 {value / total:.1%}，是当前记录中占比最大的阶段。"]


def _configuration_lines(metadata: dict, benchmark_count: int) -> list[str]:
    backend = metadata.get("opt_backend", "worker（严格模式）")
    if isinstance(backend, dict):
        stats = backend.get("stats", {}) if isinstance(backend.get("stats"), dict) else {}
        backend_text = (
            f"{backend.get('backend', 'worker')}（严格模式），workers={backend.get('workers', 'N/A')}，"
            f"restarts={stats.get('restarts', 'N/A')}，LLVM fatal={stats.get('llvm_fatal_failures', 'N/A')}，"
            f"backend failures={stats.get('backend_failures', 'N/A')}，fallbacks={stats.get('fallbacks', 'N/A')}"
        )
    else:
        backend_text = str(backend)
    return [
        f"- LLVM/opt 版本：{_version_line(metadata.get('llvm_opt_version', metadata.get('llvm_version', 'N/A')))}",
        f"- Benchmark 数：{metadata.get('benchmark_count', benchmark_count)}",
        f"- Pass set：{metadata.get('pass_config', 'N/A')}",
        f"- 搜索模式：{metadata.get('mode', 'N/A')}",
        f"- Max rounds：{metadata.get('max_rounds', 'N/A')}",
        f"- Pair mode：{metadata.get('pair_testing_mode', 'full')}",
        f"- Batch construction：{metadata.get('batch_construction_mode', 'pairwise')}",
        f"- Validation mode：{metadata.get('batch_validation_mode', 'auto')}",
        f"- LLVM 后端：{backend_text}",
    ]


def _write_program_reports(study_dir: Path, programs: list[dict], pairs: list[dict], overlap: list[dict], conflict: list[dict], reduction: list[dict], cost: list[dict]) -> None:
    tables = {
        "Pair 关系": (pairs, ["total_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "commute_ratio"]),
        "Overlap component": (overlap, ["mean_component_size", "median_component_size", "p90_component_size", "max_component_size"]),
        "Conflict component": (conflict, ["mean_component_size", "median_component_size", "p90_component_size", "max_component_size"]),
        "局部 reduction": (reduction, ["avg_naive_orderings_log10", "avg_executable_batches", "avg_local_reduction_log10", "max_local_reduction_log10"]),
        "成本": (cost, ["total_wall_time_ms", "profiling_wall_ms", "pair_testing_wall_ms", "batch_validation_wall_ms", "other_wall_ms"]),
    }
    for program_row in programs:
        program = program_row.get("program", "")
        if not program:
            continue
        lines = [f"# {program} Phasebatch 中文中间报告", "", "## 程序概况", "", *_table([program_row], ["states", "transitions", "avg_active_passes", "final_pipeline_length", "final_ir_inst_count", "dropped_active_passes"]), ""]
        for title, (rows, fields) in tables.items():
            selected = [row for row in rows if row.get("program") == program]
            lines.extend([f"## {title}", "", *_table(selected, fields), ""])
        lines.extend(["## 正确性边界", "", CORRECTNESS_BOUNDARY_ZH, ""])
        path = study_dir / "programs" / program / "advisor_program_summary_zh.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


def _version_line(value: object) -> str:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if not lines:
        return "N/A"
    return next((line for line in lines if "version" in line.lower()), lines[0])


def _figure(figures: dict[str, dict], figure_id: str) -> str:
    row = figures.get(figure_id, {})
    path = row.get("svg_path") or row.get("png_path")
    if not path:
        return f"> 图 {figure_id}：N/A"
    return f"![{row.get('title_zh', '图 ' + figure_id)}]({path})"


def _dag_lines(study_dir: Path) -> list[str]:
    paths = sorted((study_dir / "dags").glob("*/*.svg")) if (study_dir / "dags").is_dir() else []
    if not paths:
        dots = sorted((study_dir / "dags").glob("*/*.dot")) if (study_dir / "dags").is_dir() else []
        return ["> DAG SVG：N/A（Graphviz 不可用时保留 DOT）。", *[f"- `{path.relative_to(study_dir)}`" for path in dots[:10]]]
    return [f"![{path.parent.name} 状态 DAG]({path.relative_to(study_dir).as_posix()})" for path in paths[:9]]


def _table(rows: list[dict], fields: list[str], limit: int = 20) -> list[str]:
    if not rows:
        return ["N/A"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_cell(row.get(field, "")) for field in fields) + " |")
    return lines


def _talking_points(facts: dict) -> str:
    return "\n".join(
        [
            "# Phasebatch 导师汇报话术（约 5 分钟）", "",
            "## 30 秒：问题", "", "我们要回答的不是某个固定 pass 是否永远独立，而是在实际搜索到达的 IR 状态上，有多少局部顺序可以安全折叠，以及这种折叠是否真的减少了验证和搜索规模。", "",
            "## 60 秒：方法", "", "每个程序都使用相同 pass set，先 profiling，再做 full AB/BA pair testing，并通过 IR equality 和 batch correctness classifier 决定 batch 是否能执行。Overlap 只用于解释风险结构，不参与正确性证明。", "",
            "## 60 秒：Commute 和 Component 数据", "", f"当前 reached states 的 commute 比例中位数约为 {facts['median_commute_ratio']:.1%}。最大的 conflict component 为 {facts['max_conflict_component_size']} 个 pass。重点是这些数都是 state-local，而不是全局 pass independence。", "",
            "## 60 秒：Batch Reduction", "", f"平均局部 reduction log10 约为 {facts['avg_local_reduction_log10']:.2f}。这个数比较每个 reached state 的 n! 与可执行 batch 数，不能跨状态直接相乘。", "",
            "## 60 秒：成本与限制", "", f"Batch validation 占已记录总 wall-clock 的约 {facts['batch_validation_share']:.1%}。报告不主张生成程序优于 O2/O3，也不把运行时间或目标函数当作可交换证明。", "",
            "## 30 秒：下一步", "", "下一步优先检查大型 conflict component 的来源、unknown/failure 个案和 validation 成本，再决定是否细化 effect 粒度或增加缓存，而不是放宽 correctness 边界。", "",
        ]
    )


def _data_dictionary() -> str:
    entries = [
        ("active pass", "在当前 IR state 上成功运行并改变规范化 IR 的 pass。"),
        ("dormant pass", "在当前 IR state 上成功运行但没有改变规范化 IR 的 pass。"),
        ("commute", "动态执行 AB 与 BA 后，经当前 IR equality 策略证明结果等价。"),
        ("order-sensitive", "AB 与 BA 的结果在当前证据下不同。"),
        ("unknown", "由于失败、超时、比较器失败、缺失关系或范围限制，无法得到安全的 commute 结论。"),
        ("overlap component", "由 coarse footprint overlap 边连接得到的诊断组件，不构成 correctness proof。"),
        ("conflict component", "由 order-sensitive、unknown 或 missing pair relation 边连接得到的组件。"),
        ("certified batch", "经 batch validation 和 correctness classifier 允许执行的 batch。"),
        ("dropped active pass", "当前 active pass 没有被任何可执行/已分类覆盖路径解释的覆盖缺口。"),
        ("local reduction log10", "当前 reached state 上 log10(n!) 减去 log10(max(1, executable batches))。"),
        ("true relation flip", "同一 pair 在父子状态都可用，但 final relation 发生已定义变化。"),
        ("pair availability change", "同一 pair 只在父状态或子状态一侧可用，不算 true relation flip。"),
        ("wall-clock", "从阶段开始到结束的实际经过时间；并行工作不会重复累加。"),
        ("cumulative work", "多个任务各自耗时的累计量，可超过 wall-clock。"),
    ]
    lines = ["# Phasebatch 导师报告数据字典", "", "Percentile 使用确定性的 nearest-rank 方法：排序后取 `ceil(q * n)` 的 1-based 位置。", ""]
    for term, description in entries:
        lines.extend([f"## {term}", "", description, ""])
    return "\n".join(lines)


def _key_number_rows(facts: dict, programs: int, missing: int) -> list[dict]:
    return [
        {"metric": "programs", "value": str(programs), "unit": "programs", "scope": "study", "description_zh": "成功进入汇总的程序数"},
        {"metric": "median_commute_ratio", "value": f"{facts['median_commute_ratio']:.6f}", "unit": "ratio", "scope": "reached_states", "description_zh": "程序级当前状态可交换比例的中位数"},
        {"metric": "max_conflict_component_size", "value": str(facts["max_conflict_component_size"]), "unit": "passes", "scope": "reached_states", "description_zh": "最大 conflict component"},
        {"metric": "dropped_active_passes", "value": str(facts["dropped_active_passes"]), "unit": "passes", "scope": "study", "description_zh": "被静默丢弃的 active pass 数"},
        {"metric": "batch_validation_share", "value": f"{facts['batch_validation_share']:.6f}", "unit": "ratio", "scope": "wall_clock", "description_zh": "Batch validation 的 wall-clock 占比"},
        {"metric": "missing_outputs", "value": str(missing), "unit": "files", "scope": "study", "description_zh": "缺失或空的预期产物数"},
    ]


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cell(value: object) -> str:
    text = " ".join(str(value or "N/A").splitlines()).replace("|", "\\|")
    return text or "N/A"


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0
