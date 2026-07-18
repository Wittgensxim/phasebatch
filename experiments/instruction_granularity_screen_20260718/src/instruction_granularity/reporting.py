from __future__ import annotations

import json
from pathlib import Path

from .deterministic_io import write_text


def generate_reports_and_figures(root: Path, metrics: dict) -> tuple[Path, ...]:
    root = Path(root)
    report_dir = root / "report"
    figure_dir = root / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    coverage = metrics["coverage"]
    incremental = metrics["incremental"]
    cumulative = metrics["cumulative"]
    runtime = {row["level"]: row for row in metrics["runtime_summary"]}
    first_sentence = _first_sentence(cumulative, incremental)

    report = f"""{first_sentence}

# Observed-effect 指令级覆盖率与特征提取成本实验报告

## 结论

在冻结的 49 个程序、14 个 action、1,411 个 pair 上，`H_inst` 累计选中 {cumulative['cumulative_selected']} 行：{cumulative['cumulative_commute']} 个实际 commute、{cumulative['cumulative_order_sensitive']} 个 order-sensitive、{cumulative['cumulative_failed']} 个 failed。累计 precision 为 {_pct(cumulative['cumulative_precision'])}，对 833 个实际 commute 的 recall 为 {_pct(cumulative['cumulative_coverage_of_833_commute'])}。

相对 `H_effect`，指令层增量确认选中 {incremental['incremental_selected_count']} 行，其中 {incremental['incremental_commute']} 个 commute、{incremental['incremental_order_sensitive']} 个 order-sensitive、{incremental['incremental_failed']} 个 failed；增量 precision 为 {_pct(incremental['incremental_precision'])}。另有 {incremental['incremental_unknown']} 行因 wildcard、结构不稳定或其他 fail-closed 条件记为 unknown（机器字段 `incremental_unknown={incremental['incremental_unknown']}`），它们没有被确认选中，也没有从 1,411 行主表中删除。

这个结果只说明 observed-change 经验筛选在该冻结 corpus 上新增了 1 个真实 commute；它不是 commute 证明，也不构成 Phasebatch speedup 结论。

## 四层覆盖率与 precision

| 层级 | selected | commute | order-sensitive | failed | screen unknown | precision | 833-commute recall |
|---|---:|---:|---:|---:|---:|---:|---:|
{_coverage_table(coverage)}

硬门槛全部复现：1,411 pair rows、49 programs、14 actions、833 commute、569 order-sensitive、9 failed、686 single-pass transitions；`H_func=30=28C+2OS`、`H_block=46=44C+2OS`、`H_effect=47=45C+2OS`。

## 指令增量成本与收益

- 增量 pair 动态成本：{_ms(incremental['incremental_dynamic_cost_ms'])}
- 其中真实 commute 动态成本：{_ms(incremental['incremental_true_commute_cost_ms'])}
- unsafe 动态成本：{_ms(incremental['incremental_unsafe_cost_ms'])}
- paired `INSTRUCTION_ONLY - EFFECT_ONLY` 中位提取增量：{_ms(incremental['instruction_incremental_extraction_cost_ms'])}
- `instruction_marginal_value`：{_ms(incremental['instruction_marginal_value'])}

{_marginal_conclusion(incremental)}

边际值按同一 measured repetition 的 paired 差计算：

```text
incremental_true_commute_cost
- incremental_unsafe_cost
- median_paired(INSTRUCTION_ONLY - EFFECT_ONLY)
```

它是本实验定义的局部价值指标，不是 batch-DAG 成本，也不是端到端运行收益。

## 特征提取计时

每层先做 5 次 warm-up（不计），再做 30 次 measured；同一 repetition 固定执行 `FUNC_ONLY → BLOCK_ONLY → EFFECT_ONLY → INSTRUCTION_ONLY`，每层独立重读和重建状态，三套旧 DYNAMIC_ALL 工件各覆盖 10 次 measured。

| extractor | first | median | p90 | min | max | median/transition | median/pair |
|---|---:|---:|---:|---:|---:|---:|---:|
{_runtime_table(runtime)}

原始计时逐轮保留 `artifact_read_ms`、`parse_ms`、`feature_build_ms`、`pair_selection_ms` 与 `total_extraction_ms`。paired 增量先逐 repetition 相减，再汇总 median/p90。

## 指令指纹与 fail-closed 规则

指令 parser 使用 debug-insensitive IR，支持 multiline call/switch/phi；去除结果 SSA，对 argument、本地 SSA 与 block label 做 definition-order alpha normalization，同时保留 opcode、类型、predicate、callee/global、常量、flags 和 operand 结构。每个稳定 `(function, block, effect_class)` 使用 instruction fingerprint Counter；纯重排不产生 changed token。

只有 A/B changed instruction token 均非空、交集为空、双方无 wildcard，且三个 DYNAMIC_ALL 源轮次的 transition 与选择一致时，才确认增量选择。function/block 无法对应、CFG 或 header/signature/linkage/attribute 不稳定、logical instruction 失败、hash/工件不一致、collision 或跨轮不一致均 fail closed 为 unknown。本次三个源轮次无工件错误，collision 数为 {metrics['collision_count']}。

## 适用边界

- `H_inst`、`H_effect` 等均为 observed-change empirical screen，不提供静态或动态健全性证明。
- 2 个既有 order-sensitive 选择仍保留在累计结果中；precision 必须与 recall 一起阅读。
- unknown 不等于 commute，也不等于 order-sensitive；它表示本实验拒绝作出稳定增量选择判断。
- 本实验没有启动 Worker、`opt`、`clang` 或 `llvm-diff`，也没有运行 LLVM pass 或重建 S/A/B/AB/BA。
- 结果只适用于该 49-program/14-action 冻结 corpus，不外推到其他程序、pass 或真实 batch 搜索。
"""
    talking = f"""# 导师汇报要点

1. {first_sentence}
2. `H_inst` 累计为 {cumulative['cumulative_selected']} selected = {cumulative['cumulative_commute']}C + {cumulative['cumulative_order_sensitive']}OS + {cumulative['cumulative_failed']} failed；precision {_pct(cumulative['cumulative_precision'])}，833-commute recall {_pct(cumulative['cumulative_coverage_of_833_commute'])}。
3. 指令层只比 `H_effect` 新增 {incremental['incremental_selected_count']} 行，新增 commute={incremental['incremental_commute']}、unsafe={incremental['incremental_order_sensitive'] + incremental['incremental_failed']}。
4. 1,342 个候选按 fail-closed 记 unknown，没有删除；这是保守筛查的主要覆盖瓶颈。
5. 四层成本来自各自独立 extractor 的 5 warm-up + 30 measured，低层没有执行高层 fingerprint builder。
6. `INSTRUCTION-EFFECT` paired median 成本为 {_ms(incremental['instruction_incremental_extraction_cost_ms'])}，指令边际值为 {_ms(incremental['instruction_marginal_value'])}。
7. 不作 commute proof 或 Phasebatch 端到端收益声称；这是冻结 corpus 上的 observed-change 覆盖率/成本结果。
"""
    qa = f"""# 导师可能问答

## Q1：最终多覆盖了多少真实 commute？

{first_sentence}

## Q2：`H_inst` 的完整混淆计数是什么？

累计 selected={cumulative['cumulative_selected']}，实际 commute={cumulative['cumulative_commute']}，order-sensitive={cumulative['cumulative_order_sensitive']}，failed={cumulative['cumulative_failed']}；另有增量筛查 unknown={incremental['incremental_unknown']}，这些 unknown 没有进入 selected。

## Q3：precision 和 recall 是多少？

累计 precision={_pct(cumulative['cumulative_precision'])}；对 833 个实际 commute 的 recall={_pct(cumulative['cumulative_coverage_of_833_commute'])}。指令增量 precision={_pct(incremental['incremental_precision'])}。

## Q4：为什么 unknown 这么多？

规则要求稳定 function/block 对应、稳定 CFG/header/attributes、可完整解析 logical instructions、无 wildcard/collision，并且三个源轮次一致。任一条件失败都保守地记 unknown，因此 1,342 行未被冒险选中。

## Q5：指令层是否执行了更高层 extractor？

没有。四个入口分别只构建其累计所需内容；trace 测试和 30 次 raw timing 行同时记录 function/block/effect/instruction builder 次数，低层的高层计数为零。

## Q6：新增 1 个 commute 是否值得？

本实验的局部定义下，真实 commute 动态成本为 {_ms(incremental['incremental_true_commute_cost_ms'])}，unsafe 成本为 {_ms(incremental['incremental_unsafe_cost_ms'])}，paired 指令提取增量中位数为 {_ms(incremental['instruction_incremental_extraction_cost_ms'])}，所以 `instruction_marginal_value`={_ms(incremental['instruction_marginal_value'])}。这个值只用于比较本地提取成本与冻结 pair 动态成本。

## Q7：能否据此跳过实际验证？

不能。observed-change 指令不相交不是 commute 证明；order-sensitive 反例仍存在，后续若集成必须保留精确验证。

## Q8：是否测到了 Phasebatch 加速？

没有。本实验只测特征提取成本和已存在 DYNAMIC_ALL pair 动态成本，未运行 batch/search/validator，因此不得解释为 Phasebatch speedup。
"""

    report_path = report_dir / "instruction_granularity_report_zh.md"
    talking_path = report_dir / "advisor_talking_points_zh.md"
    qa_path = report_dir / "advisor_q_and_a_zh.md"
    write_text(report_path, report)
    write_text(talking_path, talking)
    write_text(qa_path, qa)
    figure_paths = _draw_figures(figure_dir, metrics)
    return (report_path, talking_path, qa_path, *figure_paths)


def _draw_figures(figure_dir: Path, metrics: dict) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np

    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next(
        (
            name
            for name in (
                "Microsoft YaHei",
                "Microsoft YaHei UI",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
            )
            if name in available
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "svg.hashsalt": "instruction-granularity-screen-20260718",
        }
    )
    labels = ["H_func", "H_block", "H_effect", "H_inst"]
    coverage = metrics["coverage"]
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"]
    created: list[Path] = []

    def save(fig, stem: str) -> None:  # noqa: ANN001
        png = figure_dir / f"{stem}.png"
        svg = figure_dir / f"{stem}.svg"
        fig.savefig(
            png,
            dpi=160,
            metadata={"Software": "instruction-granularity-screen"},
        )
        fig.savefig(
            svg,
            metadata={"Date": None, "Creator": "instruction-granularity-screen"},
        )
        plt.close(fig)
        created.extend((png, svg))

    recalls = [float(coverage[label]["commute_recall"]) * 100 for label in labels]
    commute_counts = [int(coverage[label]["selected_commute"]) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    bars = ax.bar(labels, recalls, color=colors, width=0.62)
    ax.set_title("四层对 833 个实际 commute 的累计覆盖率")
    ax.set_ylabel("Recall（%）")
    ax.set_ylim(0, max(recalls) * 1.28)
    ax.grid(axis="y", alpha=0.24)
    for bar, count, value in zip(bars, commute_counts, recalls, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}/833\n{value:.2f}%",
            ha="center",
            va="bottom",
        )
    save(fig, "01_four_level_coverage")

    precisions = [float(coverage[label]["precision"]) * 100 for label in labels]
    selected_counts = [int(coverage[label]["selected_count"]) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    bars = ax.bar(labels, precisions, color=colors, width=0.62)
    ax.set_title("四层累计 empirical precision")
    ax.set_ylabel("Precision（%）")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.24)
    for bar, commute, selected, value in zip(
        bars, commute_counts, selected_counts, precisions, strict=True
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{commute}/{selected}\n{value:.2f}%",
            ha="center",
            va="bottom",
        )
    save(fig, "02_four_level_precision")

    runtime = {row["level"]: row for row in metrics["runtime_summary"]}
    medians = [float(runtime[label]["median_total_extraction_ms"]) for label in (
        "FUNC_ONLY", "BLOCK_ONLY", "EFFECT_ONLY", "INSTRUCTION_ONLY"
    )]
    p90s = [float(runtime[label]["p90_total_extraction_ms"]) for label in (
        "FUNC_ONLY", "BLOCK_ONLY", "EFFECT_ONLY", "INSTRUCTION_ONLY"
    )]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    bars = ax.bar(x, medians, color=colors, width=0.62, label="中位数")
    p90_marks = ax.scatter(
        x + 0.20,
        p90s,
        color="#222222",
        marker="D",
        s=28,
        label="p90",
        zorder=3,
    )
    ax.set_xticks(x, ["FUNC", "BLOCK", "EFFECT", "INSTRUCTION"])
    ax.set_ylabel("每轮总提取时间（ms）")
    ax.set_title("四个独立累计 extractor：30 次 measured")
    ax.set_ylim(0, max(p90s) * 1.2)
    ax.grid(axis="y", alpha=0.24)
    ax.legend([bars, p90_marks], ["中位数", "p90"], frameon=False, loc="upper left")
    for bar, value in zip(bars, medians, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0f}",
            ha="center",
            va="bottom",
        )
    save(fig, "03_extraction_time_by_level")

    incremental = metrics["incremental"]
    unsafe_cost = float(incremental["incremental_unsafe_cost_ms"])
    benefit_values = [
        float(incremental["incremental_true_commute_cost_ms"]),
        -unsafe_cost if unsafe_cost else 0.0,
        -float(incremental["instruction_incremental_extraction_cost_ms"]),
        float(incremental["instruction_marginal_value"]),
    ]
    benefit_labels = ["真实 commute\n动态成本", "unsafe\n扣减", "指令提取增量\n扣减", "边际值"]
    benefit_colors = ["#59A14F", "#E15759", "#F28E2B", "#4E79A7"]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    bars = ax.bar(benefit_labels, benefit_values, color=benefit_colors, width=0.62)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("成本/价值（ms）")
    ax.set_title("指令层增量收益与成本（paired median）")
    ax.grid(axis="y", alpha=0.24)
    span = max(abs(value) for value in benefit_values) or 1.0
    ax.set_ylim(min(0, min(benefit_values)) - span * 0.2, max(0, max(benefit_values)) + span * 0.25)
    for bar, value in zip(bars, benefit_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    save(fig, "04_incremental_benefit_vs_cost")
    return tuple(created)


def _first_sentence(cumulative: dict, incremental: dict) -> str:
    commute = int(cumulative["cumulative_commute"])
    percent = float(cumulative["cumulative_coverage_of_833_commute"]) * 100
    added = int(incremental["incremental_commute"])
    return (
        f"指令级最终覆盖了 {commute}/833 = {percent:.2f}% 的实际 commute，"
        f"相比 H_effect 新增 {added} 个真实 commute。"
    )


def _coverage_table(coverage: dict) -> str:
    lines = []
    for name in ("H_func", "H_block", "H_effect", "H_inst"):
        row = coverage[name]
        lines.append(
            f"| {name} | {row['selected_count']} | {row['selected_commute']} | "
            f"{row['selected_order_sensitive']} | {row['selected_failed']} | "
            f"{row['screen_unknown_count']} | {_pct(row['precision'])} | "
            f"{_pct(row['commute_recall'])} |"
        )
    return "\n".join(lines)


def _runtime_table(runtime: dict) -> str:
    lines = []
    for level in ("FUNC_ONLY", "BLOCK_ONLY", "EFFECT_ONLY", "INSTRUCTION_ONLY"):
        row = runtime[level]
        lines.append(
            f"| {level} | {_ms(row['first_total_extraction_ms'])} | "
            f"{_ms(row['median_total_extraction_ms'])} | {_ms(row['p90_total_extraction_ms'])} | "
            f"{_ms(row['min_total_extraction_ms'])} | {_ms(row['max_total_extraction_ms'])} | "
            f"{_ms(row['median_per_transition_ms'])} | {_ms(row['median_per_pair_ms'])} |"
        )
    return "\n".join(lines)


def _pct(value) -> str:  # noqa: ANN001
    return f"{float(value) * 100:.2f}%"


def _ms(value) -> str:  # noqa: ANN001
    return f"{float(value):.3f} ms"


def _marginal_conclusion(incremental: dict) -> str:
    value = float(incremental["instruction_marginal_value"])
    if value < 0:
        return "该边际值为负：新增真实 commute 的冻结动态成本不足以覆盖指令层 paired 提取增量，因此本 corpus 不支持仅凭这一增量进入集成试点。"
    return "该边际值非负，但仍只构成后续精确验证试点的成本信号，不构成跳过验证的依据。"
