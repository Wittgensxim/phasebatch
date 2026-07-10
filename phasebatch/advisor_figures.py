from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402


FIGURE_FIELDS = ["figure_id", "title_zh", "png_path", "svg_path", "source_csv", "status", "warning"]
PALETTE = ["#2F6B9A", "#D66B4D", "#7A7F85", "#3F8A6B", "#C39B35", "#7A5FA3"]


def configure_chinese_font() -> tuple[str | None, str]:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name, ""
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None, "未找到可用中文字体，图片文字使用 matplotlib 默认字体回退。"


def generate_advisor_figures(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _font, font_warning = configure_chinese_font()
    plans: list[tuple[str, str, str, str, Callable[[list[dict]], object]]] = [
        ("01", "各程序当前状态下的 Pair 关系比例", "01_commute_ratio_by_program", "pair_relation_summary.csv", _plot_pair_ratios),
        ("02", "Overlap Component 大小分布", "02_overlap_component_size_distribution", "overlap_component_size_buckets.csv", _plot_component_buckets),
        ("03", "Conflict Component 大小分布", "03_conflict_component_size_distribution", "conflict_component_size_buckets.csv", _plot_component_buckets),
        ("04", "小型 Overlap Cluster 中 AB == BA 的比例", "04_small_cluster_abba_ratio", "small_overlap_cluster_abba.csv", _plot_abba),
        ("05", "各程序的局部搜索空间压缩", "05_batch_reduction_log10", "batch_reduction_program_summary.csv", _plot_reduction),
        ("06", "各程序 Phasebatch wall-clock 成本拆解", "06_cost_breakdown", "cost_breakdown_by_program.csv", _plot_cost),
        ("07", "最常出现在大型 Conflict Component 的 Pass", "07_top_conflict_passes", "top_conflict_passes.csv", _plot_conflict_passes),
        ("08", "按搜索深度观察 State-aware 变化", "08_state_aware_by_depth", "state_aware_by_depth.csv", _plot_state_aware),
        ("09", "IR Equality 证据层级", "09_equality_tiers", "equality_tier_summary_all.csv", _plot_equality),
    ]
    manifest = []
    for figure_id, title, stem, source_name, plotter in plans:
        source_path = study_dir / source_name
        rows = _read_csv(source_path)
        warning_parts = [font_warning] if font_warning else []
        if not rows:
            warning_parts.append(f"源数据 {source_name} 为空或不存在。")
        fig = plotter(rows)
        png = figures_dir / f"{stem}.png"
        svg = figures_dir / f"{stem}.svg"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
            fig.suptitle(title, fontweight="normal")
            fig.tight_layout()
            fig.savefig(png, dpi=200, bbox_inches="tight")
            fig.savefig(svg, format="svg", bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "figure_id": figure_id,
                "title_zh": title,
                "png_path": str(png.relative_to(study_dir)).replace("\\", "/"),
                "svg_path": str(svg.relative_to(study_dir)).replace("\\", "/"),
                "source_csv": source_name,
                "status": "generated",
                "warning": " ".join(warning_parts),
            }
        )
    _write_csv(study_dir / "figures_manifest.csv", FIGURE_FIELDS, manifest)
    return {"figures": len(manifest), "figures_dir": str(figures_dir), "font_warning": font_warning}


def _plot_pair_ratios(rows: list[dict]):
    fig, ax = plt.subplots(figsize=_figure_size(rows))
    if not rows:
        return _no_data(fig, ax)
    labels = [row.get("program", "") for row in rows]
    commute = [_float(row.get("commute_ratio")) for row in rows]
    sensitive = [_float(row.get("order_sensitive_ratio")) for row in rows]
    unknown = [_float(row.get("unknown_ratio")) for row in rows]
    x = list(range(len(rows)))
    ax.bar(x, commute, label="当前状态可交换", color=PALETTE[0])
    ax.bar(x, sensitive, bottom=commute, label="顺序敏感", color=PALETTE[1])
    ax.bar(x, unknown, bottom=[a + b for a, b in zip(commute, sensitive)], label="Unknown", color=PALETTE[2])
    _program_axis(ax, x, labels)
    ax.set_ylabel("Pair 比例")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3)
    return fig


def _plot_component_buckets(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not rows:
        return _no_data(fig, ax)
    buckets = ["1", "2", "3", "4-5", "6-7", "8-10", ">10"]
    counts = {bucket: sum(_int(row.get("components")) for row in rows if row.get("size_bucket") == bucket) for bucket in buckets}
    bars = ax.bar(buckets, [counts[bucket] for bucket in buckets], color=PALETTE[3])
    ax.bar_label(bars, padding=2)
    ax.set_xlabel("Component size bucket")
    ax.set_ylabel("Component 数量")
    ax.grid(axis="y", alpha=0.2)
    return fig


def _plot_abba(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not rows:
        return _no_data(fig, ax)
    buckets = [row.get("size_bucket", "") for row in rows]
    values = [_float(row.get("ab_ba_equal_ratio")) for row in rows]
    bars = ax.bar(buckets, values, color=PALETTE[4])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=2)
    ax.set_xlabel("Cluster size bucket")
    ax.set_ylabel("AB == BA 比例")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    return fig


def _plot_reduction(rows: list[dict]):
    fig, ax = plt.subplots(figsize=_figure_size(rows))
    if not rows:
        return _no_data(fig, ax)
    labels = [row.get("program", "") for row in rows]
    averages = [_float(row.get("avg_local_reduction_log10")) for row in rows]
    maxima = [_float(row.get("max_local_reduction_log10")) for row in rows]
    x = list(range(len(rows)))
    width = 0.38
    ax.bar([value - width / 2 for value in x], averages, width, label="平均局部压缩", color=PALETTE[0])
    ax.bar([value + width / 2 for value in x], maxima, width, label="最大局部压缩", color=PALETTE[4])
    _program_axis(ax, x, labels)
    ax.set_ylabel("local reduction log10")
    ax.legend(frameon=False)
    return fig


def _plot_cost(rows: list[dict]):
    fig, ax = plt.subplots(figsize=_figure_size(rows))
    if not rows:
        return _no_data(fig, ax)
    labels = [row.get("program", "") for row in rows]
    series = [
        ("Profiling", "profiling_wall_ms", PALETTE[0]),
        ("Pair testing", "pair_testing_wall_ms", PALETTE[1]),
        ("Batch validation", "batch_validation_wall_ms", PALETTE[4]),
        ("Search", "state_search_wall_ms", PALETTE[3]),
        ("其他", "other_wall_ms", PALETTE[2]),
    ]
    x = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for label, field, color in series:
        values = [_float(row.get(field)) for row in rows]
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom = [left + right for left, right in zip(bottom, values)]
    _program_axis(ax, x, labels)
    ax.set_ylabel("wall-clock（毫秒）")
    ax.legend(frameon=False, ncol=3)
    return fig


def _plot_conflict_passes(rows: list[dict]):
    selected = sorted(rows, key=lambda row: _float(row.get("weighted_conflict_score")), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(9, max(4.8, len(selected) * 0.35)))
    if not selected:
        return _no_data(fig, ax)
    selected.reverse()
    labels = [row.get("pass", "") for row in selected]
    values = [_float(row.get("weighted_conflict_score")) for row in selected]
    bars = ax.barh(labels, values, color=PALETTE[1])
    ax.bar_label(bars, padding=3, fmt="%.1f")
    ax.set_xlabel("weighted conflict score")
    return fig


def _plot_state_aware(rows: list[dict]):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if not rows:
        top.text(0.5, 0.5, "无可用数据", ha="center", va="center", transform=top.transAxes)
        top.set_axis_off()
        bottom.set_axis_off()
        return fig
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_int(row.get("depth")), []).append(row)
    depths = sorted(grouped)
    active = [sum(_float(row.get("avg_active_passes")) for row in grouped[depth]) / len(grouped[depth]) for depth in depths]
    top.plot(depths, active, marker="o", color=PALETTE[0], label="平均 active passes")
    top.set_ylabel("Active passes")
    top.grid(alpha=0.2)
    enable = [sum(_int(row.get("enable_count")) for row in grouped[depth]) for depth in depths]
    suppress = [sum(_int(row.get("suppress_count")) for row in grouped[depth]) for depth in depths]
    flips = [sum(_int(row.get("true_relation_flip_count")) for row in grouped[depth]) for depth in depths]
    bottom.plot(depths, enable, marker="o", label="Enable", color=PALETTE[3])
    bottom.plot(depths, suppress, marker="s", label="Suppress", color=PALETTE[1])
    bottom.plot(depths, flips, marker="^", label="True relation flip", color=PALETTE[4])
    bottom.set_xlabel("搜索深度")
    bottom.set_ylabel("事件数量")
    bottom.grid(alpha=0.2)
    bottom.legend(frameon=False, ncol=3)
    return fig


def _plot_equality(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not rows:
        return _no_data(fig, ax)
    tiers = ["canonical_hash", "structural_diff", "different", "failed"]
    values = [sum(_int(row.get("count")) for row in rows if row.get("equality_tier") == tier) for tier in tiers]
    bars = ax.bar(tiers, values, color=[PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[2]])
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("Pair 数量")
    ax.tick_params(axis="x", rotation=15)
    return fig


def _no_data(fig, ax):
    ax.text(0.5, 0.5, "无可用数据", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def _program_axis(ax, positions: list[int], labels: list[str]) -> None:
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2)


def _figure_size(rows: list[dict]) -> tuple[float, float]:
    return (max(9.0, min(18.0, 0.65 * max(1, len(rows)))), 5.4)


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
