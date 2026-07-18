from __future__ import annotations

from pathlib import Path
import re

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_main_report_begins_with_required_result_sentence() -> None:
    report = (ROOT / "report" / "instruction_granularity_report_zh.md").read_text(encoding="utf-8")
    first = report.splitlines()[0]
    assert first == "指令级最终覆盖了 46/833 = 5.52% 的实际 commute，相比 H_effect 新增 1 个真实 commute。"
    for required in (
        "累计选中 48 行",
        "46 个实际 commute",
        "2 个 order-sensitive",
        "0 个 failed",
        "1342",
        "instruction_marginal_value",
        "5 次 warm-up",
        "30 次 measured",
    ):
        assert required in report


def test_reports_do_not_overclaim() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "report").glob("*.md"))
    )
    forbidden_positive = (
        r"(?<!不)证明(?:了)?可交换",
        r"Phasebatch (?:端到端 )?(?:加速|speedup) (?:达到|为|是)",
        r"指令不相交(?:是|保证|证明)",
        r"sound(?:ness)? (?:is|=) true",
    )
    for pattern in forbidden_positive:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_all_four_png_and_svg_figures_exist_and_are_legible_size() -> None:
    stems = (
        "01_four_level_coverage",
        "02_four_level_precision",
        "03_extraction_time_by_level",
        "04_incremental_benefit_vs_cost",
    )
    for stem in stems:
        png = ROOT / "figures" / f"{stem}.png"
        svg = ROOT / "figures" / f"{stem}.svg"
        assert png.stat().st_size > 20_000
        assert svg.stat().st_size > 5_000
        with Image.open(png) as image:
            assert image.size == (960, 576)

