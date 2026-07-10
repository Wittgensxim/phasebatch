import csv
import json
import tempfile
import unittest
from pathlib import Path

from phasebatch.advisor_markdown import CORRECTNESS_BOUNDARY_ZH, generate_advisor_markdown


class AdvisorMarkdownTests(unittest.TestCase):
    def test_generates_required_chinese_reports_and_dropped_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            _write_csv(
                study / "program_summary.csv",
                ["program", "states", "transitions", "avg_active_passes", "commute_ratio", "unknown_ratio", "dropped_active_passes", "total_wall_time_ms"],
                [{"program": "demo", "states": "2", "transitions": "1", "avg_active_passes": "3", "commute_ratio": "0.6", "unknown_ratio": "0.1", "dropped_active_passes": "1", "total_wall_time_ms": "100"}],
            )
            _write_csv(
                study / "overlap_component_program_summary.csv",
                ["program", "mean_component_size", "median_component_size", "p90_component_size", "max_component_size", "singleton_ratio", "size_le_3_ratio", "size_ge_8_count"],
                [{"program": "demo", "mean_component_size": "2", "median_component_size": "2", "p90_component_size": "3", "max_component_size": "3", "singleton_ratio": "0.2", "size_le_3_ratio": "0.8", "size_ge_8_count": "0"}],
            )
            _write_csv(
                study / "conflict_component_program_summary.csv",
                ["program", "mean_component_size", "median_component_size", "p90_component_size", "max_component_size", "singleton_ratio", "size_le_3_ratio", "size_ge_8_count"],
                [{"program": "demo", "mean_component_size": "4", "median_component_size": "4", "p90_component_size": "8", "max_component_size": "8", "singleton_ratio": "0.1", "size_le_3_ratio": "0.4", "size_ge_8_count": "1"}],
            )
            _write_csv(
                study / "cost_breakdown_by_program.csv",
                ["program", "total_wall_time_ms", "batch_validation_wall_ms"],
                [{"program": "demo", "total_wall_time_ms": "100", "batch_validation_wall_ms": "60"}],
            )
            _write_csv(study / "unknown_failure_summary.csv", ["program", "pair_unknown", "lazy_budget_skipped"], [{"program": "demo", "pair_unknown": "1", "lazy_budget_skipped": "0"}])
            _write_csv(study / "figures_manifest.csv", ["figure_id", "title_zh", "png_path", "svg_path", "source_csv", "status", "warning"], [])

            result = generate_advisor_markdown(study, metadata={"mode": "budgeted", "benchmark_count": 1})
            report = (study / "advisor_report_zh.md").read_text(encoding="utf-8")
            talking = (study / "advisor_talking_points_zh.md").read_text(encoding="utf-8")
            dictionary = (study / "data_dictionary_zh.md").read_text(encoding="utf-8")
            metadata = json.loads((study / "report_metadata.json").read_text(encoding="utf-8"))

        self.assertIn("警告：发现 active pass 被静默丢弃", report)
        self.assertIn("## 14. 限制", report)
        self.assertIn(CORRECTNESS_BOUNDARY_ZH, report)
        self.assertIn("当前主要工程瓶颈是 batch validation", report)
        self.assertIn("部分状态存在较大冲突组件", report)
        self.assertIn("30 秒：问题", talking)
        self.assertIn("下一步", talking)
        self.assertIn("true relation flip", dictionary)
        self.assertEqual(metadata["mode"], "budgeted")
        self.assertTrue(Path(result["advisor_key_numbers_csv"]).name == "advisor_key_numbers.csv")


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
