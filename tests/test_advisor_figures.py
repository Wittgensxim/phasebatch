import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.advisor_figures import configure_chinese_font, generate_advisor_figures


class AdvisorFigureTests(unittest.TestCase):
    def test_generates_png_svg_and_manifest_without_chinese_font(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            _write_csv(
                study / "pair_relation_summary.csv",
                ["program", "commute_ratio", "order_sensitive_ratio", "unknown_ratio"],
                [{"program": "demo", "commute_ratio": "0.5", "order_sensitive_ratio": "0.4", "unknown_ratio": "0.1"}],
            )
            _write_csv(
                study / "batch_reduction_program_summary.csv",
                ["program", "avg_local_reduction_log10", "max_local_reduction_log10"],
                [{"program": "demo", "avg_local_reduction_log10": "2", "max_local_reduction_log10": "4"}],
            )

            with mock.patch(
                "phasebatch.advisor_figures.configure_chinese_font",
                return_value=(None, "未找到可用中文字体，使用默认字体。"),
            ):
                result = generate_advisor_figures(study)
            manifest = _read_csv(study / "figures_manifest.csv")

            png_files = list((study / "figures").glob("*.png"))
            svg_files = list((study / "figures").glob("*.svg"))
            sizes = [path.stat().st_size for path in png_files + svg_files]

        self.assertEqual(result["figures"], 9)
        self.assertEqual(len(manifest), 9)
        self.assertEqual(len(png_files), 9)
        self.assertEqual(len(svg_files), 9)
        self.assertTrue(all(size > 100 for size in sizes))
        self.assertTrue(all(row["status"] == "generated" for row in manifest))
        self.assertTrue(any("中文字体" in row["warning"] for row in manifest))

    def test_font_configuration_reports_missing_chinese_font(self) -> None:
        with mock.patch("phasebatch.advisor_figures.font_manager.fontManager.ttflist", []):
            font, warning = configure_chinese_font()

        self.assertIsNone(font)
        self.assertIn("中文字体", warning)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
