import tempfile
import unittest
from pathlib import Path

from phasebatch.config import load_passes
from phasebatch.pass_config import load_pass_config, load_pass_registry


class PassConfigTests(unittest.TestCase):
    def test_old_string_only_config_loads_as_specs_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passes.yaml"
            path.write_text("passes:\n  - mem2reg\n  - sroa\n", encoding="utf-8")

            specs = load_pass_config(path)

            self.assertEqual([spec.name for spec in specs], ["mem2reg", "sroa"])
            self.assertEqual([spec.pipeline for spec in specs], ["mem2reg", "sroa"])
            self.assertEqual([spec.pipeline_candidates for spec in specs], [["mem2reg"], ["sroa"]])
            self.assertEqual([spec.category for spec in specs], ["unknown", "unknown"])
            self.assertEqual([spec.stage for spec in specs], ["", ""])
            self.assertEqual([spec.enabled for spec in specs], [True, True])
            self.assertEqual(load_passes(path), ["mem2reg", "sroa"])

    def test_rich_dict_config_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passes.yaml"
            path.write_text(
                "\n".join(
                    [
                        "passes:",
                        "  - name: mem2reg",
                        "    pipeline: mem2reg",
                        "    category: scalar",
                        "    stage: v1",
                        "    enabled: true",
                    ]
                ),
                encoding="utf-8",
            )

            [spec] = load_pass_config(path)

            self.assertEqual(spec.name, "mem2reg")
            self.assertEqual(spec.pipeline, "mem2reg")
            self.assertEqual(spec.pipeline_candidates, ["mem2reg"])
            self.assertEqual(spec.category, "scalar")
            self.assertEqual(spec.stage, "v1")
            self.assertTrue(spec.enabled)

    def test_disabled_entries_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passes.yaml"
            path.write_text(
                "\n".join(
                    [
                        "passes:",
                        "  - name: enabled-pass",
                        "    pipeline: enabled-pass",
                        "  - name: disabled-pass",
                        "    pipeline: disabled-pass",
                        "    enabled: false",
                    ]
                ),
                encoding="utf-8",
            )

            specs = load_pass_config(path)

            self.assertEqual([spec.name for spec in specs], ["enabled-pass"])
            self.assertEqual(load_passes(path), ["enabled-pass"])

    def test_pipeline_candidates_are_parsed_and_pipeline_is_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passes.yaml"
            path.write_text(
                "\n".join(
                    [
                        "passes:",
                        "  - name: licm",
                        "    pipeline: licm",
                        "    pipeline_candidates:",
                        "      - loop(licm)",
                        "      - function(loop(licm))",
                        "    category: loop",
                        "    stage: v3",
                    ]
                ),
                encoding="utf-8",
            )

            [spec] = load_pass_config(path)

            self.assertEqual(spec.pipeline_candidates, ["licm", "loop(licm)", "function(loop(licm))"])
            self.assertEqual(load_passes(path), ["licm"])

    def test_registry_maps_name_to_resolved_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_passes.yaml"
            path.write_text(
                "\n".join(
                    [
                        "passes:",
                        "  - name: licm",
                        "    pipeline: function(loop(licm))",
                        "    category: loop",
                        "    stage: v3",
                    ]
                ),
                encoding="utf-8",
            )

            registry = load_pass_registry(path)

            self.assertEqual(registry.names(), ["licm"])
            self.assertEqual(registry.pipeline_for("licm"), "function(loop(licm))")
            self.assertEqual(registry.category_for("licm"), "loop")
            self.assertEqual(registry.stage_for("licm"), "v3")
            self.assertEqual(load_passes(path), ["licm"])

    def test_versioned_configs_load_with_unique_names(self) -> None:
        expected_counts = {
            "configs/core_passes_v1.yaml": 14,
            "configs/scalar_passes_v2.yaml": 19,
        }
        for config_path, expected_count in expected_counts.items():
            with self.subTest(config_path=config_path):
                specs = load_pass_config(Path(config_path))
                names = [spec.name for spec in specs]

                self.assertEqual(len(specs), expected_count)
                self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
