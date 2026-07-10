import tempfile
import unittest
from pathlib import Path

from phasebatch.staged_config import load_staged_config


class StagedConfigTests(unittest.TestCase):
    def test_loads_ordered_stages_and_resolves_relative_pass_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            manifest = root / "staged.yaml"
            manifest.write_text(
                "\n".join(
                    [
                        "root_ir_mode: inlinable-unoptimized",
                        "stages:",
                        "  - id: scalar",
                        "    passes: passes.yaml",
                        "    mode: budgeted",
                        "    max_rounds: 3",
                        "    beam_width: 4",
                        "    require_transition: true",
                        "  - id: cleanup",
                        "    passes: passes.yaml",
                        "    mode: exact",
                        "    max_rounds: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_staged_config(manifest)

        self.assertEqual(config.root_ir_mode, "inlinable-unoptimized")
        self.assertEqual([stage.stage_id for stage in config.stages], ["scalar", "cleanup"])
        self.assertEqual(config.stages[0].passes_path, passes.resolve())
        self.assertEqual(config.stages[0].beam_width, 4)
        self.assertTrue(config.stages[0].require_transition)
        self.assertEqual(config.stages[1].mode, "exact")

    def test_rejects_duplicate_stage_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            manifest = root / "staged.yaml"
            manifest.write_text(
                "stages:\n"
                "  - id: same\n"
                "    passes: passes.yaml\n"
                "  - id: same\n"
                "    passes: passes.yaml\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate stage id"):
                load_staged_config(manifest)

    def test_runtime_command_must_reference_candidate_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            manifest = root / "staged.yaml"
            manifest.write_text(
                "runtime:\n"
                "  enabled: true\n"
                "  command: [fixed-program.exe]\n"
                "stages:\n"
                "  - id: scalar\n"
                "    passes: passes.yaml\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must contain.*exe"):
                load_staged_config(manifest)


if __name__ == "__main__":
    unittest.main()
