import tempfile
import unittest
from pathlib import Path

from phasebatch.artifact_cleanup import cleanup_ir_artifacts, mark_ir_artifacts_kept


class ArtifactCleanupTests(unittest.TestCase):
    def test_cleanup_removes_ll_artifacts_and_empty_directories_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            empty_artifact_dir = run_dir / "states" / "S0001" / "artifacts"
            nested_empty_dir = run_dir / "states" / "S0002" / "artifacts" / "batch_successors"
            nonempty_dir = run_dir / "states" / "S0000"

            empty_artifact_dir.mkdir(parents=True)
            nested_empty_dir.mkdir(parents=True)
            nonempty_dir.mkdir(parents=True)
            (empty_artifact_dir / "child.ll").write_text("define void @f() {}\n", encoding="utf-8")
            (nested_empty_dir / "B0000.ll").write_text("define void @g() {}\n", encoding="utf-8")
            (nonempty_dir / "input.ll").write_text("define void @root() {}\n", encoding="utf-8")
            (nonempty_dir / "states.csv").write_text("state_id\nS0000\n", encoding="utf-8")

            result = cleanup_ir_artifacts(run_dir)

            remaining_ll = list(run_dir.rglob("*.ll"))

            self.assertEqual(remaining_ll, [])
            self.assertFalse(empty_artifact_dir.exists())
            self.assertFalse(nested_empty_dir.exists())
            self.assertTrue(nonempty_dir.exists())
            self.assertTrue((nonempty_dir / "states.csv").exists())
            self.assertTrue(run_dir.exists())
            self.assertEqual(result["ir_artifacts_cleaned"], "true")
            self.assertEqual(result["deleted_ir_artifacts"], "3")
            self.assertGreaterEqual(int(result["deleted_empty_dirs"]), 2)

    def test_cleanup_missing_run_dir_reports_zero_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = cleanup_ir_artifacts(Path(tmp) / "missing")

        self.assertEqual(result["deleted_ir_artifacts"], "0")
        self.assertEqual(result["deleted_empty_dirs"], "0")

    def test_mark_ir_artifacts_kept_reports_no_empty_directory_cleanup(self) -> None:
        result = mark_ir_artifacts_kept()

        self.assertEqual(result["ir_artifacts_cleaned"], "false")
        self.assertEqual(result["deleted_ir_artifacts"], "0")
        self.assertEqual(result["deleted_empty_dirs"], "0")


if __name__ == "__main__":
    unittest.main()
