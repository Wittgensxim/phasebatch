import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.mainline import expand_inputs, run_mainline


class MainlineRunnerTests(unittest.TestCase):
    def test_input_glob_expansion_warns_for_unmatched_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.c"
            second = root / "b.ll"
            ignored = root / "note.txt"
            first.write_text("int a(void){return 0;}\n", encoding="utf-8")
            second.write_text("define i32 @b() { ret i32 0 }\n", encoding="utf-8")
            ignored.write_text("not input\n", encoding="utf-8")
            warnings: list[str] = []

            expanded = expand_inputs([str(root / "*.c"), str(second), str(root / "missing*.c"), str(ignored)], warnings.append)

        self.assertEqual([path.name for path in expanded], ["a.c", "b.ll"])
        self.assertIn("input glob matched no files", warnings[0])
        self.assertIn("skipping unsupported input", warnings[1])

    def test_output_directory_name_uses_input_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            passes_path = root / "passes.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.mainline.run_explore_batches", side_effect=_fake_explore):
                result = run_mainline(
                    [str(input_path)],
                    out_dir,
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=3,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=20,
                    validate_batches=False,
                )

            runs = _read_csv(out_dir / "mainline_runs.csv")
            states_exists = (out_dir / "branch" / "states.csv").exists()

        self.assertEqual(result["programs"], 1)
        self.assertEqual(runs[0]["program"], "branch")
        self.assertEqual(Path(runs[0]["output_dir"]).name, "branch")
        self.assertTrue(states_exists)

    def test_existing_output_directory_without_overwrite_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            passes_path = root / "passes.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            (root / "out" / "branch").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "output directory already exists"):
                run_mainline(
                    [str(input_path)],
                    root / "out",
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=20,
                    validate_batches=False,
                )

    def test_overwrite_removes_old_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            passes_path = root / "passes.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            old_file = root / "out" / "branch" / "old.txt"
            old_file.parent.mkdir(parents=True)
            old_file.write_text("stale\n", encoding="utf-8")

            with mock.patch("phasebatch.mainline.run_explore_batches", side_effect=_fake_explore):
                run_mainline(
                    [str(input_path)],
                    root / "out",
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=20,
                    validate_batches=False,
                    overwrite=True,
                )

        self.assertFalse(old_file.exists())

    def test_continue_on_error_records_failed_program_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.c"
            bad = root / "bad.c"
            passes_path = root / "passes.yaml"
            good.write_text("int good(void){return 0;}\n", encoding="utf-8")
            bad.write_text("int bad(void){return 1;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            def fake_explore(input_path, out_dir, passes_path, **kwargs):
                if Path(input_path).name == "bad.c":
                    raise RuntimeError("boom")
                return _fake_explore(input_path, out_dir, passes_path, **kwargs)

            with mock.patch("phasebatch.mainline.run_explore_batches", side_effect=fake_explore):
                result = run_mainline(
                    [str(good), str(bad)],
                    root / "out",
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=20,
                    validate_batches=False,
                    continue_on_error=True,
                )

            runs = _read_csv(root / "out" / "mainline_runs.csv")
            good_states_exists = (root / "out" / "good" / "states.csv").exists()

        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual([row["status"] for row in runs], ["success", "failed"])
        self.assertIn("boom", runs[1]["error_message"])
        self.assertTrue(good_states_exists)

    def test_aggregate_files_are_generated_from_per_program_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.c"
            second = root / "b.c"
            passes_path = root / "passes.yaml"
            first.write_text("int a(void){return 0;}\n", encoding="utf-8")
            second.write_text("int b(void){return 0;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.mainline.run_explore_batches", side_effect=_fake_explore):
                result = run_mainline(
                    [str(first), str(second)],
                    root / "out",
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=20,
                    validate_batches=False,
                )

            states = _read_csv(root / "out" / "mainline_aggregate_states.csv")
            batches = _read_csv(root / "out" / "mainline_aggregate_batches.csv")
            coverage = _read_csv(root / "out" / "mainline_aggregate_coverage.csv")
            overlap = _read_csv(root / "out" / "mainline_aggregate_overlap.csv")
            missing = _read_csv(root / "out" / "mainline_missing_outputs.csv")

        self.assertEqual(result["successes"], 2)
        self.assertEqual([row["program"] for row in states], ["a", "b"])
        self.assertEqual([row["program"] for row in batches], ["a", "b"])
        self.assertEqual([row["program"] for row in coverage], ["a", "b"])
        self.assertEqual([row["program"] for row in overlap], ["a", "b"])
        self.assertEqual({row["status"] for row in missing}, {"present"})

    def test_eval_objective_runs_after_mainline_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            passes_path = root / "passes.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.mainline.run_explore_batches", side_effect=_fake_explore), \
                mock.patch("phasebatch.mainline.eval_batch_objectives", return_value={"rows": 1}) as fake_eval:
                result = run_mainline(
                    [str(input_path)],
                    out_dir,
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=20,
                    validate_batches=False,
                    eval_objective="ir-inst-count",
                )

        fake_eval.assert_called_once_with(out_dir, objective="ir-inst-count", recursive=True)
        self.assertEqual(result["objective_rows"], 1)


def _fake_explore(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    program = Path(out_dir).name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "states.csv", ["program", "state_id"], [{"program": program, "state_id": "S0000"}])
    _write_csv(out_dir / "aggregate_by_depth.csv", ["program", "depth", "num_states"], [{"program": program, "depth": "0", "num_states": "1"}])
    _write_csv(out_dir / "aggregate_batch_summary.csv", ["program", "depth", "states"], [{"program": program, "depth": "0", "states": "1"}])
    _write_csv(out_dir / "aggregate_coverage_summary.csv", ["program", "depth", "states"], [{"program": program, "depth": "0", "states": "1"}])
    _write_csv(out_dir / "aggregate_overlap_summary.csv", ["program", "depth", "states"], [{"program": program, "depth": "0", "states": "1"}])
    return {"program": program, "states": 1, "batch_transitions": 0}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
