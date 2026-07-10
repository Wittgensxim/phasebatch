import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.runtime_rerank import (
    CommandOutcome,
    benchmark_executables,
    choose_runtime_winner,
    rerank_terminal_states,
    select_runtime_candidates,
)
from phasebatch.staged_config import RuntimeConfig


class RuntimeRerankTests(unittest.TestCase):
    def test_selects_only_terminal_or_selected_states_and_deduplicates_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states_dir = root / "states"
            states_dir.mkdir()
            for state_id, text in {
                "S0000": "define void @f() { ret void }\n",
                "S0001": "define void @f() { call void @g() ret void }\n",
                "S0002": "define void @f() { call void @g() ret void }\n",
                "S0003": "define void @f() { ret void }\n",
            }.items():
                state_dir = states_dir / state_id
                state_dir.mkdir()
                (state_dir / "input.ll").write_text(text, encoding="utf-8")

            _write_csv(
                root / "states.csv",
                ["state_id", "state_hash", "parent_state_id", "transition_pass", "ir_path", "is_duplicate", "duplicate_of"],
                [
                    _state("S0000", "h0", "", "", states_dir / "S0000" / "input.ll"),
                    _state("S0001", "h1", "S0000", "pass-a", states_dir / "S0001" / "input.ll"),
                    _state("S0002", "h1", "S0000", "pass-b", states_dir / "S0002" / "input.ll", duplicate_of="S0001"),
                    _state("S0003", "h3", "S0000", "pass-b", states_dir / "S0003" / "input.ll"),
                ],
            )
            _write_csv(
                root / "leaf_states.csv",
                ["state_id", "objective_value", "is_leaf", "selected_as_final"],
                [
                    {"state_id": "S0000", "objective_value": "30", "is_leaf": "false", "selected_as_final": "false"},
                    {"state_id": "S0001", "objective_value": "10", "is_leaf": "true", "selected_as_final": "false"},
                    {"state_id": "S0002", "objective_value": "10", "is_leaf": "true", "selected_as_final": "false"},
                    {"state_id": "S0003", "objective_value": "20", "is_leaf": "false", "selected_as_final": "true"},
                ],
            )
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - pass-a\n  - pass-b\n", encoding="utf-8")

            candidates = select_runtime_candidates(root, passes, top_k=5)

        self.assertEqual({candidate.state_id for candidate in candidates}, {"S0001", "S0003"})
        by_id = {candidate.state_id: candidate for candidate in candidates}
        self.assertEqual(by_id["S0001"].pipeline, ("pass-a",))
        self.assertEqual(by_id["S0003"].pipeline, ("pass-b",))

    def test_benchmarks_in_cyclic_order_and_selects_lowest_median(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executables = {
                "S0001": root / "S0001.exe",
                "S0002": root / "S0002.exe",
            }
            for path in executables.values():
                path.write_text("", encoding="utf-8")
            config = RuntimeConfig(
                enabled=True,
                top_k=2,
                warmups=1,
                trials=3,
                timeout=5,
                command=("{exe}",),
            )

            def fake_runner(command, timeout, cwd):
                del timeout, cwd
                state_id = Path(command[0]).stem
                return CommandOutcome(returncode=0, time_ms=80.0 if state_id == "S0002" else 100.0)

            trials, summary = benchmark_executables(executables, config, command_runner=fake_runner)
            winner = choose_runtime_winner(summary)

        formal = [row for row in trials if row["kind"] == "execute"]
        self.assertEqual(len(formal), 6)
        self.assertEqual(
            [(row["trial"], row["order_position"], row["state_id"]) for row in formal[:4]],
            [("1", "1", "S0001"), ("1", "2", "S0002"), ("2", "1", "S0002"), ("2", "2", "S0001")],
        )
        self.assertEqual(winner["state_id"], "S0002")
        self.assertEqual(winner["median_ms"], "80.000")

    def test_nonzero_exit_candidate_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executables = {"bad": root / "bad.exe", "good": root / "good.exe"}
            for path in executables.values():
                path.write_text("", encoding="utf-8")
            config = RuntimeConfig(enabled=True, top_k=2, warmups=1, trials=2, expected_exit_code=0)

            def fake_runner(command, timeout, cwd):
                del timeout, cwd
                return CommandOutcome(returncode=1 if Path(command[0]).stem == "bad" else 0, time_ms=1.0)

            _trials, summary = benchmark_executables(executables, config, command_runner=fake_runner)

        by_id = {row["state_id"]: row for row in summary}
        self.assertEqual(by_id["bad"]["eligible"], "false")
        self.assertEqual(by_id["good"]["eligible"], "true")

    def test_compile_failure_is_never_executed_and_artifacts_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            states_dir = run_dir / "states"
            states_dir.mkdir(parents=True)
            for state_id in ("S0000", "S0001", "S0002", "S0003"):
                state_dir = states_dir / state_id
                state_dir.mkdir()
                (state_dir / "input.ll").write_text("define i32 @main() { ret i32 0 }\n", encoding="utf-8")
            _write_csv(
                run_dir / "states.csv",
                [
                    "state_id",
                    "state_hash",
                    "parent_state_id",
                    "transition_pass",
                    "ir_path",
                    "is_duplicate",
                    "duplicate_of",
                ],
                [
                    _state("S0000", "h0", "", "", states_dir / "S0000" / "input.ll"),
                    _state("S0001", "h1", "S0000", "pass-a", states_dir / "S0001" / "input.ll"),
                    _state("S0002", "h2", "S0000", "pass-b", states_dir / "S0002" / "input.ll"),
                    _state("S0003", "h3", "S0000", "pass-c", states_dir / "S0003" / "input.ll"),
                ],
            )
            _write_csv(
                run_dir / "leaf_states.csv",
                ["state_id", "objective_value", "is_leaf", "selected_as_final"],
                [
                    {"state_id": "S0001", "objective_value": "10", "is_leaf": "true", "selected_as_final": "true"},
                    {"state_id": "S0002", "objective_value": "20", "is_leaf": "true", "selected_as_final": "false"},
                    {"state_id": "S0003", "objective_value": "1", "is_leaf": "false", "selected_as_final": "false"},
                ],
            )
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - pass-a\n  - pass-b\n  - pass-c\n", encoding="utf-8")
            out_dir = root.relative_to(Path.cwd()) / "runtime"
            config = RuntimeConfig(enabled=True, top_k=2, warmups=1, trials=2)
            commands = []

            def fake_runner(command, timeout, cwd):
                del timeout, cwd
                commands.append(list(command))
                if command[0] == "llc":
                    state_id = Path(command[1]).parent.name
                    if state_id == "S0001":
                        return CommandOutcome(returncode=1, time_ms=1.0, stderr="compile failed")
                    Path(command[-1]).write_text("object", encoding="utf-8")
                elif command[0] == "clang":
                    Path(command[-1]).write_text("executable", encoding="utf-8")
                return CommandOutcome(returncode=0, time_ms=5.0)

            result = rerank_terminal_states(
                run_dir,
                passes,
                out_dir,
                config,
                tools={"llc": "llc", "clang": "clang"},
                command_runner=fake_runner,
            )

        self.assertEqual(result.winner.state_id, "S0002")
        flattened = "\n".join(" ".join(command) for command in commands)
        self.assertNotIn("S0003", flattened)
        self.assertFalse(any(Path(command[0]).stem == "S0001" for command in commands if command[0].endswith(".exe")))
        output_paths = [Path(command[command.index("-o") + 1]) for command in commands if "-o" in command]
        self.assertTrue(all(path.is_absolute() for path in output_paths))
        self.assertTrue(result.candidates_csv.name == "runtime_candidates.csv")
        self.assertTrue(result.trials_csv.name == "runtime_trials.csv")
        self.assertTrue(result.summary_csv.name == "runtime_summary.csv")


def _state(state_id, state_hash, parent, transition, ir_path, duplicate_of=""):
    return {
        "state_id": state_id,
        "state_hash": state_hash,
        "parent_state_id": parent,
        "transition_pass": transition,
        "ir_path": str(ir_path),
        "is_duplicate": "true" if duplicate_of else "false",
        "duplicate_of": duplicate_of,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
