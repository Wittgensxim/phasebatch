import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.batch_objective import count_ir_instructions, eval_batch_objectives


class BatchObjectiveTests(unittest.TestCase):
    def test_count_ir_instructions_ignores_metadata_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ir = Path(tmp) / "input.ll"
            ir.write_text(
                "\n".join(
                    [
                        "source_filename = \"x.c\"",
                        "target triple = \"x86_64-pc-windows-msvc\"",
                        "declare i32 @puts(ptr)",
                        "",
                        "define i32 @f(i32 %x) #0 {",
                        "entry:",
                        "  %a = add i32 %x, 1",
                        "  br label %next",
                        "",
                        "next:",
                        "  %b = mul i32 %a, 2",
                        "  ret i32 %b",
                        "}",
                        "attributes #0 = { nounwind }",
                        "!llvm.module.flags = !{!0}",
                        "!0 = !{i32 1, !\"wchar_size\", i32 4}",
                    ]
                ),
                encoding="utf-8",
            )

            count = count_ir_instructions(ir)

        self.assertEqual(count, 4)

    def test_objective_signal_generated_for_mock_batch_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_program_run(Path(tmp) / "branch", program="branch")

            result = eval_batch_objectives(run_dir, objective="ir-inst-count")
            rows = _read_csv(run_dir / "objective_signal.csv")
            summary = (run_dir / "objective_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["objective_signal_csv"], str(run_dir / "objective_signal.csv"))
        self.assertEqual(result["rows"], 1)
        self.assertEqual(rows[0]["program"], "branch")
        self.assertEqual(rows[0]["transition_kind"], "batch")
        self.assertEqual(rows[0]["batch_id"], "B0000")
        self.assertEqual(rows[0]["correctness_class"], "certified_batch")
        self.assertEqual(rows[0]["ir_inst_before"], "3")
        self.assertEqual(rows[0]["ir_inst_after"], "2")
        self.assertEqual(rows[0]["ir_inst_delta"], "-1")
        self.assertEqual(rows[0]["ir_inst_reduction_pct"], "33.33")
        self.assertEqual(rows[0]["objective_kind"], "ir-inst-count")
        self.assertIn("objective signal only; not used as commutation proof", rows[0]["objective_note"])
        self.assertIn("# Objective Signal Summary", summary)
        self.assertIn("Objective signals are used only for evaluation and ranking. They are not used as commutation or independence proof.", summary)

    def test_recursive_mode_creates_aggregate_objective_signal_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mainline"
            _write_program_run(root / "branch", program="branch")
            _write_program_run(root / "loop", program="loop")

            result = eval_batch_objectives(root, objective="ir-inst-count", recursive=True)
            aggregate = _read_csv(root / "aggregate_objective_signal.csv")
            summary = (root / "objective_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["program_dirs"], 2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual([row["program"] for row in aggregate], ["branch", "loop"])
        self.assertIn("## Aggregate Table", summary)
        self.assertIn("| branch | 1 | -1.00 | 33.33 | 0 | 1 | 0 |", summary)
        self.assertIn("## Top Improvements", summary)
        self.assertIn("## Worsened Transitions", summary)

    def test_eval_objective_does_not_modify_batch_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_program_run(Path(tmp) / "branch", program="branch")
            correctness_path = run_dir / "states" / "S0000" / "batch_correctness.csv"
            before = correctness_path.read_text(encoding="utf-8")

            eval_batch_objectives(run_dir, objective="ir-inst-count")
            after = correctness_path.read_text(encoding="utf-8")

        self.assertEqual(after, before)
        self.assertIn("can_hard_fold", after)
        self.assertIn("true", after)

    def test_child_ir_path_uses_child_state_id_for_non_duplicate_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "memory"
            _write_multi_child_program_run(run_dir)

            eval_batch_objectives(run_dir, objective="ir-inst-count")
            rows = _read_csv(run_dir / "objective_signal.csv")

        self.assertEqual(
            [Path(row["child_ir_path"]).parts[-3:] for row in rows],
            [
                ("states", "S0001", "input.ll"),
                ("states", "S0002", "input.ll"),
                ("states", "S0003", "input.ll"),
            ],
        )
        self.assertEqual([row["child_state_id"] for row in rows], ["S0001", "S0002", "S0003"])
        self.assertEqual([row["ir_inst_after"] for row in rows], ["1", "2", "4"])

    def test_duplicate_child_ir_path_uses_duplicate_of_but_keeps_child_state_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "memory"
            _write_multi_child_program_run(run_dir, duplicate_children={"S0002": "S0001"})

            eval_batch_objectives(run_dir, objective="ir-inst-count")
            rows = _read_csv(run_dir / "objective_signal.csv")
            duplicate_row = next(row for row in rows if row["child_state_id"] == "S0002")

        self.assertEqual(Path(duplicate_row["child_ir_path"]).parts[-3:], ("states", "S0001", "input.ll"))
        self.assertEqual(duplicate_row["child_state_id"], "S0002")
        self.assertEqual(duplicate_row["ir_inst_after"], "1")

    def test_missing_child_ir_records_note_and_skips_numeric_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_program_run(Path(tmp) / "branch", program="branch")
            _write_csv(
                run_dir / "batch_state_transitions.csv",
                [
                    "program",
                    "parent_state_id",
                    "child_state_id",
                    "batch_id",
                    "batch_passes",
                    "batch_size",
                    "parent_hash",
                    "child_hash",
                    "is_duplicate",
                    "duplicate_of",
                    "validation_status",
                ],
                [
                    {
                        "program": "branch",
                        "parent_state_id": "S0000",
                        "child_state_id": "S9999",
                        "batch_id": "B0000",
                        "batch_passes": "instcombine",
                        "batch_size": "1",
                        "parent_hash": "parent-hash",
                        "child_hash": "missing",
                        "is_duplicate": "false",
                        "duplicate_of": "",
                        "validation_status": "all_permutations_same",
                    }
                ],
            )

            eval_batch_objectives(run_dir, objective="ir-inst-count")
            rows = _read_csv(run_dir / "objective_signal.csv")

        self.assertEqual(rows[0]["child_state_id"], "S9999")
        self.assertEqual(Path(rows[0]["child_ir_path"]).parts[-3:], ("states", "S9999", "input.ll"))
        self.assertEqual(rows[0]["ir_inst_before"], "")
        self.assertEqual(rows[0]["ir_inst_after"], "")
        self.assertEqual(rows[0]["ir_inst_delta"], "")
        self.assertEqual(rows[0]["ir_inst_reduction_pct"], "")
        self.assertIn("missing_child_ir", rows[0]["objective_note"])


def _write_program_run(run_dir: Path, program: str) -> Path:
    parent_dir = run_dir / "states" / "S0000"
    child_dir = run_dir / "states" / "S0001"
    parent_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    (parent_dir / "input.ll").write_text(
        "\n".join(
            [
                "define i32 @f(i32 %x) {",
                "entry:",
                "  %a = add i32 %x, 1",
                "  %b = mul i32 %a, 2",
                "  ret i32 %b",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (child_dir / "input.ll").write_text(
        "\n".join(
            [
                "define i32 @f(i32 %x) {",
                "entry:",
                "  %a = add i32 %x, 1",
                "  ret i32 %a",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "state_hash", "depth", "ir_path", "state_dir"],
        [
            {
                "program": program,
                "state_id": "S0000",
                "state_hash": "parent-hash",
                "depth": "0",
                "ir_path": str(parent_dir / "input.ll"),
                "state_dir": str(parent_dir),
            },
            {
                "program": program,
                "state_id": "S0001",
                "state_hash": "child-hash",
                "depth": "1",
                "ir_path": str(child_dir / "input.ll"),
                "state_dir": str(child_dir),
            },
        ],
    )
    _write_csv(
        run_dir / "batch_state_transitions.csv",
        [
            "program",
            "parent_state_id",
            "child_state_id",
            "batch_id",
            "batch_passes",
            "batch_size",
            "parent_hash",
            "child_hash",
            "is_duplicate",
            "duplicate_of",
            "validation_status",
        ],
        [
            {
                "program": program,
                "parent_state_id": "S0000",
                "child_state_id": "S0001",
                "batch_id": "B0000",
                "batch_passes": "instcombine;simplifycfg",
                "batch_size": "2",
                "parent_hash": "parent-hash",
                "child_hash": "child-hash",
                "is_duplicate": "false",
                "duplicate_of": "",
                "validation_status": "all_permutations_same",
            }
        ],
    )
    _write_csv(
        parent_dir / "batch_correctness.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "batch_id",
            "batch_passes",
            "batch_size",
            "validation_status",
            "correctness_class",
            "can_hard_fold",
            "can_execute",
            "reason",
        ],
        [
            {
                "program": program,
                "state_id": "S0000",
                "state_hash": "parent-hash",
                "batch_id": "B0000",
                "batch_passes": "instcombine;simplifycfg",
                "batch_size": "2",
                "validation_status": "all_permutations_same",
                "correctness_class": "certified_batch",
                "can_hard_fold": "true",
                "can_execute": "true",
                "reason": "",
            }
        ],
    )
    return run_dir


def _write_multi_child_program_run(run_dir: Path, duplicate_children: dict[str, str] | None = None) -> Path:
    duplicate_children = duplicate_children or {}
    parent_dir = run_dir / "states" / "S0000"
    parent_dir.mkdir(parents=True)
    (parent_dir / "input.ll").write_text(_ir_with_instruction_count(5), encoding="utf-8")
    child_counts = {"S0001": 1, "S0002": 2, "S0003": 4}
    for state_id, count in child_counts.items():
        if state_id in duplicate_children:
            continue
        child_dir = run_dir / "states" / state_id
        child_dir.mkdir(parents=True)
        (child_dir / "input.ll").write_text(_ir_with_instruction_count(count), encoding="utf-8")

    state_rows = [
        {
            "program": "memory",
            "state_id": "S0000",
            "state_hash": "h0",
            "depth": "0",
            "ir_path": str(parent_dir / "input.ll"),
            "state_dir": str(parent_dir),
            "is_duplicate": "false",
            "duplicate_of": "",
        }
    ]
    for state_id in ["S0001", "S0002", "S0003"]:
        actual_dir = run_dir / "states" / state_id
        # Deliberately poison S0002/S0003 metadata to ensure evaluator does not
        # reuse S0001 paths for non-duplicate transitions.
        listed_dir = run_dir / "states" / "S0001" if state_id in {"S0002", "S0003"} else actual_dir
        state_rows.append(
            {
                "program": "memory",
                "state_id": state_id,
                "state_hash": f"h{state_id[-1]}",
                "depth": "1",
                "ir_path": str(listed_dir / "input.ll"),
                "state_dir": str(listed_dir),
                "is_duplicate": "true" if state_id in duplicate_children else "false",
                "duplicate_of": duplicate_children.get(state_id, ""),
            }
        )
    _write_csv(run_dir / "states.csv", ["program", "state_id", "state_hash", "depth", "ir_path", "state_dir", "is_duplicate", "duplicate_of"], state_rows)
    _write_csv(
        run_dir / "batch_state_transitions.csv",
        [
            "program",
            "parent_state_id",
            "child_state_id",
            "batch_id",
            "batch_passes",
            "batch_size",
            "parent_hash",
            "child_hash",
            "is_duplicate",
            "duplicate_of",
            "validation_status",
        ],
        [
            {
                "program": "memory",
                "parent_state_id": "S0000",
                "child_state_id": state_id,
                "batch_id": f"B000{index}",
                "batch_passes": "instcombine",
                "batch_size": "1",
                "parent_hash": "h0",
                "child_hash": f"h{state_id[-1]}",
                "is_duplicate": "true" if state_id in duplicate_children else "false",
                "duplicate_of": duplicate_children.get(state_id, ""),
                "validation_status": "all_permutations_same",
            }
            for index, state_id in enumerate(["S0001", "S0002", "S0003"])
        ],
    )
    _write_csv(
        parent_dir / "batch_correctness.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "batch_id",
            "batch_passes",
            "batch_size",
            "validation_status",
            "correctness_class",
            "can_hard_fold",
            "can_execute",
            "reason",
        ],
        [
            {
                "program": "memory",
                "state_id": "S0000",
                "state_hash": "h0",
                "batch_id": f"B000{index}",
                "batch_passes": "instcombine",
                "batch_size": "1",
                "validation_status": "all_permutations_same",
                "correctness_class": "certified_batch",
                "can_hard_fold": "true",
                "can_execute": "true",
                "reason": "",
            }
            for index in range(3)
        ],
    )
    return run_dir


def _ir_with_instruction_count(count: int) -> str:
    lines = ["define i32 @f(i32 %x) {", "entry:"]
    for index in range(max(0, count - 1)):
        source = "%x" if index == 0 else f"%v{index - 1}"
        lines.append(f"  %v{index} = add i32 {source}, 1")
    value = "%x" if count <= 1 else f"%v{count - 2}"
    lines.append(f"  ret i32 {value}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
