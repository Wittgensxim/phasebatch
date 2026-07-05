import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phasebatch.pass_audit import audit_passes
from phasebatch.pass_config import load_pass_config
from phasebatch.schema import RunResult


class PassAuditTests(unittest.TestCase):
    def test_audit_accepts_string_only_pass_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = _write_ir(root / "input.ll", 3)
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - pass-a\n", encoding="utf-8")

            audit_passes(input_ll, passes, root / "out", tools=_tools(), opt_runner=_runner({"pass-a": "same"}))

            rows = _read_csv(root / "out" / "pass_audit.csv")
            self.assertEqual(rows[0]["pass"], "pass-a")
            self.assertEqual(rows[0]["resolved_pipeline"], "pass-a")
            self.assertEqual(rows[0]["valid_on_input"], "true")

    def test_rich_config_resolves_second_candidate_after_first_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = _write_ir(root / "input.ll", 3)
            passes = root / "passes.yaml"
            passes.write_text(
                "\n".join(
                    [
                        "passes:",
                        "  - name: licm",
                        "    pipeline_candidates:",
                        "      - licm",
                        "      - function(loop(licm))",
                        "    category: loop",
                        "    stage: v3",
                    ]
                ),
                encoding="utf-8",
            )

            audit_passes(
                input_ll,
                passes,
                root / "out",
                tools=_tools(),
                opt_runner=_runner({"licm": "fail", "function(loop(licm))": "changed"}),
            )

            [row] = _read_csv(root / "out" / "pass_audit.csv")
            self.assertEqual(row["candidate_index"], "1")
            self.assertEqual(row["candidate_pipeline"], "function(loop(licm))")
            self.assertEqual(row["resolved_pipeline"], "function(loop(licm))")
            self.assertEqual(row["recommended_action"], "needs_nested_pipeline")

    def test_invalid_pass_goes_to_invalid_passes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = _write_ir(root / "input.ll", 2)
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - bad-pass\n", encoding="utf-8")

            audit_passes(input_ll, passes, root / "out", tools=_tools(), opt_runner=_runner({"bad-pass": "fail"}))

            audit_rows = _read_csv(root / "out" / "pass_audit.csv")
            invalid_rows = _read_csv(root / "out" / "invalid_passes.csv")
            self.assertEqual(audit_rows[0]["valid_on_input"], "false")
            self.assertEqual(invalid_rows[0]["pass"], "bad-pass")
            self.assertEqual(invalid_rows[0]["attempted_candidates"], "bad-pass")

    def test_valid_but_dormant_pass_is_kept_in_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = _write_ir(root / "input.ll", 3)
            passes = root / "passes.yaml"
            passes.write_text(
                "\n".join(
                    [
                        "passes:",
                        "  - name: dormant-pass",
                        "    pipeline: dormant-pass",
                        "    category: scalar",
                        "    stage: v2",
                    ]
                ),
                encoding="utf-8",
            )

            audit_passes(input_ll, passes, root / "out", tools=_tools(), opt_runner=_runner({"dormant-pass": "same"}))

            [row] = _read_csv(root / "out" / "pass_audit.csv")
            self.assertEqual(row["active_on_input"], "false")
            self.assertEqual(row["recommended_action"], "keep_dormant")
            specs = load_pass_config(root / "out" / "resolved_passes.yaml")
            self.assertEqual([spec.name for spec in specs], ["dormant-pass"])
            self.assertEqual([spec.pipeline for spec in specs], ["dormant-pass"])

    def test_c_input_uses_prepare_input_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_c = root / "input.c"
            input_c.write_text("int f(void) { return 1; }\n", encoding="utf-8")
            prepared = _write_ir(root / "prepared.ll", 1)
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - pass-a\n", encoding="utf-8")

            with patch("phasebatch.pass_audit.prepare_input_ir", return_value=prepared) as prepare:
                audit_passes(input_c, passes, root / "out", tools=_tools(), opt_runner=_runner({"pass-a": "same"}))

            prepare.assert_called_once()
            self.assertTrue((root / "out" / "pass_audit_summary.md").exists())


def _tools() -> dict:
    return {"clang": "clang", "opt": "opt"}


def _runner(behavior_by_pipeline: dict[str, str]):
    def run(_opt: str, input_ll: Path, pipeline: str, output_ll: Path, _timeout: int) -> RunResult:
        behavior = behavior_by_pipeline.get(pipeline, "fail")
        if behavior == "fail":
            return RunResult(
                command=["opt", f"-passes={pipeline}"],
                returncode=1,
                stdout="",
                stderr=f"unknown pass {pipeline}",
                time_ms=1.0,
                failure_kind="nonzero_exit",
                output_path=output_ll,
            )
        if behavior == "same":
            output_ll.parent.mkdir(parents=True, exist_ok=True)
            output_ll.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            _write_ir(output_ll, 4)
        return RunResult(
            command=["opt", f"-passes={pipeline}"],
            returncode=0,
            stdout="",
            stderr="",
            time_ms=2.0,
            output_path=output_ll,
        )

    return run


def _write_ir(path: Path, instructions: int) -> Path:
    body = "\n".join(f"  %v{i} = add i32 {i}, {i}" for i in range(instructions))
    terminator = f"  ret i32 {'%v' + str(instructions - 1) if instructions else '0'}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"define i32 @f() {{\n{body}\n{terminator}\n}}\n", encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
